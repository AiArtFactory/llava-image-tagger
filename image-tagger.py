"""
Image Tagger - A tool for generating captions/tags using LLaVA models and managing tag files.

This script operates in three distinct modes:
1. TAG mode: Generate AI captions/tags for images (with options to append/prepend to existing)
2. EDIT mode: Modify existing tag files (prepend, append, remove tags manually)
3. LIST mode: Analyze and list tag frequencies

Usage examples:
    # TAG mode - generate new captions (overwrites existing)
    python image_tagger.py tag -i /path/to/images -s "Descriptive"

    # TAG mode - prepend AI captions to existing tags (creates hybrid datasets)
    python image_tagger.py tag -i /path/to/images -s "Descriptive" --prepend-to-existing -c "trigger_word"

    # TAG mode - append AI captions to existing tags
    python image_tagger.py tag -i /path/to/images -s "Descriptive" --append-to-existing

    # EDIT mode - manually modify existing tags
    python image_tagger.py edit -t /path/to/tags -a prepend -c "artist:name"

    # LIST mode - analyze tags
    python image_tagger.py list -t /path/to/tags
"""

import argparse
import gc
import json
import logging
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, LogitsProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_MODEL = "fancyfeast/llama-joycaption-beta-one-hf-llava"
VALID_CAPTION_TYPES = [
    "",
    "Descriptive",
    "Descriptive (Casual)",
    "Straightforward",
    "Stable Diffusion Prompt",
    "MidJourney",
    "Danbooru tag list",
    "e621 tag list",
    "Rule34 tag list",
    "Booru-like tag list",
    "Art Critic",
    "Product Listing",
    "Social Media Post",
]

CAPTION_PROMPT_TEMPLATES = {
    "Descriptive": [
        "Write a detailed description for this image.",
        "Write a {length} detailed description for this image.",
    ],
    "Descriptive (Casual)": [
        "Write a descriptive caption for this image in a casual tone.",
        "Write a {length} descriptive caption for this image in a casual tone.",
    ],
    "Straightforward": [
        "Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with 'This image is…' or similar phrasing.",
        "Write a {length} straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with 'This image is…' or similar phrasing.",
    ],
    "Stable Diffusion Prompt": [
        "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
        "Output a {length} stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
    ],
    "MidJourney": [
        "Write a MidJourney prompt for this image.",
        "Write a {length} MidJourney prompt for this image.",
    ],
    "Danbooru tag list": [
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text.",
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {length} length.",
    ],
    "e621 tag list": [
        "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
        "Write a {length} comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
    ],
    "Rule34 tag list": [
        "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with 'character:', and 'meta:'. Then all the general tags.",
        "Write a {length} comma-separated list of rule34 tags in alphabetical order for this image. Start with character, Then all the general tags.",
    ],
    "Booru-like tag list": [
        "Write a list of Booru-like tags for this image.",
        "Write a {length} list of Booru-like tags for this image.",
    ],
    "Art Critic": [
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
    ],
    "Product Listing": [
        "Write a caption for this image as though it were a product listing.",
        "Write a {length} caption for this image as though it were a product listing.",
    ],
    "Social Media Post": [
        "Write a caption for this image as if it were being used for a social media post.",
        "Write a {length} caption for this image as if it were being used for a social media post.",
    ],
}


# --- Device Selection ---
def get_device() -> str:
    """Determine the best available device."""
    if torch.backends.mps.is_available():
        logger.info("Using Apple MPS (Metal) for acceleration")
        return "mps"
    elif torch.cuda.is_available():
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return "cuda"
    else:
        logger.info("No GPU detected, using CPU (will be slow)")
        return "cpu"


DEVICE = get_device()


# --- Logits Processor ---
class SanitizeLogitsProcessor(LogitsProcessor):
    """Clamps inf/nan values in logits to prevent MPS sampling crashes."""

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(scores):
            scores = scores.float()
        scores = torch.where(
            torch.isnan(scores),
            torch.full_like(scores, -1e4),
            scores,
        )
        scores = torch.clamp(scores, min=-1e4, max=1e4)
        return scores


# --- Model Management ---
class ModelManager:
    """Manages the LLaVA model lifecycle."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.dtype = None

    def load(self, model_path: str, dtype_override: Optional[str] = None) -> None:
        """Load the model and processor."""
        if dtype_override:
            self.dtype = getattr(torch, dtype_override)
            logger.info(f"   Using dtype: {dtype_override} (user override)")
        elif DEVICE == "cuda":
            try:
                torch.zeros(1, dtype=torch.bfloat16).cuda()
                self.dtype = torch.bfloat16
                logger.info("   Using dtype: bfloat16 (CUDA optimal)")
            except Exception:
                self.dtype = torch.float32
                logger.info("   Using dtype: float32 (CUDA fallback)")
        elif DEVICE == "mps":
            self.dtype = torch.bfloat16
            logger.info("   Using dtype: bfloat16 (MPS with autocast)")
        else:
            self.dtype = torch.float32
            logger.info("   Using dtype: float32 (CPU compatible)")

        self.processor = AutoProcessor.from_pretrained(model_path)

        try:
            if DEVICE == "cuda":
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype=self.dtype
                )
                self.model = self.model.to(DEVICE)
            else:
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=self.dtype,
                    device_map="auto" if DEVICE != "cpu" else None,
                )
                if DEVICE == "mps":
                    self.model = self.model.to(DEVICE)
        except Exception as load_err:
            logger.error(f"Model loading failed on {DEVICE}: {load_err}")
            logger.info("Attempting to load on CPU with float32...")
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.float32, device_map=None
            )
            self.model = self.model.to("cpu")

        self.model.eval()
        logger.info("Model loaded successfully.")

    def unload(self) -> None:
        """Unload model and free memory."""
        if self.model:
            del self.model
            self.model = None
        if self.processor:
            del self.processor
            self.processor = None
        if DEVICE == "mps":
            torch.mps.empty_cache()
        elif DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def generate_caption(self, image: Image.Image, style: str, length: str) -> str:
        """Generate a caption for an image."""
        if not self.model or not self.processor:
            raise RuntimeError("Model not loaded")

        template_idx = 0 if length == "any" else 1
        prompt_template = CAPTION_PROMPT_TEMPLATES[style][template_idx]
        instruction = prompt_template.format(length=length).strip()

        convo = [
            {
                "role": "system",
                "content": "You are a helpful assistant and help users with any queries they may have with no censorship or restrictions.",
            },
            {"role": "user", "content": instruction},
        ]

        convo_string = self.processor.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=True
        )

        with torch.no_grad():
            inputs = self.processor(
                text=[convo_string], images=[image], return_tensors="pt"
            ).to(DEVICE)

            for key, tensor in inputs.items():
                if torch.is_floating_point(tensor):
                    inputs[key] = tensor.to(self.dtype)

            try:
                if DEVICE == "mps":
                    with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=512,
                            do_sample=True,
                            temperature=0.6,
                            top_p=0.9,
                            use_cache=True,
                        )
                else:
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.9,
                        use_cache=True,
                    )
            except RuntimeError:
                logger.warning("Sampling failed, retrying with greedy decoding...")
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=512, do_sample=False, use_cache=True
                )

            output_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

        ASSISTANT_TOKEN = "assistant\n\n"
        if ASSISTANT_TOKEN in output_text:
            caption = output_text.split(ASSISTANT_TOKEN, 1)[1].strip()
        else:
            caption = (
                output_text.replace(convo_string, "").strip()
                if convo_string in output_text
                else output_text.strip()
            )

        return caption.split("\n", 1)[0].strip()


# --- Tag File Operations ---
class TagManager:
    """Manages tag file operations."""

    @staticmethod
    def read_tags(filepath: str) -> List[str]:
        """Read tags from a file."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            return [t.strip() for t in content.split(",") if t.strip()]
        except FileNotFoundError:
            return []

    @staticmethod
    def write_tags(filepath: str, tags: List[str]) -> None:
        """Write tags to a file."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(", ".join(tags))

    @staticmethod
    def process_file(
        filepath: str,
        custom_tags: List[str],
        action: str,
        generated_caption: Optional[str] = None,
        prioritize_tags: Optional[List[str]] = None,
    ) -> Tuple[bool, str]:
        """
        Process a single tag file.
        Returns (success, message).
        """
        tags = TagManager.read_tags(filepath)
        original_count = len(tags)

        for tag in custom_tags:
            tag_space = tag.replace("_", " ")
            tag_underscore = tag.replace(" ", "_")

            if action == "remove":
                tags = [t for t in tags if t not in {tag, tag_space, tag_underscore}]
            elif action in ("prepend", "append"):
                variants = {tag, tag_space, tag_underscore}
                if not any(v in tags for v in variants):
                    if action == "prepend":
                        tags.insert(0, tag)
                    else:
                        tags.append(tag)

        if action == "generate-prepend" and generated_caption:
            # Insert the generated caption at the beginning
            tags.insert(0, generated_caption)

        # Handle prioritize_tags - move to front if found anywhere in tags
        if prioritize_tags:
            for prio_tag in reversed(prioritize_tags):
                tag_space = prio_tag.replace("_", " ")
                tag_underscore = prio_tag.replace(" ", "_")
                variants = {prio_tag, tag_space, tag_underscore}

                # Remove all instances from current positions
                found = any(v in tags for v in variants)
                if found:
                    tags = [t for t in tags if t not in variants]
                    # Insert at the very front
                    tags.insert(0, prio_tag)

        if (
            len(tags) != original_count
            or action in ("prepend", "append")
            or generated_caption
            or prioritize_tags
        ):
            TagManager.write_tags(filepath, tags)
            if action == "generate-prepend":
                return (
                    True,
                    f"Updated {os.path.basename(filepath)} (prepended AI caption)",
                )
            return True, f"Updated {os.path.basename(filepath)}"
        return False, f"No changes for {os.path.basename(filepath)}"

    @staticmethod
    def collect_counts(root_dir: str, extension: str) -> Counter:
        """Collect tag frequencies from all files in a directory."""
        counter = Counter()
        for dirpath, _, filenames in os.walk(root_dir):
            for fn in filenames:
                if fn.endswith(extension):
                    full_path = os.path.join(dirpath, fn)
                    try:
                        tags = TagManager.read_tags(full_path)
                        counter.update(tags)
                    except Exception:
                        continue
        return counter


# --- Tag Generation ---
class TagGenerator:
    """Handles AI-powered tag generation."""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def process_image(
        self,
        image_path: str,
        styles: List[str],
        length: str,
        dry_run: bool = False,
        write_file: bool = True,
        append_to_existing: bool = False,
        prepend_to_existing: bool = False,
        prioritize_tags: Optional[List[str]] = None,
    ) -> Tuple[str, bool]:
        """Process a single image and generate captions."""
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error opening image {image_path}: {e}")
            return "", False

        captions = []
        for style in styles:
            try:
                caption = self.model_manager.generate_caption(image, style, length)
                captions.append(caption)
            except Exception as e:
                logger.error(f"Error generating {style} caption: {e}")
                captions.append("")

        if not any(captions):
            logger.error(f"No captions generated for {os.path.basename(image_path)}")
            return "", False

        final_caption = ", ".join(filter(None, captions))

        if write_file:
            base_path, _ = os.path.splitext(image_path)
            caption_path = base_path + ".txt"

            if dry_run:
                logger.info(f"[DRY RUN] Would save to {caption_path}")
                logger.info(f"Preview: {final_caption[:100]}...")
            else:
                # Handle append/prepend modes
                if append_to_existing and os.path.exists(caption_path):
                    existing_tags = TagManager.read_tags(caption_path)
                    combined_tags = existing_tags + [final_caption]
                    TagManager.write_tags(caption_path, combined_tags)
                    logger.info(f"Saved (appended): {caption_path}")
                elif prepend_to_existing and os.path.exists(caption_path):
                    existing_tags = TagManager.read_tags(caption_path)
                    # Prepend caption first
                    combined_tags = [final_caption] + existing_tags

                    # Handle prioritize_tags - move to front
                    if prioritize_tags:
                        for prio_tag in reversed(prioritize_tags):
                            tag_space = prio_tag.replace("_", " ")
                            tag_underscore = prio_tag.replace(" ", "_")
                            variants = {prio_tag, tag_space, tag_underscore}

                            # Find and remove from current position
                            found = any(v in combined_tags for v in variants)
                            if found:
                                combined_tags = [
                                    t for t in combined_tags if t not in variants
                                ]
                                combined_tags.insert(0, prio_tag)

                    TagManager.write_tags(caption_path, combined_tags)
                    logger.info(f"Saved (prepended): {caption_path}")
                else:
                    TagManager.write_tags(caption_path, [final_caption])
                    logger.info(f"Saved: {caption_path}")

        return final_caption, True

    def process_directory(
        self,
        directory: str,
        styles: List[str],
        length: str,
        dry_run: bool = False,
        recursive: bool = True,
        append_to_existing: bool = False,
        prepend_to_existing: bool = False,
        prioritize_tags: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Process all images in a directory."""
        extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")
        image_files = []

        if recursive:
            for root, _, files in os.walk(directory):
                for f in files:
                    if f.lower().endswith(extensions):
                        image_files.append(os.path.join(root, f))
        else:
            for f in os.listdir(directory):
                if f.lower().endswith(extensions):
                    image_files.append(os.path.join(directory, f))

        if not image_files:
            logger.error(f"No images found in {directory}")
            return {}

        logger.info(f"Found {len(image_files)} images to process")
        results = {}

        # Create progress bar with ETA
        progress_bar = tqdm(
            total=len(image_files),
            desc="Processing images",
            unit="img",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        start_time = time.time()

        for i, img_path in enumerate(image_files):
            caption, success = self.process_image(
                img_path,
                styles,
                length,
                dry_run,
                append_to_existing=append_to_existing,
                prepend_to_existing=prepend_to_existing,
                prioritize_tags=prioritize_tags,
            )
            if success:
                results[img_path] = caption

            # Update progress bar
            progress_bar.update(1)

            # Memory cleanup every 10 images
            if (i + 1) % 10 == 0:
                gc.collect()
                if DEVICE == "mps":
                    torch.mps.empty_cache()
                elif DEVICE == "cuda":
                    torch.cuda.empty_cache()

        progress_bar.close()

        elapsed_time = time.time() - start_time
        avg_time_per_image = elapsed_time / len(image_files) if image_files else 0
        logger.info(
            f"Processed {len(results)}/{len(image_files)} images successfully "
            f"({elapsed_time:.1f}s total, {avg_time_per_image:.2f}s per image)"
        )

        return results


# --- CLI Setup ---
def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="Image Tagger: Generate AI captions and manage tag files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # TAG mode - generate AI captions
  python image_tagger.py tag -i /path/to/images -s "Descriptive"

  # EDIT mode - modify existing tags
  python image_tagger.py edit -t /path/to/tags --action prepend --custom-tag "artist:name"

  # LIST mode - analyze tag frequencies
  python image_tagger.py list -t /path/to/tags --sort-descending

  # ORGANIZE mode - sort by parenthetical concept tags
  python image_tagger.py organize -i /path/to/dataset --dry-run
  python image_tagger.py organize -i /path/to/dataset -o ./organized
        """,
    )

    subparsers = parser.add_subparsers(dest="mode", help="Operating mode")

    # === TAG mode ===
    tag_parser = subparsers.add_parser(
        "tag", help="Generate AI captions/tags for images (requires --image-path)"
    )
    tag_parser.add_argument(
        "-i", "--image-path", required=True, help="Path to image file or directory"
    )
    tag_parser.add_argument(
        "-s",
        "--caption-style",
        action="append",
        required=True,
        help=f"Caption style(s). Valid: {', '.join(VALID_CAPTION_TYPES[1:])}",
    )
    tag_parser.add_argument(
        "-l",
        "--caption-length",
        choices=["any", "very short", "short", "medium-length", "long", "very long"],
        default="long",
        help="Caption length preference",
    )
    tag_parser.add_argument(
        "-m", "--model-path", default=DEFAULT_MODEL, help="Model path or HuggingFace ID"
    )
    tag_parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        help="Override model dtype (default: auto-select)",
    )
    tag_parser.add_argument(
        "--dry-run", action="store_true", help="Preview without writing files"
    )
    tag_parser.add_argument(
        "--no-recursive", action="store_true", help="Don't process subdirectories"
    )
    tag_parser.add_argument(
        "--append-to-existing",
        action="store_true",
        help="Append generated captions to existing tag files instead of overwriting",
    )
    tag_parser.add_argument(
        "--prepend-to-existing",
        action="store_true",
        help="Prepend generated captions to existing tag files instead of overwriting",
    )
    tag_parser.add_argument(
        "-c",
        "--trigger-word",
        help="Comma-separated trigger word(s) to prioritize to front (only with --prepend-to-existing)",
    )

    # === EDIT mode ===
    edit_parser = subparsers.add_parser(
        "edit", help="Modify existing tag files (prepend, append, remove)"
    )
    edit_parser.add_argument(
        "-t", "--tag-dir", required=True, help="Directory containing tag files"
    )
    edit_parser.add_argument(
        "-a",
        "--action",
        choices=["prepend", "append", "remove"],
        required=True,
        help="Tag modification action (prepend: add to front, append: add to end, remove: delete from file)",
    )
    edit_parser.add_argument(
        "-c",
        "--custom-tag",
        required=True,
        help="Comma-separated tags to add/remove",
    )
    edit_parser.add_argument(
        "-e", "--extension", choices=[".txt", ".caption"], default=".txt"
    )
    edit_parser.add_argument(
        "-r", "--recursive", action="store_true", help="Process subdirectories"
    )
    edit_parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without writing"
    )

    # === LIST mode ===
    list_parser = subparsers.add_parser("list", help="List and analyze tag frequencies")
    list_parser.add_argument(
        "-t", "--tag-dir", required=True, help="Directory containing tag files"
    )
    list_parser.add_argument(
        "-e", "--extension", choices=[".txt", ".caption"], default=".txt"
    )
    list_parser.add_argument(
        "--sort-descending",
        action="store_true",
        help="Sort by frequency (highest first)",
    )
    list_parser.add_argument(
        "-o", "--output", help="Output file path (default: tags_with_counts.txt)"
    )

    # === ORGANIZE mode ===
    organize_parser = subparsers.add_parser(
        "organize",
        help="Organize images into concept folders based on hierarchical parenthetical tags",
        description="""
Organize images by analyzing parenthetical tags in the format 'TagName \\(Qualifier\\)'.

IMPORTANT FORMAT REQUIREMENTS:
  - Tags MUST be formatted EXACTLY as: 'TagName \\(Qualifier\\)'
  - The backslashes before parentheses are REQUIRED
  - Tags without this exact format will be IGNORED
  - Generic tags (1girl, solo, from behind, etc.) are ignored

This mode is designed for datasets using Danbooru-style hierarchical tags where concepts
are represented as: 'ConceptName \\(CharacterName\\)' - for example:
  - Sweater School Uniform \\(Misuzu Gundou\\)
  - Casual Clothes \\(misuzu gundou\\)
  - Gym Clothes \\(Misuzu Gundou\\)

The mode will:
  1. Extract all tags matching the \\(...\\) format
  2. Identify universal tags (present in ALL files) -> stored in _common/
  3. Group remaining tags by concept -> create folders for each
  4. Copy images + tag files to appropriate concept folders

Both Japanese and English concept tags are supported and will be merged.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    organize_parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
        help="Directory containing images and tag files to organize",
    )
    organize_parser.add_argument(
        "-o",
        "--output-dir",
        default="./organized",
        help="Output directory for organized dataset (default: ./organized)",
    )
    organize_parser.add_argument(
        "--min-frequency",
        type=int,
        default=3,
        help="Minimum number of images required to create a concept folder (default: 3)",
    )
    organize_parser.add_argument(
        "-e",
        "--extension",
        choices=[".txt", ".caption"],
        default=".txt",
        help="Extension of tag files",
    )
    organize_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview organization without copying files",
    )

    return parser


# --- Command Handlers ---
def handle_tag_mode(args: argparse.Namespace) -> int:
    """Handle the 'tag' subcommand."""
    # Validate image path
    if not os.path.exists(args.image_path):
        logger.error(f"Image path does not exist: {args.image_path}")
        return 1

    # Normalize caption styles
    styles = []
    for style in args.caption_style:
        normalized = style.strip()
        matched = None
        for valid in VALID_CAPTION_TYPES:
            if valid.lower() == normalized.lower():
                matched = valid
                break
        if matched:
            styles.append(matched)
        else:
            logger.warning(f"Unknown style '{style}', skipping")

    if not styles:
        logger.error("No valid caption styles specified")
        return 1

    logger.info(f"Caption styles: {', '.join(styles)}")
    logger.info(f"Caption length: {args.caption_length}")

    # Check if local model path exists
    is_local = os.path.isabs(args.model_path) or os.path.exists(args.model_path)
    if is_local and not os.path.exists(args.model_path):
        logger.error(f"Model path not found: {args.model_path}")
        return 1

    # Load model
    model_manager = ModelManager()
    model_manager.load(args.model_path, args.dtype)

    # Parse trigger words if provided
    trigger_words = []
    if args.trigger_word:
        trigger_words = [t.strip() for t in args.trigger_word.split(",") if t.strip()]
        if trigger_words:
            logger.info(f"Trigger words to prioritize: {', '.join(trigger_words)}")

    try:
        generator = TagGenerator(model_manager)

        if os.path.isfile(args.image_path):
            logger.info(f"Processing single file: {args.image_path}")
            caption, success = generator.process_image(
                args.image_path,
                styles,
                args.caption_length,
                args.dry_run,
                append_to_existing=getattr(args, "append_to_existing", False),
                prepend_to_existing=getattr(args, "prepend_to_existing", False),
                prioritize_tags=trigger_words
                if getattr(args, "prepend_to_existing", False)
                else None,
            )
            return 0 if success else 1
        else:
            logger.info(f"Processing directory: {args.image_path}")
            results = generator.process_directory(
                args.image_path,
                styles,
                args.caption_length,
                args.dry_run,
                recursive=not args.no_recursive,
                append_to_existing=getattr(args, "append_to_existing", False),
                prepend_to_existing=getattr(args, "prepend_to_existing", False),
                prioritize_tags=trigger_words
                if getattr(args, "prepend_to_existing", False)
                else None,
            )
            logger.info(f"Processed {len(results)} images successfully")
            return 0 if results else 1
    finally:
        model_manager.unload()


def handle_edit_mode(args: argparse.Namespace) -> int:
    """Handle the 'edit' subcommand - for modifying existing tag files."""
    if not os.path.isdir(args.tag_dir):
        logger.error(f"Tag directory not found: {args.tag_dir}")
        return 1

    # Parse custom tags
    if not args.custom_tag:
        logger.error("--custom-tag is required")
        return 1

    custom_tags = [t.strip() for t in args.custom_tag.split(",") if t.strip()]
    if not custom_tags:
        logger.error("No custom tags provided")
        return 1

    logger.info(f"Tag directory: {args.tag_dir}")
    logger.info(f"Action: {args.action}")
    logger.info(f"Tags: {', '.join(custom_tags)}")

    # Find files to process
    files_to_process = []
    if args.recursive:
        for root, _, files in os.walk(args.tag_dir):
            for f in files:
                if f.endswith(args.extension):
                    files_to_process.append(os.path.join(root, f))
    else:
        for f in os.listdir(args.tag_dir):
            if f.endswith(args.extension):
                files_to_process.append(os.path.join(args.tag_dir, f))

    if not files_to_process:
        logger.error(f"No {args.extension} files found in {args.tag_dir}")
        return 1

    logger.info(f"Found {len(files_to_process)} files to process")

    if args.dry_run:
        logger.info("[DRY RUN] Would modify these files:")
        for f in files_to_process:
            logger.info(f"  - {f}")
        return 0

    updated = 0
    progress_bar = tqdm(
        total=len(files_to_process),
        desc=f"Processing {args.action}",
        unit="file",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    start_time = time.time()

    for filepath in files_to_process:
        success, msg = TagManager.process_file(filepath, custom_tags, args.action)
        if success:
            updated += 1
            logger.debug(f"  {msg}")
        else:
            logger.debug(f"  {msg}")
        progress_bar.update(1)

    progress_bar.close()

    elapsed_time = time.time() - start_time
    logger.info(
        f"Updated {updated}/{len(files_to_process)} files successfully ({elapsed_time:.1f}s)"
    )
    return 0


def handle_list_mode(args: argparse.Namespace) -> int:
    """Handle the 'list' subcommand."""
    if not os.path.isdir(args.tag_dir):
        logger.error(f"Tag directory not found: {args.tag_dir}")
        return 1

    logger.info(f"Analyzing tags in: {args.tag_dir}")

    counts = TagManager.collect_counts(args.tag_dir, args.extension)

    if not counts:
        logger.warning(f"No {args.extension} files found or no tags extracted")
        return 1

    sorted_items = sorted(
        counts.items(), key=lambda kv: kv[1], reverse=args.sort_descending
    )

    # Display results
    logger.info(f"Found {len(sorted_items)} unique tags:")
    for tag, cnt in sorted_items:
        print(f"{tag}: {cnt}")

    # Save to file
    output_path = args.output or os.path.join(os.getcwd(), "tags_with_counts.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for tag, cnt in sorted_items:
            f.write(f"{tag}: {cnt}\n")
    logger.info(f"Saved results to: {output_path}")

    return 0


def handle_organize_mode(args: argparse.Namespace) -> int:
    r"""Handle the 'organize' subcommand - organize images by parenthetical concept tags.

    This mode analyzes tag files for tags in the format `tagname \(qualifier\)`
    (parenthetical tags with backslash-escaped parentheses) and organizes images
    into concept-specific folders based on these tags.

    IMPORTANT: This mode ONLY recognizes tags formatted EXACTLY as:
        TagName \\(Qualifier\\)

    Tags without this exact format (including parentheses without backslashes)
    will be ignored for organization purposes. Generic tags like "1girl",
    "solo", "from behind" are also ignored.

    The organization process:
    1. Scans all tag files for parenthetical tags
    2. Identifies "universal" tags (present in all files) -> stored in _common/
    3. Groups remaining tags by concept (e.g., outfits, styles)
    4. Creates folders for each concept with matching images
    5. Copies both image and tag file to concept folders
    """
    if not os.path.isdir(args.input_dir):
        logger.error(f"Input directory not found: {args.input_dir}")
        return 1

    # Find all tag files
    tag_files = []
    image_extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif")

    for f in os.listdir(args.input_dir):
        if f.endswith(args.extension):
            tag_files.append(os.path.join(args.input_dir, f))

    if not tag_files:
        logger.error(f"No {args.extension} files found in {args.input_dir}")
        return 1

    logger.info(f"Found {len(tag_files)} tag files to analyze")

    # Step 1: Extract all parenthetical tags from all files
    # Pattern: tag \(qualifier\) with literal backslashes
    all_file_tags = {}  # filepath -> set of tags
    tag_frequency = {}  # tag -> count

    for tag_file in tag_files:
        try:
            with open(tag_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse tags (comma-separated)
            tags = [t.strip() for t in content.split(",") if t.strip()]

            # Filter for parenthetical tags with backslashes: Tag \(Qualifier\)
            parenthetical_tags = set()
            for tag in tags:
                # Match pattern: anything \(anything\)
                # Using regex to find tags with \(...\)
                if "\\(" in tag and "\\)" in tag:
                    parenthetical_tags.add(tag)
                    tag_frequency[tag] = tag_frequency.get(tag, 0) + 1

            all_file_tags[tag_file] = parenthetical_tags

        except Exception as e:
            logger.warning(f"Error reading {tag_file}: {e}")
            continue

    if not all_file_tags:
        logger.error("No valid parenthetical tags found in any files")
        logger.error(
            "Note: Tags must be formatted as 'TagName \\(Qualifier\\)' with backslashes"
        )
        return 1

    # Step 2: Identify universal tags (present in all files)
    total_files = len(all_file_tags)
    universal_tags = set()

    for tag, count in tag_frequency.items():
        if count == total_files:
            universal_tags.add(tag)
            logger.info(f"Universal tag: {tag}")

    # Step 3: Group remaining tags by concept
    # Concept key extraction: normalize the tag to create folder names
    concept_groups = {}  # concept_key -> {files: [], tags: set()}

    for tag_file, tags in all_file_tags.items():
        # Skip universal tags
        concept_tags = tags - universal_tags

        for tag in concept_tags:
            # Extract concept name from tag
            # Tag format: "ConceptName \\(CharacterName\\)"
            # We want to group by concept, not by character
            concept_key = extract_concept_key(tag)

            if concept_key not in concept_groups:
                concept_groups[concept_key] = {"files": [], "tags": set()}

            concept_groups[concept_key]["files"].append(tag_file)
            concept_groups[concept_key]["tags"].add(tag)

    # Step 4: Filter concepts by min_frequency
    valid_concepts = {}
    for concept_key, data in concept_groups.items():
        unique_files = len(set(data["files"]))
        if unique_files >= args.min_frequency:
            valid_concepts[concept_key] = data
            logger.info(f"Concept '{concept_key}': {unique_files} images")

    if not valid_concepts:
        logger.error(f"No concepts found with at least {args.min_frequency} images")
        return 1

    # Step 5: Create output structure
    if args.dry_run:
        logger.info("[DRY RUN] Would create the following structure:")
        logger.info(f"  {args.output_dir}/")
        logger.info(f"  ├── _common/")
        logger.info(f"  │   └── common_tags.txt")

        for concept_key in sorted(valid_concepts.keys()):
            file_count = len(set(valid_concepts[concept_key]["files"]))
            logger.info(f"  ├── {concept_key}/ ({file_count} images)")

        # Count unmatched files
        matched_files = set()
        for data in valid_concepts.values():
            matched_files.update(data["files"])
        unmatched_count = len(set(all_file_tags.keys()) - matched_files)

        if unmatched_count > 0:
            logger.info(f"  └── _unmatched/ ({unmatched_count} images)")

        return 0

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create _common folder with universal tags
    common_dir = os.path.join(args.output_dir, "_common")
    os.makedirs(common_dir, exist_ok=True)

    common_tags_file = os.path.join(common_dir, "common_tags.txt")
    with open(common_tags_file, "w", encoding="utf-8") as f:
        f.write(", ".join(sorted(universal_tags)))
    logger.info(f"Created _common/ with {len(universal_tags)} universal tags")

    # Create concept folders and copy files
    processed_files = set()

    for concept_key in tqdm(
        sorted(valid_concepts.keys()), desc="Creating concept folders"
    ):
        data = valid_concepts[concept_key]
        concept_dir = os.path.join(args.output_dir, concept_key)
        os.makedirs(concept_dir, exist_ok=True)

        # Copy unique files to concept folder
        for tag_file in set(data["files"]):
            if tag_file in processed_files:
                continue  # Skip if already processed (multi-concept images go to first concept)

            # Get base filename
            base_name = os.path.splitext(os.path.basename(tag_file))[0]

            # Find corresponding image file
            image_file = None
            for ext in image_extensions:
                for case_ext in [ext, ext.upper()]:
                    potential_img = os.path.join(args.input_dir, base_name + case_ext)
                    if os.path.exists(potential_img):
                        image_file = potential_img
                        break
                if image_file:
                    break

            if not image_file:
                logger.warning(f"No image found for {tag_file}, skipping")
                continue

            # Copy tag file
            dest_tag = os.path.join(concept_dir, base_name + args.extension)
            shutil.copy2(tag_file, dest_tag)

            # Copy image file
            dest_img = os.path.join(
                concept_dir, base_name + os.path.splitext(image_file)[1]
            )
            shutil.copy2(image_file, dest_img)

            processed_files.add(tag_file)

    # Create _unmatched folder for files not in any concept
    unmatched_files = set(all_file_tags.keys()) - processed_files
    if unmatched_files:
        unmatched_dir = os.path.join(args.output_dir, "_unmatched")
        os.makedirs(unmatched_dir, exist_ok=True)

        for tag_file in unmatched_files:
            base_name = os.path.splitext(os.path.basename(tag_file))[0]

            # Find corresponding image
            image_file = None
            for ext in image_extensions:
                for case_ext in [ext, ext.upper()]:
                    potential_img = os.path.join(args.input_dir, base_name + case_ext)
                    if os.path.exists(potential_img):
                        image_file = potential_img
                        break
                if image_file:
                    break

            if image_file:
                dest_tag = os.path.join(unmatched_dir, base_name + args.extension)
                dest_img = os.path.join(
                    unmatched_dir, base_name + os.path.splitext(image_file)[1]
                )
                shutil.copy2(tag_file, dest_tag)
                shutil.copy2(image_file, dest_img)

        logger.info(f"Created _unmatched/ with {len(unmatched_files)} files")

    logger.info(
        f"Organization complete: {len(valid_concepts)} concepts, {len(processed_files)} images organized"
    )
    return 0


# Mapping of Japanese concepts to English concept keys
CONCEPT_NAME_MAPPING = {
    # Japanese outfit names -> English concept keys
    "セーターの学校制服": "sweater_school_uniform",
    "カジュアル": "casual_clothes",
    "体操着": "gym_clothes",
    "水着": "swimsuit",
    "スイムウェア": "swimsuit",
    "緑ジャケットの高校制服": "green_jacket_highschool_uniform",
    "ラーメン屋の制服": "ramen_shop_uniform",
    "セラフク": "serafuku",
    "せらふく": "serafuku",
    "みすずちゃん": "kid_gundou",
    "群堂ちゃん": "kid_gundou",
}


def extract_concept_key(tag: str) -> str:
    """Extract concept key from a parenthetical tag.

    Input: "Sweater School Uniform \\(Misuzu Gundou\\)"
    Output: "sweater_school_uniform"

    Input: "セーターの学校制服 \\(群堂 みすず\\)"
    Output: "sweater_school_uniform" (normalized to English if possible)
    """
    # Remove the parenthetical part
    if "\\(" in tag:
        concept_part = tag.split("\\(")[0].strip()
    else:
        concept_part = tag.strip()

    # Check if this Japanese concept has an English mapping
    if concept_part in CONCEPT_NAME_MAPPING:
        return CONCEPT_NAME_MAPPING[concept_part]

    # Otherwise normalize to lowercase, replace spaces with underscores
    concept_key = re.sub(r"[^\w\s-]", "", concept_part).lower().strip()
    concept_key = re.sub(r"[-\s]+", "_", concept_key)

    return concept_key


# --- Main Entry Point ---
def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        return 1

    if args.mode == "tag":
        return handle_tag_mode(args)
    elif args.mode == "edit":
        return handle_edit_mode(args)
    elif args.mode == "list":
        return handle_list_mode(args)
    elif args.mode == "organize":
        return handle_organize_mode(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
