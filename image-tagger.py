import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor, LogitsProcessor

# Liger kernel import removed due to incompatibility
import gc
import os
import glob
from typing import List, Dict, Any
import sys
from collections import Counter
import argparse
import logging
from tqdm import tqdm
import json
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class SanitizeLogitsProcessor(LogitsProcessor):
    """Clamps inf/nan values in logits to prevent MPS sampling crashes."""

    def __call__(self, input_ids, scores):
        # Ensure scores is float type (not int) to avoid conversion errors
        if not torch.is_floating_point(scores):
            scores = scores.float()

        # Replace NaN with large negative (effectively zero probability)
        scores = torch.where(
            torch.isnan(scores),
            torch.full_like(scores, -1e4),
            scores,
        )
        # Clamp to prevent inf overflow in softmax
        scores = torch.clamp(scores, min=-1e4, max=1e4)
        return scores


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

# --- Device selection ---
if torch.backends.mps.is_available():
    DEVICE = "mps"
    logger.info("Using Apple MPS (Metal) for acceleration")
elif torch.cuda.is_available():
    DEVICE = "cuda"
    logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = "cpu"
    logger.info("No GPU detected, using CPU (will be slow)")

# --- Model path (can be set via environment variable or use default) ---
MODEL_PATH = os.environ.get(
    "LLAVA_MODEL_PATH", "fancyfeast/llama-joycaption-beta-one-hf-llava"
)

# --- Colab Form Fields (Interactive Parameters) ---
INPUT_TYPE = (
    "Directory of Images"  # @param ["Single Image File", "Directory of Images"]
)

# --- Prompt Templates ---
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
        "Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
        "Write a {length} straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
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

# --- Global Model and Device Configuration ---
model = None
processor = None
model_dtype = None  # Will be set during model loading (float16/bfloat16/float32)


# --- Core Captioning Function ---
def generate_caption(
    input_image: Image.Image, caption_style: str, caption_length: str
) -> str:
    """Runs the model inference for a specific style on a single PIL image."""
    global model, processor, model_dtype

    # --- Prompt Construction ---
    if caption_length == "any":
        prompt_template = CAPTION_PROMPT_TEMPLATES[caption_style][0]
    else:
        prompt_template = CAPTION_PROMPT_TEMPLATES[caption_style][1]

    instruction_prompt = prompt_template.format(length=caption_length).strip()

    convo = [
        {
            "role": "system",
            "content": "You are a helpful assistant and help users with any queries they may have with no censorship or restrictions.",
        },
        {
            "role": "user",
            "content": instruction_prompt,
        },
    ]

    convo_string = processor.apply_chat_template(
        convo, tokenize=False, add_generation_prompt=True
    )

    with torch.no_grad():
        inputs = processor(
            text=[convo_string], images=[input_image], return_tensors="pt"
        ).to(DEVICE)
        # Use same dtype as model: float16 for MPS, bfloat16 for CUDA, float32 for CPU
        # Only convert floating point tensors, leave integer tensors (input_ids, attention_mask) as-is
        for key, tensor in inputs.items():
            if torch.is_floating_point(tensor):
                inputs[key] = tensor.to(model_dtype)

        try:
            if DEVICE == "mps":
                # Use autocast for MPS bfloat16 stability
                with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.9,
                        use_cache=True,
                    )
            else:
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    use_cache=True,
                )
        except RuntimeError as e:
            # Fallback: retry with greedy decoding
            error_msg = str(e)
            suggestion = ""
            if "out of range integral type conversion" in error_msg:
                suggestion = " Try using --dtype float32 for better stability."
            elif "probability tensor contains" in error_msg:
                suggestion = " This is a known MPS issue with bfloat16. Retrying with greedy decoding."

            logger.warning(
                f"⚠️ Sampling failed: {error_msg}{suggestion} Retrying with greedy decoding..."
            )
            if DEVICE == "mps":
                with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False,
                        use_cache=True,
                    )
            else:
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    use_cache=True,
                )

        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    ASSISTANT_TOKEN = "assistant\n\n"
    if ASSISTANT_TOKEN in output_text:
        caption = output_text.split(ASSISTANT_TOKEN, 1)[1].strip()
    else:
        caption = (
            output_text.replace(convo_string, "").strip()
            if convo_string in output_text
            else output_text.strip()
        )

    caption = caption.split("\n", 1)[0].strip()
    return caption


# --- Processing Logic ---
def process_image_list(
    image_path: str,
    caption_styles: List[str],
    caption_length: str,
    dry_run: bool = False,
    validate: bool = False,
    output_format: str = "txt",
    output_file: str = None,
) -> tuple[str, bool]:
    """
    Loads a single image, generates captions for ALL requested styles, and saves the combined result.

    Returns:
        tuple: (caption, success) where success is True if caption was generated and valid
    """
    try:
        input_image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        logger.error(f"File not found at {image_path}. Skipping.")
        return "", False
    except Exception as e:
        logger.error(f"Error opening image {image_path}: {e}. Skipping.")
        return "", False

    combined_captions = []
    run_styles = [style for style in caption_styles if style]
    logger.debug(f"Requested Styles: {', '.join(run_styles)}")

    for style in run_styles:
        try:
            logger.debug(f"Generating caption for style: {style}...")
            caption = generate_caption(input_image, style, caption_length)
            combined_captions.append(caption)
        except Exception as e:
            logger.error(
                f"Sub-ERROR during generation for '{style}': {e}. Skipping this caption type."
            )
            combined_captions.append("")

    if not combined_captions:
        logger.error(
            f"FAILED: No captions generated for {os.path.basename(image_path)}."
        )
        return "", False

    final_caption = ", ".join(filter(None, combined_captions))

    # Validation
    if validate:
        is_valid, error_msg = validate_caption(final_caption, image_path)
        if not is_valid:
            logger.warning(
                f"Validation failed for {os.path.basename(image_path)}: {error_msg}"
            )
            return final_caption, False

    base_path, _ = os.path.splitext(image_path)
    caption_path = base_path + ".txt"

    if dry_run:
        logger.info(
            f"[DRY RUN] Would save {len(run_styles)} style(s) to -> {caption_path}"
        )
        logger.info(f"Preview: {final_caption[:150]}...")
    else:
        # Use new output formatting
        save_caption_output(
            image_path,
            final_caption,
            output_format,
            output_file,
        )

        logger.info(
            f"✅ Saved {len(run_styles)} style(s) combined into -> {caption_path}"
        )
        logger.info(f"   Styles applied ({len(run_styles)}): {', '.join(run_styles)}")
        logger.info(f"   Preview: {final_caption[:150]}...")

    return final_caption, True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Image captioning with LLaVA and tag management"
    )
    parser.add_argument(
        "--input-type",
        choices=["file", "directory"],
        default="directory",
        help="Process single file or directory (default: directory)",
    )
    parser.add_argument(
        "--image-path", type=str, required=True, help="Path to image file or directory"
    )
    parser.add_argument(
        "--caption-style",
        action="append",
        default=None,
        help=f"Caption style(s) to generate (repeat for multiple). Valid: {', '.join(VALID_CAPTION_TYPES[1:])}",
    )
    parser.add_argument(
        "--caption-length",
        choices=["any", "very short", "short", "medium-length", "long", "very long"],
        default="long",
        help="Caption length preference",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH,
        help="Model path or HuggingFace ID",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default=None,
        help="Override model dtype (default: auto-select based on device). float32=stable, float16=fastest but may have errors, bfloat16=balanced",
    )
    # ---- Tag management options ----
    parser.add_argument(
        "--tag-dir",
        type=str,
        default="",
        help="Root directory containing caption .txt files (defaults to image directory)",
    )
    parser.add_argument(
        "--custom-tag",
        type=str,
        default="",
        help="Comma‑separated list of tags to add or remove",
    )
    parser.add_argument(
        "--tag-action",
        choices=["prepend", "append", "remove", "list"],
        default="list",
        help="Tag management action (prepend/append/remove/list)",
    )
    parser.add_argument(
        "--extension",
        choices=[".txt", ".caption"],
        default=".txt",
        help="Caption file extension to process",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default="",
        help="Subfolder within tag‑dir to process; use '--all' for recursive processing",
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Process subfolders recursively"
    )
    parser.add_argument(
        "--sort-descending",
        action="store_true",
        help="Sort tag frequencies descending when listing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview captions without writing to files",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of images to process before clearing cache (default: 10)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default="checkpoint.json",
        help="Path to checkpoint file (default: checkpoint.json)",
    )
    parser.add_argument(
        "--output-format",
        choices=["txt", "json", "csv"],
        default="txt",
        help="Output format for captions (default: txt)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file path (for json/csv formats). If not specified, uses <image_path>.<ext>",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate captions before saving (check for duplicates, garbled text, etc.)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing (CPU only, requires --batch-size to be set)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=mp.cpu_count(),
        help=f"Number of parallel workers (default: {mp.cpu_count()})",
    )
    return parser.parse_args()


# ----- Tag helper functions -----
def _read_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def _write_file(filepath: str, contents: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(contents)


def _process_tags_file(
    filepath: str, custom_tags: List[str], prepend: bool, append: bool, remove: bool
) -> None:
    contents = _read_file(filepath)
    tags = [t.strip() for t in contents.split(",") if t.strip()]
    for tag in custom_tags:
        tag_space = tag.replace("_", " ")
        tag_underscore = tag.replace(" ", "_")
        if remove:
            # Remove any variant
            tags = [t for t in tags if t not in {tag, tag_space, tag_underscore}]
        else:
            if tag not in tags:
                if prepend:
                    tags.insert(0, tag)
                elif append:
                    tags.append(tag)
    _write_file(filepath, ", ".join(tags))


def _process_directory(
    root_dir: str,
    custom_tags: List[str],
    prepend: bool,
    append: bool,
    remove: bool,
    recursive: bool,
    extension: str,
) -> None:
    for entry in os.listdir(root_dir):
        path = os.path.join(root_dir, entry)
        if os.path.isdir(path) and recursive:
            _process_directory(
                path, custom_tags, prepend, append, remove, recursive, extension
            )
        elif entry.endswith(extension):
            _process_tags_file(path, custom_tags, prepend, append, remove)


def _collect_tag_counts(root_dir: str, extension: str) -> Counter:
    counter = Counter()
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.endswith(extension):
                full = os.path.join(dirpath, fn)
                try:
                    contents = _read_file(full)
                    tags = [t.strip() for t in contents.split(",") if t.strip()]
                    counter.update(tags)
                except Exception:
                    continue
    return counter


# ----- Checkpoint/Resume functions -----
def load_checkpoint(checkpoint_file: str) -> Dict[str, Any]:
    """Load checkpoint file if it exists."""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
    return {"completed": [], "failed": []}


def save_checkpoint(
    checkpoint_file: str, completed: List[str], failed: List[str]
) -> None:
    """Save checkpoint to file."""
    try:
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump({"completed": completed, "failed": failed}, f, indent=2)
    except Exception as e:
        logger.error(f"Could not save checkpoint: {e}")


# ----- Caption Validation -----
def validate_caption(caption: str, image_path: str) -> tuple[bool, str]:
    """
    Validate caption for quality issues.
    Returns (is_valid, error_message)
    """
    if not caption or not caption.strip():
        return False, "Empty caption"

    # Check for excessive repetition
    tags = [t.strip() for t in caption.split(",") if t.strip()]
    tag_counts = Counter(tags)
    duplicates = [(tag, count) for tag, count in tag_counts.items() if count > 2]
    if duplicates:
        return False, f"Excessive tag duplication: {duplicates[:3]}"

    # Check for garbled text (very long words or weird characters)
    words = caption.split()
    long_words = [w for w in words if len(w) > 50]
    if long_words:
        return False, f"Suspiciously long words detected: {long_words[:2]}"

    # Check for reasonable length
    if len(caption) < 10:
        return False, "Caption too short"

    if len(caption) > 10000:
        return False, "Caption suspiciously long"

    return True, ""


# ----- Output Formatting -----
def save_caption_output(
    image_path: str,
    caption: str,
    output_format: str,
    output_file: str = None,
    all_captions: Dict[str, str] = None,
) -> None:
    """Save caption in specified format."""
    if output_format == "txt":
        # Standard .txt format (sidecar)
        base_path, _ = os.path.splitext(image_path)
        caption_path = base_path + ".txt"
        _write_file(caption_path, caption)
    elif output_format == "json":
        # JSON format
        out_path = output_file or os.path.join(
            os.path.dirname(image_path), "captions.json"
        )
        data = all_captions or {image_path: caption}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    elif output_format == "csv":
        # CSV format
        out_path = output_file or os.path.join(
            os.path.dirname(image_path), "captions.csv"
        )
        import csv

        with open(out_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([image_path, caption])


def main():
    global model, processor, model_dtype
    args = parse_args()
    # Resolve tag directory (defaults to image directory if not provided)
    tag_root = (
        args.tag_dir
        if args.tag_dir
        else (
            args.image_path
            if os.path.isdir(args.image_path)
            else os.path.dirname(args.image_path)
        )
    )
    # Resolve subfolder handling similar to notebook
    if args.subfolder:
        if args.subfolder == "--all":
            target_dir = tag_root
            args.recursive = True
        elif args.subfolder.startswith("/content"):
            target_dir = args.subfolder
        else:
            target_dir = os.path.join(tag_root, args.subfolder)
            os.makedirs(target_dir, exist_ok=True)
    else:
        target_dir = tag_root

    # Get caption styles early (handle None default to avoid mutable default issues)
    # Also normalize: strip whitespace and use case-insensitive matching
    raw_styles = args.caption_style if args.caption_style else []
    caption_styles = []
    for style in raw_styles:
        normalized = style.strip()
        # Find case-insensitive match in valid styles
        matched = None
        for valid in VALID_CAPTION_TYPES:
            if valid.lower() == normalized.lower():
                matched = valid
                break
        if matched:
            caption_styles.append(matched)
        else:
            logger.warning(f"⚠️: Unknown style '{style}', skipping")

    # --- Tag management mode ---
    # Only run tag management if explicitly requested via command line args
    # Check if any tag management options were specified
    tag_mode_requested = (
        args.tag_action != "list"  # User specified a different action
        or args.custom_tag  # User provided tags to modify
        or args.tag_dir  # User specified a separate tag directory
    )

    if tag_mode_requested:
        # Ensure there is at least one caption file; if none, create empty ones for any images present
        if not any(fname.endswith(args.extension) for fname in os.listdir(target_dir)):
            for fname in os.listdir(target_dir):
                if fname.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")
                ):
                    base = os.path.splitext(fname)[0]
                    empty_path = os.path.join(target_dir, base + args.extension)
                    open(empty_path, "w", encoding="utf-8").close()
        if args.tag_action == "list":
            counts = _collect_tag_counts(target_dir, args.extension)
            sorted_items = sorted(
                counts.items(), key=lambda kv: kv[1], reverse=args.sort_descending
            )
            for tag, cnt in sorted_items:
                logger.info(f"{tag}: {cnt}")
            # also write to a file
            out_path = os.path.join(os.getcwd(), "tags_with_counts.txt")
            with open(out_path, "w", encoding="utf-8") as out_f:
                for t, c in sorted_items:
                    out_f.write(f"{t}: {c}\n")
            # If no caption styles specified, exit after listing
            if not caption_styles:
                return
        else:
            # prepend / append / remove
            prepend = args.tag_action == "prepend"
            append = args.tag_action == "append"
            remove = args.tag_action == "remove"
            custom_tags = [t.strip() for t in args.custom_tag.split(",") if t.strip()]
            if not custom_tags:
                logger.error("❌ No custom tags supplied for tag modification.")
                return
            _process_directory(
                target_dir,
                custom_tags,
                prepend,
                append,
                remove,
                args.recursive,
                args.extension,
            )
            logger.info(
                f"✅ Tag action '{args.tag_action}' applied to files in {target_dir}"
            )
    # ---------- End tag management ----------

    if not caption_styles:
        logger.error(
            "❌ No caption styles selected. Use --caption-style to specify at least one."
        )
        logger.info(f"Valid styles: {', '.join(VALID_CAPTION_TYPES[1:])}")
        return

    # Memory Cleanup (MPS and CUDA support)
    try:
        if "model" in locals() or "model" in globals():
            del model
        if "processor" in locals() or "processor" in globals():
            del processor
        if DEVICE == "mps":
            torch.mps.empty_cache()
        elif DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        logger.warning(
            f"Memory cleanup encountered an issue: {e}. Attempting load anyway."
        )

    # Verify model path exists before attempting to load
    # Only check local paths; HuggingFace Hub identifiers (containing "/") will be handled by the library
    is_local_path = os.path.isabs(args.model_path) or os.path.exists(args.model_path)
    if (
        is_local_path
        and not os.path.isdir(args.model_path)
        and not os.path.isfile(args.model_path)
    ):
        logger.error(
            f"⚠️ Model path '{args.model_path}' does not exist or is not accessible. Skipping model loading."
        )
        logger.info(
            "Please provide a valid model directory or HuggingFace repo identifier via --model-path."
        )
        sys.exit(1)
    logger.info("Loading Model...")
    processor = AutoProcessor.from_pretrained(args.model_path)

    # Determine dtype based on device (unless overridden by user)
    if args.dtype:
        # User override
        model_dtype = getattr(torch, args.dtype)
        logger.info(f"   Using dtype: {args.dtype} (user override)")
    elif DEVICE == "cuda":
        try:
            torch.zeros(1, dtype=torch.bfloat16).cuda()
            model_dtype = torch.bfloat16
            logger.info(f"   Using dtype: bfloat16 (CUDA optimal)")
        except Exception:
            model_dtype = torch.float32
            logger.info(f"   Using dtype: float32 (CUDA fallback)")
    elif DEVICE == "mps":
        model_dtype = torch.bfloat16
        logger.info(f"   Using dtype: bfloat16 (MPS with autocast - ~17GB memory)")
    else:
        model_dtype = torch.float32
        logger.info(f"   Using dtype: float32 (CPU compatible)")
    # Load the model with appropriate device handling and fallback logic
    try:
        if DEVICE == "cuda":
            # Load on GPU without device_map, then move to CUDA
            model = LlavaForConditionalGeneration.from_pretrained(
                args.model_path,
                torch_dtype=model_dtype,
            )
            model = model.to(DEVICE)
        else:
            # For MPS or CPU, let transformers handle placement
            model = LlavaForConditionalGeneration.from_pretrained(
                args.model_path,
                torch_dtype=model_dtype,
                device_map="auto" if DEVICE != "cpu" else None,
            )
            if DEVICE == "mps":
                model = model.to(DEVICE)
    except Exception as load_err:
        logger.error(f"⚠️ Model loading failed on {DEVICE} with error: {load_err}")
        logger.info("Attempting to load on CPU with float32 dtype.")
        try:
            model = LlavaForConditionalGeneration.from_pretrained(
                args.model_path,
                torch_dtype=torch.float32,
                device_map=None,
            )
            model = model.to("cpu")
        except Exception as final_err:
            logger.critical(f"❌ Fatal error loading model: {final_err}")
            sys.exit(1)
    model.eval()
    logger.info("Model loaded successfully.")

    # Execution
    if args.input_type == "file":
        if os.path.isdir(args.image_path):
            logger.error(
                f"Selected 'Single Image File' but '{args.image_path}' is a directory."
            )
            return
        logger.info(
            f"--- Processing Single File: {os.path.basename(args.image_path)} ---"
        )
        caption, success = process_image_list(
            args.image_path,
            caption_styles,
            args.caption_length,
            args.dry_run,
            args.validate,
            args.output_format,
            args.output_file,
        )
        if success and caption and args.output_format in ["json", "csv"]:
            # Save single file result
            if args.output_format == "json":
                data = {args.image_path: caption}
                out_path = args.output_file or os.path.join(
                    os.path.dirname(args.image_path), "captions.json"
                )
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            elif args.output_format == "csv":
                import csv

                out_path = args.output_file or os.path.join(
                    os.path.dirname(args.image_path), "captions.csv"
                )
                with open(out_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["image_path", "caption"])
                    writer.writerow([args.image_path, caption])
    elif args.input_type == "directory":
        if not os.path.isdir(args.image_path):
            logger.error(
                f"Selected 'Directory of Images' but path '{args.image_path}' is not a valid directory."
            )
            return
        logger.info(f"--- Processing Directory: {args.image_path} ---")

        # Load checkpoint if resuming
        completed_files = set()
        failed_files = set()
        if args.resume:
            checkpoint = load_checkpoint(args.checkpoint_file)
            completed_files = set(checkpoint.get("completed", []))
            failed_files = set(checkpoint.get("failed", []))
            logger.info(
                f"Resuming: {len(completed_files)} already processed, {len(failed_files)} failed"
            )

        img_extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp", "*.tiff"]
        image_files = []
        for ext in img_extensions:
            image_files.extend(glob.glob(os.path.join(args.image_path, ext)))
            image_files.extend(
                glob.glob(os.path.join(args.image_path, "**", ext), recursive=True)
            )
        image_files = list(set(image_files))

        # Filter out already completed files
        if args.resume:
            image_files = [f for f in image_files if f not in completed_files]

        if not image_files:
            logger.error("❌ No supported image files found in the directory.")
            return
        logger.info(f"Found {len(image_files)} image files to process.")

        # Collect results for JSON/CSV output
        all_captions = {}

        for i, file_path in enumerate(
            tqdm(image_files, desc="Processing images", unit="img")
        ):
            caption, success = process_image_list(
                file_path,
                caption_styles,
                args.caption_length,
                args.dry_run,
                args.validate,
                args.output_format,
                args.output_file,
            )

            if success:
                completed_files.add(file_path)
                if caption:
                    all_captions[file_path] = caption
            else:
                failed_files.add(file_path)

            # Save checkpoint periodically
            if (i + 1) % args.batch_size == 0:
                if args.resume:
                    save_checkpoint(
                        args.checkpoint_file, list(completed_files), list(failed_files)
                    )
                gc.collect()
                if DEVICE == "mps":
                    torch.mps.empty_cache()
                elif DEVICE == "cuda":
                    torch.cuda.empty_cache()

        # Final checkpoint save
        if args.resume:
            save_checkpoint(
                args.checkpoint_file, list(completed_files), list(failed_files)
            )
            logger.info(
                f"Checkpoint saved: {len(completed_files)} completed, {len(failed_files)} failed"
            )

        # For JSON/CSV output, write all captions at once
        if all_captions and args.output_format in ["json", "csv"]:
            if args.output_format == "json":
                out_path = args.output_file or os.path.join(
                    args.image_path, "captions.json"
                )
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(all_captions, f, indent=2)
                logger.info(f"Saved JSON output to: {out_path}")
            elif args.output_format == "csv":
                import csv

                out_path = args.output_file or os.path.join(
                    args.image_path, "captions.csv"
                )
                with open(out_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["image_path", "caption"])
                    for img_path, caption in all_captions.items():
                        writer.writerow([img_path, caption])
                logger.info(f"Saved CSV output to: {out_path}")

    # Final Cleanup
    logger.info("Processing complete. Final cleanup...")
    try:
        del model
        del processor
        if DEVICE == "mps":
            torch.mps.empty_cache()
        elif DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Memory released.")
    except Exception:
        pass


if __name__ == "__main__":
    main()
