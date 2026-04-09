# Image Tagger

A powerful Python-based CLI tool for generating, managing, and organizing image captions and tags using large multimodal models like LLaVA.

## Features

- **TAG Mode**: Automatically generate high-quality AI captions/tags for images using LLaVA models. Supports various styles including Descriptive, Stable Diffusion prompts, MidJourney prompts, and Booru-style tag lists (Danbooru, e621, Rule34).
- **EDIT Mode**: Perform bulk modifications on existing tag files, such as prepending, appending, or removing specific tags.
- **LIST Mode**: Analyze your datasets by generating frequency reports of all tags found in a directory.
- **ORGANIZE Mode**: Automatically structure your image datasets into hierarchical directories based on parenthetical concept tags (e.g., 'Concept \(Character\)')

## Installation

### Prerequisites

- Python 3.8+
- NVIDIA GPU (CUDA) or Apple Silicon (MPS) is highly recommended for performance (those are what I've tested this on) though it *can* be used on a cpu as well. AMD and Intel GPUs will *probably* work too but I have not tested it on those kinds yet. 

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/AiArtFactory/similar-image-remover.git
   cd similar-image-remover
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The tool is operated via the `image-tagger.py` script using subcommands.

### TAG Mode
Generate captions for a single image or an entire directory.

```bash
# Generate descriptive captions for all images in a folder
python image-tagger.py tag -i /path/to/images -s "Descriptive"

# Prepend AI captions to existing tags with a trigger word
python image-tagger.py tag -i /path/to/images -s "Danbooru tag list" --prepend-to-existing -c "trigger_word"

# Append AI captions to existing tags
python image-tagger.py tag -i /path/to/images -s "Stable Diffusion Prompt" --append-to-existing
```

**Available Styles:**
- Descriptive
- Descriptive (Casual)
- Straightforward
- Stable Diffusion Prompt
- MidJourney
- Danbooru tag list
- e621 tag list
- Rule34 tag list
- Booru-like tag list
- Art Critic
- Product Listing
- Social Media Post

### EDIT Mode
Modify existing `.txt` or `.caption` files.

```bash
# Add a tag to the front of all files in a directory
python image-tagger.py edit -t /path/to/tags -a prepend -c "artist:name"

# Remove a tag from all files recursively
python image-tagger.py edit -t /path/to/tags -a remove -c "old_tag"
```

### LIST Mode
View tag statistics.

```bash
# List all tags and their frequencies in descending order
python image-tagger.py list -t /path/to/tags --sort-descending
```

### ORGANIZE Mode
Sort images and tags into concept-based folders. This mode relies on the format `TagName \(Qualifier\)`.

```bash
# Organize a dataset into a new directory
python image-tagger.py organize -i /path/to/dataset -o ./organized
```

## Hardware Acceleration

The script automatically detects and uses the best available hardware:
- **CUDA**: For NVIDIA GPUs.
- **MPS**: For Apple Silicon (M1/M2/M3).
- **CPU**: Fallback if no GPU is detected.

## Technical Details

- **Model**: Defaults to `fancyfeast/llama-joycaption-beta-one-hf-llava`.
- **Memory Management**: Includes automatic garbage collection and cache clearing to optimize usage during large batch processes.
