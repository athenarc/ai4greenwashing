#!/bin/bash

inputs_dir="."
outputs_dir="."

# Allow overriding input/output directories via arguments
IN=${1:-"$inputs_dir"}
OUT=${2:-"$outputs_dir"}
mkdir -p "$OUT"

# Process all .jpg images in the input directory
find "$IN" -type f -iname '*.jpg' | while IFS= read -r img; do
    # Query LLM for a title/description of the image
    title=$(llm -m gemma3 \
        "Give a description for each table and figure in this image, if there exists one. Describe what is shown analytically, especially the numerical data. If you are not sure or it is not clear what is shown, do not speculate. Don't provide other output." \
        -a "$img" -o seed 0 -o temperature 0 < /dev/null)

    # Extract the base filename (without extension)
    base_name=$(basename "$img" .jpg)

    printf '%s\n' "$title" > "$OUT/${base_name}.txt"
done