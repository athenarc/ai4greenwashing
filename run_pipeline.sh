#!/bin/bash


PDF_PATH=""
OUTPUT_DIR=""
RESULTS_DIR="./results"
OVERWRITE_STRATEGY="no"
READER="pymupdf"
ANNOTATORS="kpi"

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --pdf) PDF_PATH="$2"; shift 2 ;;
        --outdir) OUTPUT_DIR="$2"; shift 2 ;;
        --results) RESULTS_DIR="$2"; shift 2 ;;
        --overwrite_strategy) OVERWRITE_STRATEGY="$2"; shift 2 ;;
        --reader) READER="$2"; shift 2 ;;
        --annotators) 
            shift
            while [[ "$#" -gt 0 && ! "$1" =~ ^-- ]]; do
                ANNOTATORS+="${ANNOTATORS:+ }$1"
                shift
            done
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

if [[ -z "$PDF_PATH" || -z "$OUTPUT_DIR" ]]; then
    echo "Usage: $0 --pdf <pdf_path> --outdir <figure_description_output_dir> [--results <results_dir>] [--overwrite_strategy <strategy>] [--reader <reader>] [--annotators <annotators>]"
    exit 1
fi


python /home/geoka/Desktop/greenwashing/ai4greenwashing/reportparse/util/create_image_descriptions.py --pdf_path "$PDF_PATH" --output_dir "$OUTPUT_DIR"
python -m reportparse.main \
    -i "$PDF_PATH" \
    -o "$RESULTS_DIR" \
    --input_type "pdf" \
    --overwrite_strategy "$OVERWRITE_STRATEGY" \
    --reader "$READER" \
    --annotators $ANNOTATORS \
    --figure_description_dir "$OUTPUT_DIR"