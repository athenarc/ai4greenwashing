import os
import tempfile
import argparse
from pdf2image import convert_from_path

# Argument parsing
parser = argparse.ArgumentParser(description="Convert PDF pages to JPEG images.")
parser.add_argument("pdf_path", help="Path to the input PDF file.")
parser.add_argument("output_dir", help="Directory to save the output JPEG images.")
args = parser.parse_args()

pdf_path = args.pdf_path
output_dir = args.output_dir

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Extract the base name of the PDF file
pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]

# Convert PDF to images using a temporary directory
with tempfile.TemporaryDirectory() as temp_dir:
    images = convert_from_path(
        pdf_path,
        output_folder=temp_dir,
        fmt='jpeg',
        dpi=300,
        thread_count=8
    )

    for idx, img in enumerate(images, start=1):
        output_filename = f"{pdf_basename}_page_{idx}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        img.save(output_path, 'JPEG')
        print(f"Saved {output_path}")
