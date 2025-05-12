from pdf2image import convert_from_path
import os
import tempfile

# Set paths
pdf_path = '/home/geoka/Desktop/greenwashing/ai4greenwashing/reportparse/asset/example.pdf'
output_dir = '/home/geoka/Desktop/greenwashing/ai4greenwashing/reportparse/asset/pdf2image'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Extract the base name for naming the images
pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]

with tempfile.TemporaryDirectory() as temp_dir:
    # Convert PDF to images
    images = convert_from_path(pdf_path, output_folder=temp_dir, fmt='jpeg', dpi=300, thread_count=8)

    # Save images one by one
    for idx, img in enumerate(images, start=1):
        output_filename = f"{pdf_basename}_page_{idx}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        img.save(output_path, 'JPEG')
        print(f"Saved {output_path}")
