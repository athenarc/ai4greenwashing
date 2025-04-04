import subprocess
from pathlib import Path

input_dir = Path("reportparse/asset/small")
pdf_files = input_dir.rglob("*.pdf")
output_base = Path("./results/tmux_llm_agg")

base_cmd = [
    "python", "-m", "reportparse.main",
    "--input_type", "pdf",
    "--overwrite_strategy", "all",
    "--reader", "pymupdf",
    "--annotators", "llm_agg"
]

for pdf in pdf_files:
    basename = pdf.stem  # filename without extension
    output_dir = output_base / basename
    output_dir.mkdir(parents=True, exist_ok=True)

    # Assemble full command
    cmd = base_cmd + [
        "-i", str(pdf),
        "-o", str(output_dir)
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

input_dir = Path("reportparse/asset/small2")

# Find all PDF files under input_dir
pdf_files = input_dir.rglob("*.pdf")

base_cmd = [
    "python", "-m", "reportparse.main",
    "--input_type", "pdf",
    "--overwrite_strategy", "all",
    "--reader", "pymupdf",
    "--annotators", "llm_agg"
]

for pdf in pdf_files:
    basename = pdf.stem  # filename without extension
    output_dir = output_base / basename
    output_dir.mkdir(parents=True, exist_ok=True)

    # Assemble full command
    cmd = base_cmd + [
        "-i", str(pdf),
        "-o", str(output_dir)
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
