import os
import argparse
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Authenticate client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Argument parsing
parser = argparse.ArgumentParser(description="Process ESG PDF page images with Gemini and extract structured data.")
parser.add_argument("input_dir", help="Directory containing input .jpg files named example_page_<i>.jpg")
parser.add_argument("start", type=int, help="Start index of pages")
parser.add_argument("end", type=int, help="End index of pages (inclusive)")
args = parser.parse_args()

# Prompt definition
system_prompt = """You are an expert analyst of environmental, social, and governance (ESG) disclosures. You will receive ONE JPEG that represents an entire PDF page from a corporate sustainability report. The page may contain multiple charts, tables, infographics, or photos with overlaid text.

Detect every semantically distinct visual element in reading order and extract all quantitative and textual information.  Return a **single valid JSON** object with the exact schema below-no free text, comments, or Markdown fences.

{
  "visuals": [
    {
      "id": <integer>,
      "visual_type": "<chart | table | infographic | photo | other>",
      "title": "<string | null>",
      "description": "<description text>",
      "series": [
        {
          "series_name": "<string | null>",
          "x_label": "<string | null>",
          "y_label": "<string | null>",
          "unit": "<string | null>",
          "data_points": [
            { "x": "<string | number>", "y": <number> }
          ]
        }
      ] | null,
      "table": {
        "headers": ["<string>", ...] | null,
        "rows": [
          { "<header_1>": "<value>", "<header_2>": "<value>", ... }
        ] | null
      } | null,
      "caption_text": "<string | null>",
      "raw_text": "<string | null>"
    }
  ]
}

Rules:
1. Use null for any field that is missing or unreadable.
2. Preserve the key order exactly as given.
3. Provide a detailed description of the visual element and its information in the "description" field.
4. Output only the JSON object-no additional text."""

# Process files
for i in range(args.start, args.end + 1):
    image_path = os.path.join(args.input_dir, f"example_page_{i}.jpg")
    output_path = os.path.join(args.input_dir, f"example_page_{i}.txt")

    my_file = client.files.upload(file=image_path)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[my_file, system_prompt]
    )

    with open(output_path, "w") as f:
        f.write(response.text)
