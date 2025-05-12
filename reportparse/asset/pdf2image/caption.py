from google import genai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

my_file = client.files.upload(file="/home/geoka/Desktop/greenwashing/ai4greenwashing/reportparse/asset/pdf2image/example_page_4.jpg")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[
        my_file,
        """You are an expert analyst reviewing this image.

Instructions:
1. Identify and number each table and figure clearly (e.g., Figure 1, Table 1).
2. For each, provide:
   a. A concise title or description (if applicable).
   b. An analytical summary of what is shown, with focus on numerical data.
   c. Any important trends, comparisons, or anomalies.
3. Do not speculate. If something is unclear or ambiguous, state “Not clearly visible” or “Not specified”.
4. Exclude any content not derived directly from the image.

Output Format (strict):
---
Figure/Table X: [Title or type]
a. Description: [Concise summary of content]
b. Analysis: [Analytical description of key numerical values, trends, comparisons]
c. Notes: [Any clarifications or “Not clearly visible” if needed]
---

Only output the formatted analysis as specified above. Do not provide additional explanations or commentary."""
    ]
)

print(response.text)