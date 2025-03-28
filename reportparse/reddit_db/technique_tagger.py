import subprocess
import re
import json

def extract_techniques(text, model="llama3.2"):
    prompt = f"""You're a greenwashing expert. Read the post below and list any greenwashing techniques it describes.

Output as a JSON list of strings (e.g., ["carbon offsets", "green packaging"]).

Post:
{text}
"""
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE
    )
    try:
        out = result.stdout.decode("utf-8")
        match = re.search(r'\[.*?\]', out, re.DOTALL)
        return json.loads(match.group(0)) if match else []
    except Exception:
        return []