import re

def extract_label(text):
    try:
        match = re.search(
            r"Result of the statement:(.*?)Justification:", text, re.DOTALL
        )
        return match.group(1).strip() if match else ""
    except Exception as e:
        print(f"Error during label extraction: {e}")
        return None


def extract_justification(text):
    try:
        match = re.search(r"Justification:\s*(.*)", text, re.DOTALL)
        return match.group(1).strip() if match else ""
    except Exception as e:
        print(f"Error during justification extraction: {e}")
        return None
