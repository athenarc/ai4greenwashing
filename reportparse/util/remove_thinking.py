import re

def remove_think_blocks(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()