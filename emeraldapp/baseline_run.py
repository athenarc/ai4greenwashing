from model_loader import call_llm, extract_label, extract_justification
from prompts import zero_shot, few_shot


def run(claim, few_shot_flag):

    prompt = few_shot if few_shot_flag else zero_shot
    full_prompt = prompt + f"\n<CLAIM>:\n{claim}\n</CLAIM>"
    
    response_text = call_llm(full_prompt)
    
    return {
        "label": extract_label(response_text),
        "justification": extract_justification(response_text),
        "subgraph": None
    }