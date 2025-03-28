from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
def cti_classification(claim: str):
    model_names = {
        "climate": "climatebert/distilroberta-base-climate-detector",
        "commitment": "climatebert/distilroberta-base-climate-commitment",
        "sentiment": "climatebert/distilroberta-base-climate-sentiment",
        "specificity": "climatebert/distilroberta-base-climate-specificity",
    }

    results = {}

    for category, model_name in model_names.items():
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)
        result = pipe(claim, padding=True, truncation=True)

        results[category] = result

        del model, tokenizer, pipe
        torch.cuda.empty_cache()

    return results