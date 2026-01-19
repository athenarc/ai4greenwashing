import em_rag_run
import kg_rag_run
import model_loader
from prompts import hybrid_prompt


def run(claim, company_name, year, few_shot_flag):
    print("Calling Baseline model...")
    kg_result = kg_rag_run.run(claim, company_name, year, few_shot_flag)
    print(kg_result)
    print()
    print("Calling RAG model...")
    rag_result = em_rag_run.run(claim, company_name, year, few_shot_flag)
    print(rag_result)
    print()
    # get baseline label/justification
    kg_label = kg_result["label"]
    kg_justification = kg_result["justification"]
    kg_context=kg_result["subgraph"]
    # get rag label/justification
    rag_label = rag_result["label"]
    rag_justification = rag_result["justification"]
    rag_context = rag_result["passages"]
    # fix hybrid prompt
    llm_prompt = hybrid_prompt.format(
        claim=claim,
        rag_label=rag_label,
        rag_justification=rag_justification,
        graphrag_label=kg_label,
        graphrag_justification=kg_justification,
    )
    # call prometheus model
    print("Calling Hybrid model...")
    hybrid_result = model_loader.call_prom(llm_prompt)
    print(hybrid_result)
    if model_loader.extract_justification(hybrid_result) == kg_justification:
        return {
            "label": model_loader.extract_best_label(hybrid_result),
            "justification": model_loader.extract_justification(hybrid_result),
            "subgraph": kg_context,
            "paggages": None
        }
    elif model_loader.extract_justification(hybrid_result) == rag_justification:
        return {
            "label": model_loader.extract_best_label(hybrid_result),
            "justification": model_loader.extract_justification(hybrid_result),
            "subgraph": None,
            "paggages": rag_context
        }
    else:
        return {
            "label": model_loader.extract_best_label(hybrid_result),
            "justification": model_loader.extract_justification(hybrid_result),
            "subgraph": None,
            "paggages": None
        }