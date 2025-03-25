import subprocess

def is_relevant_llm(text, model="llama3.2"):
    prompt = f"""You're an assistant helping analyze online posts for greenwashing. 
Is the following Reddit post about greenwashing or related to corporate sustainability claims?

Respond only with YES or NO.

Post:
{text}
"""
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE
    )
    out = result.stdout.decode("utf-8").strip().lower()
    return "yes" in out
