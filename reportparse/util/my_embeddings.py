# my_embeddings.py
from sentence_transformers import SentenceTransformer
import torch

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2", device=_DEVICE)

def get_embedder() -> SentenceTransformer:
    return _EMBEDDER
