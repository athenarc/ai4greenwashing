from sentence_transformers import SentenceTransformer, util
import torch  

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def single_text_embedding(text):
    embedding = model.encode(text, convert_to_tensor=True)
    torch.cuda.empty_cache()  # Free unused GPU memory
    return embedding

def cos_sim(embedding1, embedding2):
    return util.pytorch_cos_sim(embedding1, embedding2).item()


