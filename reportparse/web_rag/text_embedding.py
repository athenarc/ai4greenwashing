from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def single_text_embedding(text):
    return model.encode(text, convert_to_tensor=True)  

def cos_sim(embedding1, embedding2):
    return util.pytorch_cos_sim(embedding1, embedding2).item()

