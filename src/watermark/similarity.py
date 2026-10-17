from sentence_transformers import SentenceTransformer, util


def load_similarity_model():

    return SentenceTransformer("all-MiniLM-L6-v2")


def semantic_similarity(similarity_model, text_a, text_b):

    embeddings = similarity_model.encode([text_a, text_b], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1]).item()
    return score