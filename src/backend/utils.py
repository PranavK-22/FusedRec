import numpy as np

def cosine_similarity_matrix(matrix, query_vector):
    dot_product = np.dot(matrix, query_vector)
    norm_a = np.linalg.norm(matrix, axis=1)
    norm_b = np.linalg.norm(query_vector)
    return dot_product / (norm_a * norm_b + 1e-10)