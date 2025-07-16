import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

texts = [
    "chahat gupta is a good coder and loves her family and she loves noodles and she loves fictional"
]

os.makedirs("vector_store", exist_ok=True)

model = SentenceTransformer("thenlper/gte-small")
embeddings = model.encode(texts, normalize_embeddings=True)
np.save("vector_store/embeddings.npy", embeddings)

with open("vector_store/id_to_text.pkl", "wb") as f:
    pickle.dump({i: text for i, text in enumerate(texts)}, f)

print("✅ Vector store built successfully.")
