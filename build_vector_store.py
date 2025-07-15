import os
import faiss
import pickle
import numpy as np
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

texts = [
    "chahat gupta is a good coder and loves her family and she loves noodles and she loves fictional"
]

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

response = client.embeddings.create(input=texts, model="text-embedding-3-small")
embeddings = np.array([d.embedding for d in response.data], dtype="float32")
faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

os.makedirs("vector_store", exist_ok=True)
faiss.write_index(index, "vector_store/index.faiss")

with open("vector_store/id_to_text.pkl", "wb") as f:
    pickle.dump({i: text for i, text in enumerate(texts)}, f)
