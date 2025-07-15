import os
import faiss
import pickle
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

INDEX_PATH = "vector_store/index.faiss"
MAPPING_PATH = "vector_store/id_to_text.pkl"

class VectorStore:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        if not os.path.exists(INDEX_PATH) or not os.path.exists(MAPPING_PATH):
            raise FileNotFoundError("FAISS index or mapping file not found.")

        self.index = faiss.read_index(INDEX_PATH)
        with open(MAPPING_PATH, "rb") as f:
            self.id_to_text = pickle.load(f)

    def embed(self, text: str) -> np.ndarray:
        try:
            response = self.client.embeddings.create(
                input=[text],
                model="text-embedding-3-small"
            )
            embedding = np.array([response.data[0].embedding], dtype="float32")
            faiss.normalize_L2(embedding)
            return embedding
        except Exception as e:
            raise RuntimeError(f"[Embedding Error] {e}")

    def search(self, prompt: str, top_k: int = 3) -> str:
        embedding = self.embed(prompt)
        D, I = self.index.search(embedding, top_k)
        return "\n".join([self.id_to_text[i] for i in I[0] if i in self.id_to_text])
