import os
import pickle
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

EMBEDDINGS_PATH = "vector_store/embeddings.npy"
MAPPING_PATH = "vector_store/id_to_text.pkl"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class VectorStore:
    def __init__(self):
        self.embed_model = SentenceTransformer("thenlper/gte-small")
        if not os.path.exists(EMBEDDINGS_PATH) or not os.path.exists(MAPPING_PATH):
            raise FileNotFoundError("Embedding or mapping files are missing.")

        self.embeddings = np.load(EMBEDDINGS_PATH)
        with open(MAPPING_PATH, "rb") as f:
            self.id_to_text = pickle.load(f)

    def embed(self, text: str) -> np.ndarray:
        return self.embed_model.encode([text], normalize_embeddings=True).astype("float32")

    def search(self, prompt: str, top_k: int = 3) -> str:
        query_vec = self.embed(prompt)
        sim_scores = cosine_similarity(query_vec, self.embeddings)[0]
        top_indices = sim_scores.argsort()[::-1][:top_k]
        return "\n".join([self.id_to_text[i] for i in top_indices if i in self.id_to_text])
