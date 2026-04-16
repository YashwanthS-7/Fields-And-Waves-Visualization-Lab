import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from .llm_service import generate_explanation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "index", "faiss.index")
META_PATH = os.path.join(BASE_DIR, "index", "metadata.pkl")

index = None
texts = None
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def load_index():
    global index, texts
    if index is None:
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(f"Index not found at {INDEX_PATH}")
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            texts = pickle.load(f)

def get_answer(question: str, session_id: str = "anonymous", top_k=5):
    load_index()

    query_vector = embedder.encode([question])
    _, indices = index.search(query_vector, top_k)

    chunks = [texts[i] for i in indices[0]]
    context = "\n\n".join(chunks)

    return generate_explanation(context, question)
