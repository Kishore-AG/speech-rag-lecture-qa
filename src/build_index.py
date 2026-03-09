import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNK_FILE = "data/chunks/chunks.json"
INDEX_FILE = "data/faiss_index.bin"
METADATA_FILE = "data/faiss_metadata.json"

model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def build_index():
    with open(CHUNK_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [chunk["text"] for chunk in chunks]

    print("Generating embeddings...")

    embeddings = model.encode(texts, show_progress_bar=True)

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f)

    print("FAISS index saved")
    print(f"Total vectors indexed: {len(embeddings)}")


if __name__ == "__main__":
    build_index()