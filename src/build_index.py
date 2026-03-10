"""
FAISS Index Builder
Creates a semantic search index from preprocessed lecture chunks.
Uses embeddings to enable efficient similarity-based retrieval.
"""

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import CHUNKS_FILE, INDEX_FILE, METADATA_FILE, INDEXING_CONFIG

print(f"Loading embedding model: {INDEXING_CONFIG['embedding_model']}...")
embedding_model = SentenceTransformer(INDEXING_CONFIG["embedding_model"])
print("Model loaded\n")


def build_index():
    """
    Build FAISS index from chunks.json for semantic similarity search.
    Stores index and metadata for fast retrieval during QA.
    """
    print(f"Loading chunks from {CHUNKS_FILE}...")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        print("No chunks found. Run preprocess.py first.")
        return

    texts = [chunk["text"] for chunk in chunks]

    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True,
        batch_size=INDEXING_CONFIG["batch_size"]
    )

    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]

    print(f"\nCreating FAISS index (dimension: {dimension})...")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save index
    faiss.write_index(index, INDEX_FILE)
    print(f"✓ Index saved: {INDEX_FILE}")

    # Save metadata
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    print(f"✓ Metadata saved: {METADATA_FILE}")

    print(f"\n✓ Index building complete:")
    print(f"  - Total vectors indexed: {len(embeddings)}")
    print(f"  - Embedding dimension: {dimension}")
    print(f"  - Index type: {INDEXING_CONFIG['index_type']}")


if __name__ == "__main__":
    build_index()