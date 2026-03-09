import json
import faiss
import numpy as np
import re
import torch
from typing import List, Dict, Optional

from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
from rank_bm25 import BM25Okapi

# ---------- Configuration & Files ----------
INDEX_FILE = "data/faiss_index.bin"
METADATA_FILE = "data/faiss_metadata.json"

# ---------- Model Loading with Fallback ----------
print("Loading Embedding Model...")
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

print("Loading Reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("Loading LLM (This may take a moment)...")
# Try Llama 3.1 8B, fallback to Phi-3-mini if out of memory
try:
    print("  Attempting: Meta-Llama-3.1-8B-Instruct...")
    LLM_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    qa_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        load_in_8bit=True
    )
    print("  ✓ Llama 3.1 8B loaded successfully")
except Exception as e:
    print(f"  ✗ Llama 3.1 8B failed: {str(e)[:100]}...")
    print("  Falling back to Phi-3-mini...")
    LLM_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    qa_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16
    )
    print("  ✓ Phi-3-mini loaded successfully")

# Set pad_token if not already set (required for batch generation)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"  Set pad_token to eos_token")

# Stopwords for keyword extraction
STOPWORDS = {
    "what", "is", "the", "a", "an", "in", "of", "to", "are",
    "tell", "me", "can", "you", "please", "describe", "explain",
    "and", "or", "for", "about", "from", "with", "by", "on", "at",
    "as", "be", "been", "being", "do", "does", "did", "will",
    "would", "should", "could", "have", "has", "had", "this", "that"
}

# ---------- Helper: Causal LM Generation ----------
def generate_text(prompt: str, max_new_tokens: int = 100) -> str:
    # Generate text using Decoder-Only models (Llama, Phi, etc) with proper token slicing
    try:
        inputs = tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=3000
        ).to(qa_model.device)
        
        with torch.no_grad():
            outputs = qa_model.generate(
                **inputs,
                max_new_tokens=min(max_new_tokens, 150),
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_beams=1
            )
        
        input_length = inputs.input_ids.shape[1]
        response = tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        ).strip()
        
        return response if response else "Unable to generate response."
    
    except Exception as e:
        print(f"  ⚠ Generation error: {str(e)[:80]}")
        return f"Error: {str(e)[:100]}"

# ---------- Query Expansion ----------
def extract_keywords(question):
    words = question.lower().split()
    keywords = [w for w in words if w not in STOPWORDS and len(w) > 2]
    keywords = [w.rstrip("s") for w in keywords]
    return keywords

def generate_synonym_query(question):
    # Paraphrase question for query expansion using native Llama/Phi task
    prompt = f"paraphrase: {question}"
    return generate_text(prompt, max_new_tokens=40)

def generate_context_variant(keywords_str):
    prompt = f"""Complete this sentence with 3-4 words describing what a spoken transcript discusses.
Topic keywords: {keywords_str}
The speaker ___"""
    
    prefix = generate_text(prompt, max_new_tokens=10)
    return f"{prefix} {keywords_str}"

def generate_queries(question):
    cleaned_question = re.sub(r"[^\w\s]", "", question.strip())
    keywords = extract_keywords(cleaned_question)

    if not keywords:
        return [cleaned_question]

    keywords_str = " ".join(keywords)
    context_variant = generate_context_variant(keywords_str)
    synonym_query = generate_synonym_query(cleaned_question)

    print(f"  [Context variant]: {context_variant}")
    print(f"  [Synonym query]  : {synonym_query}")

    variants = [cleaned_question, keywords_str, context_variant, synonym_query]
    
    seen = set()
    unique = []
    for v in variants:
        if v.strip() and v.strip() not in seen:
            seen.add(v.strip())
            unique.append(v)

    return unique

# ---------- FAISS Retrieval ----------
def encode_query(query):
    # Normalize whitespace and encode query with L2 normalization for FAISS
    query = " ".join(query.split())
    if len(query) > 512:
        query = query[:512]
    
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    embedding = embedding_model.encode([prefixed]).astype("float32")
    
    norm = np.linalg.norm(embedding[0])
    if norm > 0:
        embedding[0] = embedding[0] / norm
    
    return embedding

def retrieve_chunks(query, index, metadata, top_k=30):
    # Semantic retrieval via FAISS
    query_embedding = encode_query(query)
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for idx in indices[0]:
        if idx < len(metadata):
            results.append(metadata[idx])
    return results

# ---------- BM25 Retrieval (Keyword-based) ----------
def setup_bm25(metadata: List[Dict]) -> BM25Okapi:
    # Build BM25 index from chunk texts for keyword-based retrieval
    tokenized_docs = []
    for chunk in metadata:
        tokens = chunk["text"].lower().split()
        tokenized_docs.append(tokens)
    return BM25Okapi(tokenized_docs)

def retrieve_bm25(query: str, bm25_index: BM25Okapi, metadata: List[Dict], top_k: int = 30) -> List[Dict]:
    # Keyword retrieval via BM25 with score-based ranking
    tokens = query.lower().split()
    if not tokens:
        return []
    
    scores = bm25_index.get_scores(tokens)
    scored = [(i, scores[i]) for i in range(len(scores)) if scores[i] > 0]
    if not scored:
        return []
    
    scored.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in scored[:top_k]]
    
    results = []
    for idx in top_indices:
        if idx < len(metadata):
            results.append(metadata[idx])
    return results

def hybrid_retrieve(query: str, index, metadata: List[Dict], bm25_index: Optional[BM25Okapi], top_k: int = 50) -> List[Dict]:
    # Combine semantic + keyword retrieval using weighted scoring
    if bm25_index is None:
        return retrieve_chunks(query, index, metadata, top_k=top_k)
    
    semantic = retrieve_chunks(query, index, metadata, top_k=top_k)
    keyword = retrieve_bm25(query, bm25_index, metadata, top_k=top_k)
    
    combined = {}
    for pos, chunk in enumerate(semantic):
        cid = chunk["chunk_id"]
        if cid not in combined:
            combined[cid] = {"chunk": chunk, "score": 1.0 - (pos / len(semantic))}
    
    for pos, chunk in enumerate(keyword):
        cid = chunk["chunk_id"]
        if cid in combined:
            combined[cid]["score"] += 0.2 * (1.0 - (pos / len(keyword)))
        else:
            if len(combined) < top_k:
                combined[cid] = {"chunk": chunk, "score": 0.4 * (1.0 - (pos / len(keyword)))}
    
    sorted_results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
    return [item["chunk"] for item in sorted_results[:top_k]]

def rerank(question: str, chunks: List[Dict], top_k: int = 15) -> List[Dict]:
    # Rerank chunks with cross-encoder + diversity scoring (90% relevance, 10% diversity)
    if not chunks:
        return []
    pairs = [(question, chunk["text"]) for chunk in chunks]
    scores = reranker.predict(pairs)
    chunk_lengths = np.array([len(chunk["text"].split()) for chunk in chunks])
    if len(chunk_lengths) > 1:
        length_scores = 1.0 / (1.0 + np.abs(chunk_lengths - chunk_lengths.mean()) / (chunk_lengths.std() + 1e-6))
    else:
        length_scores = np.ones_like(scores)
    combined_scores = 0.9 * scores + 0.1 * length_scores
    ranked = sorted(
        zip(chunks, combined_scores),
        key=lambda x: x[1],
        reverse=True
    )
    return [chunk for chunk, _ in ranked[:top_k]]

def generate_answer(question: str, context_chunks: List[Dict]) -> str:
    # Generate direct answer using causal LLM with focused RAG context (top 3 chunks)
    # Fixes: (1) prompt ends at ANSWER: only (no filler), (2) use 3 chunks not 12, (3) exact one sentence, (4) 150 tokens max
    if not context_chunks:
        return "No relevant information found in the lecture."
    context_text = "\n\n".join([c["text"] for c in context_chunks[:3]])
    prompt = f"""Answer the question using only the transcript below. Write exactly one complete sentence.

TRANSCRIPT:
{context_text}

QUESTION: {question}

ANSWER:"""
    response = generate_text(prompt, max_new_tokens=150)
    if not response:
        return "This information is not covered in the provided transcript."
    first_sentence = response.split(".")[0].strip()
    if first_sentence:
        return first_sentence + "."
    return response.strip()

def main():
    # Load FAISS index and metadata, setup BM25, run interactive QA loop
    print("Loading FAISS index...")
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"Index loaded: {index.ntotal} vectors")
    print("Setting up BM25 for keyword retrieval...")
    bm25_index = setup_bm25(metadata)
    print("Ready.\n")
    question = input("Ask a question: ")
    queries = generate_queries(question)
    print(f"\nExpanded into {len(queries)} queries:")
    for q in queries:
        print(f"  → '{q}'")
    print("\nRetrieving context (hybrid: semantic + keyword)...")
    retrieved = []
    for q in queries:
        chunks = hybrid_retrieve(q, index, metadata, bm25_index, top_k=50)
        retrieved.extend(chunks)
        print(f"  Query '{q}' → {len(chunks)} chunks")
    unique_chunks = list({c["chunk_id"]: c for c in retrieved}.values())
    print(f"\nTotal unique chunks: {len(unique_chunks)}")
    print("Reranking for best relevance...")
    reranked = rerank(question, unique_chunks, top_k=15)
    print(f"Top reranked: {len(reranked)} chunks\n")
    print("Context chunks for LLM:")
    for i, chunk in enumerate(reranked[:12], 1):
        words = len(chunk['text'].split())
        print(f"  {i}. [{words}w] {chunk['text'][:70]}...")
    print("\nGenerating answer (this may take a moment)...")
    answer = generate_answer(question, reranked)
    print("\n" + "="*60)
    print("ANSWER:")
    print("="*60)
    print(answer)
    print("="*60)

if __name__ == "__main__":
    main()