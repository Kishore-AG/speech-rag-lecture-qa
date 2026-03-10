"""
Question Answering System with Hybrid Retrieval
Combines semantic (embedding-based) and keyword-based (BM25) retrieval
followed by reranking and context-aware answer generation using LLM.
"""

import json
import faiss
import numpy as np
import re
import torch
from typing import List, Dict, Optional

from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
from rank_bm25 import BM25Okapi

from config import (
    INDEX_FILE, METADATA_FILE, QA_CONFIG,
    QUERY_PARAPHRASE_PROMPT, CONTEXT_VARIANT_PROMPT, QA_PROMPT
)

# ============================================================================
# MODEL LOADING WITH FALLBACK
# ============================================================================

print("Loading Embedding Model...")
embedding_model = SentenceTransformer(QA_CONFIG["embedding_model"])

print("Loading Reranker...")
reranker = CrossEncoder(QA_CONFIG["reranker_model"])

print("Loading LLM (This may take a moment)...")
# Try primary LLM, fallback to secondary if out of memory
try:
    print(f"  Attempting: {QA_CONFIG['primary_llm']}...")
    LLM_MODEL_ID = QA_CONFIG["primary_llm"]
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    qa_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        load_in_8bit=True
    )
    print(f"  ✓ {QA_CONFIG['primary_llm'].split('/')[-1]} loaded successfully")
except Exception as e:
    print(f"  ✗ Primary LLM failed: {str(e)[:100]}...")
    print(f"  Falling back to {QA_CONFIG['fallback_llm']}...")
    LLM_MODEL_ID = QA_CONFIG["fallback_llm"]
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    qa_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16
    )
    print(f"  ✓ {QA_CONFIG['fallback_llm'].split('/')[-1]} loaded successfully")

# Set pad_token if not already set (required for batch generation)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"  Set pad_token to eos_token")


# ============================================================================
# TEXT GENERATION HELPER
# ============================================================================

def generate_text(prompt: str, max_new_tokens: int = 100) -> str:
    """
    Generate text using causal LLM with proper token handling.
    Works with decoder-only models like Llama and Phi.
    """
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
                max_new_tokens=min(max_new_tokens, QA_CONFIG["max_new_tokens"]),
                temperature=QA_CONFIG["temperature"],
                top_p=QA_CONFIG["top_p"],
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_beams=QA_CONFIG["num_beams"]
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


# ============================================================================
# QUERY EXPANSION FOR BETTER RETRIEVAL
# ============================================================================

def extract_keywords(question):
    """Extract content keywords from question, removing stopwords."""
    words = question.lower().split()
    keywords = [w for w in words if w not in QA_CONFIG["stopwords"] and len(w) > 2]
    keywords = [w.rstrip("s") for w in keywords]  # Remove plural 's'
    return keywords


def generate_synonym_query(question):
    """Generate paraphrased version of question for query expansion."""
    prompt = QUERY_PARAPHRASE_PROMPT.format(question=question)
    return generate_text(prompt, max_new_tokens=40)


def generate_context_variant(keywords_str):
    """Generate contextual variation describing what speaker might discuss."""
    prompt = CONTEXT_VARIANT_PROMPT.format(keywords=keywords_str)
    prefix = generate_text(prompt, max_new_tokens=10)
    return f"{prefix} {keywords_str}"


def generate_queries(question):
    """
    Generate multiple query variations for better retrieval coverage.
    Returns: list of query strings (cleaned original + variants)
    """
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

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for v in variants:
        if v.strip() and v.strip() not in seen:
            seen.add(v.strip())
            unique.append(v)

    return unique


# ============================================================================
# SEMANTIC RETRIEVAL (FAISS)
# ============================================================================

def encode_query(query):
    """
    Encode query for FAISS semantic search with L2 normalization.
    Uses instructional prefix for better embedding quality.
    """
    query = " ".join(query.split())  # Normalize whitespace
    if len(query) > 512:
        query = query[:512]

    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    embedding = embedding_model.encode([prefixed]).astype("float32")

    # L2 normalize for FAISS L2 distance
    norm = np.linalg.norm(embedding[0])
    if norm > 0:
        embedding[0] = embedding[0] / norm

    return embedding


def retrieve_chunks(query, index, metadata, top_k=None):
    """
    Semantic retrieval: find top_k most similar chunks using FAISS.
    """
    if top_k is None:
        top_k = QA_CONFIG["top_k_retrieval"]

    query_embedding = encode_query(query)
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for idx in indices[0]:
        if idx < len(metadata):
            results.append(metadata[idx])
    return results


# ============================================================================
# KEYWORD RETRIEVAL (BM25)
# ============================================================================

def setup_bm25(metadata: List[Dict]) -> BM25Okapi:
    """Build BM25 index for keyword-based retrieval."""
    tokenized_docs = []
    for chunk in metadata:
        tokens = chunk["text"].lower().split()
        tokenized_docs.append(tokens)
    return BM25Okapi(tokenized_docs)


def retrieve_bm25(query: str, bm25_index: BM25Okapi, metadata: List[Dict], top_k: int = None) -> List[Dict]:
    """Keyword retrieval via BM25 scoring."""
    if top_k is None:
        top_k = QA_CONFIG["top_k_retrieval"]

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


# ============================================================================
# HYBRID RETRIEVAL (Semantic + Keyword)
# ============================================================================

def hybrid_retrieve(query: str, index, metadata: List[Dict], bm25_index: Optional[BM25Okapi], top_k: int = None) -> List[Dict]:
    """
    Combine semantic and keyword retrieval with weighted scoring.
    Semantic results weighted more heavily than keyword results.
    """
    if top_k is None:
        top_k = QA_CONFIG["top_k_retrieval"]

    if bm25_index is None:
        return retrieve_chunks(query, index, metadata, top_k=top_k)

    semantic = retrieve_chunks(query, index, metadata, top_k=top_k)
    keyword = retrieve_bm25(query, bm25_index, metadata, top_k=top_k)

    # Combine with weighted scoring
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


# ============================================================================
# RERANKING FOR BEST RELEVANCE
# ============================================================================

def rerank(question: str, chunks: List[Dict], top_k: int = None) -> List[Dict]:
    """
    Rerank chunks using cross-encoder + diversity scoring.
    Balances relevance (90%) and diversity (10%).
    """
    if top_k is None:
        top_k = QA_CONFIG["top_k_rerank"]

    if not chunks:
        return []

    pairs = [(question, chunk["text"]) for chunk in chunks]
    scores = reranker.predict(pairs)
    
    # Diversity score: prefer chunks of similar length to mean
    chunk_lengths = np.array([len(chunk["text"].split()) for chunk in chunks])
    if len(chunk_lengths) > 1:
        length_scores = 1.0 / (1.0 + np.abs(chunk_lengths - chunk_lengths.mean()) / (chunk_lengths.std() + 1e-6))
    else:
        length_scores = np.ones_like(scores)

    # Combined score with configured weights
    combined_scores = (
        QA_CONFIG["relevance_weight"] * scores +
        QA_CONFIG["diversity_weight"] * length_scores
    )

    ranked = sorted(
        zip(chunks, combined_scores),
        key=lambda x: x[1],
        reverse=True
    )
    return [chunk for chunk, _ in ranked[:top_k]]


# ============================================================================
# ANSWER GENERATION
# ============================================================================

def generate_answer(question: str, context_chunks: List[Dict]) -> str:
    """
    Generate answer to question using top context chunks.
    Returns exactly one complete sentence based on lecture content.
    """
    if not context_chunks:
        return "No relevant information found in the lecture."

    # Use top_k_context chunks for LLM context
    context_text = "\n\n".join(
        [c["text"] for c in context_chunks[:QA_CONFIG["top_k_context"]]]
    )

    prompt = QA_PROMPT.format(
        context=context_text,
        question=question
    )

    response = generate_text(prompt, max_new_tokens=QA_CONFIG["max_new_tokens"])

    if not response:
        return "This information is not covered in the provided transcript."

    # Extract first complete sentence
    first_sentence = response.split(".")[0].strip()
    if first_sentence:
        return first_sentence + "."
    return response.strip()


# ============================================================================
# MAIN QA LOOP
# ============================================================================

def main():
    """Load index and run interactive question-answering loop."""
    print("Loading FAISS index...")
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"✓ Index loaded: {index.ntotal} vectors\n")

    print("Setting up BM25 for keyword retrieval...")
    bm25_index = setup_bm25(metadata)
    print("✓ BM25 index ready\n")

    question = input("Ask a question about the lecture: ")

    queries = generate_queries(question)
    print(f"\nExpanded into {len(queries)} queries:")
    for q in queries:
        print(f"  → '{q}'")

    print("\nRetrieving context (hybrid: semantic + keyword)...")
    retrieved = []
    for q in queries:
        chunks = hybrid_retrieve(q, index, metadata, bm25_index, top_k=QA_CONFIG["top_k_retrieval"])
        retrieved.extend(chunks)
        print(f"  Query '{q}' → {len(chunks)} chunks")

    # Deduplicate
    unique_chunks = list({c["chunk_id"]: c for c in retrieved}.values())
    print(f"\nTotal unique chunks: {len(unique_chunks)}")

    print("Reranking for best relevance...")
    reranked = rerank(question, unique_chunks, top_k=QA_CONFIG["top_k_rerank"])
    print(f"✓ Top reranked: {len(reranked)} chunks\n")

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