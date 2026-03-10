"""
Lecture Transcript Preprocessing Module
Hybrid semantic + agentic chunking with comprehensive noise cleaning.

NOISE TYPES HANDLED (pattern-based, not hardcoded):
- TYPE 1: Short isolated utterances ("Okay.", "Yeah.", "Hmm.")
- TYPE 2: Repeated sentence blocks (Whisper duplicates)
- TYPE 3: Non-ASCII/Cyrillic characters
- TYPE 4: Letter-by-letter spelled names ("T-A-R-I-K")
- TYPE 5: ASR filler words (um, uh, hmm, etc.)
- TYPE 6: Consecutive repeated words

Uses semantic similarity + LLM for intelligent topic boundary detection.
"""

import os
import re
import json
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5Tokenizer, T5ForConditionalGeneration
from config import (
    TRANSCRIPT_DIR, CHUNKS_FILE, PREPROCESS_CONFIG,
    TOPIC_BOUNDARY_PROMPT
)

# Load models once at startup
print("Loading models for preprocessing...")
embedding_model = SentenceTransformer(PREPROCESS_CONFIG["embedding_model"])
tokenizer = T5Tokenizer.from_pretrained(PREPROCESS_CONFIG["llm_model"])
llm = T5ForConditionalGeneration.from_pretrained(PREPROCESS_CONFIG["llm_model"])
print("Models ready.\n")


# ────────────────────────────────────────────────────────────────────────────── 
# NOISE CLEANING (applied to raw text BEFORE sentence splitting)
# ──────────────────────────────────────────────────────────────────────────────

def strip_non_ascii(text):
    """TYPE 3: Remove non-ASCII/Cyrillic characters outside ASCII range."""
    return re.sub(r'[^\x00-\x7F]+', ' ', text)


def strip_spelled_names(text):
    """TYPE 4: Remove letter-by-letter spelled sequences like 'T-A-R-I-K'."""
    return re.sub(r'\b([A-Z]-){2,}[A-Z]\b', '', text)


def remove_isolated_utterances(text):
    """TYPE 1: Remove standalone utterances <= 3 words."""
    parts = re.split(r'(?<=[.?!])\s+', text)
    filtered = []
    for part in parts:
        word_count = len(part.split())
        if word_count > 3:
            filtered.append(part)
    return ' '.join(filtered)


def deduplicate_sentences(text):
    """TYPE 2: Remove repeated sentence blocks using sliding window."""
    sentences = re.split(r'(?<=[.?!])\s+', text)
    seen_window = []
    deduped = []
    for sent in sentences:
        normalized = sent.strip().lower()
        if normalized and normalized not in seen_window:
            deduped.append(sent)
            seen_window.append(normalized)
            if len(seen_window) > 5:
                seen_window.pop(0)
    return ' '.join(deduped)


def clean_raw_transcript(raw_text):
    """Apply all raw-text noise cleaners in sequence."""
    text = raw_text
    text = strip_non_ascii(text)  # TYPE 3: Remove non-ASCII
    text = strip_spelled_names(text)  # TYPE 4: Strip spelled names
    text = deduplicate_sentences(text)  # TYPE 2: Deduplicate repetitions
    text = remove_isolated_utterances(text)  # TYPE 1: Remove isolated utterances
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text


# ────────────────────────────────────────────────────────────────────────────── 
# SENTENCE-LEVEL CLEANING
# ──────────────────────────────────────────────────────────────────────────────

def split_into_sentences(raw_text):
    """
    Split transcript into sentences, filter fragments < 5 words.
    Applied AFTER clean_raw_transcript() so noise is already removed.
    """
    sentences = re.split(r'(?<=[.?!])\s+', raw_text.strip())
    return [s.strip() for s in sentences if len(s.split()) >= 5]


def clean_sentence(sentence):
    """Clean sentence: lowercase, remove filler words, deduplicate words."""
    text = sentence.lower().replace("\n", " ")
    words = text.split()
    cleaned = []
    for word in words:
        word = re.sub(r"[^\w']", "", word)
        if not word or word in PREPROCESS_CONFIG["filler_words"]:
            continue
        if cleaned and word == cleaned[-1]:
            continue
        cleaned.append(word)
    return " ".join(cleaned).strip()


# ────────────────────────────────────────────────────────────────────────────── 
# STAGE 1: SEMANTIC SCORING
# ──────────────────────────────────────────────────────────────────────────────

def embed_sentences(sentences):
    """Generate embeddings for sentence list."""
    return embedding_model.encode(sentences, show_progress_bar=False)


def score_all_boundaries(embeddings):
    """Compute cosine similarity between adjacent sentence pairs."""
    candidates = []
    threshold = PREPROCESS_CONFIG["semantic_threshold"]
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(
            embeddings[i].reshape(1, -1),
            embeddings[i + 1].reshape(1, -1)
        )[0][0]
        if sim < threshold:
            candidates.append((i, round(float(sim), 3)))
    return candidates


# ────────────────────────────────────────────────────────────────────────────── 
# STAGE 2: TWO-TIER BOUNDARY CONFIRMATION
# ──────────────────────────────────────────────────────────────────────────────

def agentic_confirm_pair(sent_a, sent_b):
    """
    Use LLM to determine if two sentences discuss different topics.
    Returns True if they are a topic boundary, False otherwise.
    """
    prompt = TOPIC_BOUNDARY_PROMPT.format(sent_a=sent_a, sent_b=sent_b)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    outputs = llm.generate(
        inputs.input_ids,
        max_new_tokens=5,
        num_beams=2,
        early_stopping=True
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
    return answer.startswith("yes")


def get_confirmed_boundaries(sentences, candidates):
    """
    Two-tier routing:
    - Tier 1: Auto-confirm boundaries with very low similarity
    - Tier 2: Ask LLM for ambiguous boundaries
    """
    confirmed = []
    tier1_count = 0
    tier2_count = 0
    auto_threshold = PREPROCESS_CONFIG["auto_confirm_threshold"]

    print(f"    Total candidates  : {len(candidates)}")

    for idx, sim in candidates:
        if sim < auto_threshold:
            confirmed.append(idx)
            tier1_count += 1
            print(f"      ✓ AUTO     [{idx:3d}→{idx+1}] sim={sim}: "
                  f"'{sentences[idx][:45]}...'")
        else:
            sent_b = sentences[idx + 1] if idx + 1 < len(sentences) else ""
            if not sent_b:
                continue
            is_boundary = agentic_confirm_pair(sentences[idx], sent_b)
            tier2_count += 1
            if is_boundary:
                confirmed.append(idx)
                print(f"      ✓ AGENTIC  [{idx:3d}→{idx+1}] sim={sim}: "
                      f"'{sentences[idx][:45]}...'")
            else:
                print(f"      ✗ REJECTED [{idx:3d}→{idx+1}] sim={sim}: "
                      f"'{sentences[idx][:45]}...'")

    print(f"    Auto-confirmed    : {tier1_count}")
    print(f"    Agentic confirmed : {tier2_count} "
          f"(from {len(candidates)-tier1_count} Tier 2 candidates)")
    print(f"    Total confirmed   : {len(confirmed)}")
    return confirmed


# ────────────────────────────────────────────────────────────────────────────── 
# STAGE 3: BUILD CHUNKS & APPLY GUARDS
# ──────────────────────────────────────────────────────────────────────────────

def build_raw_chunks(sentences, boundaries):
    """Build chunks from sentence boundaries."""
    boundary_set = set(boundaries)
    chunks, current = [], []
    for i, sent in enumerate(sentences):
        current.append(sent)
        if i in boundary_set:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks


def recursive_split(chunk):
    """Recursively split oversized chunks at word midpoint."""
    words = chunk.split()
    max_words = PREPROCESS_CONFIG["max_chunk_words"]
    if len(words) <= max_words:
        return [chunk]
    mid = len(words) // 2
    return (recursive_split(" ".join(words[:mid])) + 
            recursive_split(" ".join(words[mid:])))


def size_guard(chunks):
    """Merge undersized chunks, recursively split oversized chunks."""
    min_words = PREPROCESS_CONFIG["min_chunk_words"]
    max_words = PREPROCESS_CONFIG["max_chunk_words"]
    
    merged, buffer = [], ""
    for chunk in chunks:
        if buffer:
            combined = (buffer + " " + chunk).strip()
            if len(combined.split()) >= min_words:
                merged.append(combined)
                buffer = ""
            else:
                buffer = combined
        else:
            if len(chunk.split()) < min_words:
                buffer = chunk
            else:
                merged.append(chunk)
    if buffer:
        if merged:
            merged[-1] = (merged[-1] + " " + buffer).strip()
        else:
            merged.append(buffer)

    guarded = []
    for chunk in merged:
        guarded.extend(recursive_split(chunk))
    return [c.strip() for c in guarded if c.strip()]


# ────────────────────────────────────────────────────────────────────────────── 
# FALLBACK: FIXED-SIZE CHUNKING
# ──────────────────────────────────────────────────────────────────────────────

def fixed_chunk(text):
    """Fixed-size chunking with overlap (fallback method)."""
    chunk_size = PREPROCESS_CONFIG["fixed_chunk_size"]
    overlap = PREPROCESS_CONFIG["fixed_overlap"]
    
    words, chunks, i = text.split(), [], 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


# ────────────────────────────────────────────────────────────────────────────── 
# FULL PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def hybrid_chunk(raw_text, filename=""):
    """Full hybrid pipeline with comprehensive noise cleaning."""
    cleaned_raw = clean_raw_transcript(raw_text)
    raw_sentences = split_into_sentences(cleaned_raw)
    print(f"  Sentences after noise clean + filter: {len(raw_sentences)}")
    
    min_sentences = PREPROCESS_CONFIG["min_sentences_for_hybrid"]
    if len(raw_sentences) < min_sentences:
        print(f"  ⚠ Too few sentences — using fixed fallback")
        clean = " ".join(clean_sentence(s) for s in raw_sentences)
        chunks = fixed_chunk(clean)
        return [
            {
                "text": c,
                "chunk_method": "fixed_fallback",
                "word_count": len(c.split())
            }
            for c in chunks if c.strip()
        ]

    cleaned = [clean_sentence(s) for s in raw_sentences]
    cleaned = [s for s in cleaned if s.strip()]
    
    print(f"  [Stage 1] Embedding {len(cleaned)} sentences...")
    embeddings = embed_sentences(cleaned)
    candidates = score_all_boundaries(embeddings)
    tier1 = sum(1 for _, sim in candidates if sim < PREPROCESS_CONFIG["auto_confirm_threshold"])
    tier2 = len(candidates) - tier1
    print(f"  [Stage 1] Candidates: {len(candidates)} (Tier1={tier1} auto, Tier2={tier2} agentic)")

    print(f"  [Stage 2] Two-tier confirmation...")
    confirmed = get_confirmed_boundaries(cleaned, candidates)
    raw_chunks = build_raw_chunks(cleaned, confirmed)
    final_chunks = size_guard(raw_chunks)
    print(f"  [Stage 3] Raw: {len(raw_chunks)} → After size guard: {len(final_chunks)}")
    
    return [
        {
            "text": chunk,
            "chunk_method": "hybrid",
            "word_count": len(chunk.split())
        }
        for chunk in final_chunks if chunk.strip()
    ]


# ────────────────────────────────────────────────────────────────────────────── 
# PROCESS ALL TRANSCRIPTS
# ──────────────────────────────────────────────────────────────────────────────

def process_transcripts():
    """Process all transcripts in TRANSCRIPT_DIR and save chunks."""
    all_chunks = []
    chunk_id = 0

    transcript_files = sorted(
        f for f in os.listdir(TRANSCRIPT_DIR) if f.endswith(".txt")
    )

    if not transcript_files:
        print(f"No .txt files found in {TRANSCRIPT_DIR}")
        return

    for filename in transcript_files:
        path = os.path.join(TRANSCRIPT_DIR, filename)
        print(f"\n{'─'*60}")
        print(f"Processing: {filename}")
        print(f"{'─'*60}")

        with open(path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        chunks = hybrid_chunk(raw_text, filename)

        print(f"\n  ✓ {filename}: {len(chunks)} chunks")
        for i, c in enumerate(chunks[:3], 1):
            print(f"    [{c['word_count']}w] '{c['text'][:70]}...'")
        if len(chunks) > 3:
            print(f"    ... and {len(chunks)-3} more")

        for chunk in chunks:
            all_chunks.append({
                "chunk_id": chunk_id,
                "source_file": filename,
                "text": chunk["text"],
                "chunk_method": chunk["chunk_method"],
                "word_count": chunk["word_count"],
            })
            chunk_id += 1

    os.makedirs(os.path.dirname(CHUNKS_FILE), exist_ok=True)

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    wc = [c["word_count"] for c in all_chunks]
    max_words = PREPROCESS_CONFIG["max_chunk_words"]
    h_count = sum(1 for c in all_chunks if c["chunk_method"] == "hybrid")
    f_count = sum(1 for c in all_chunks if c["chunk_method"] == "fixed_fallback")
    over = [c for c in all_chunks if c["word_count"] > max_words]

    print(f"\n{'═'*60}")
    print(f"CHUNKING SUMMARY")
    print(f"{'═'*60}")
    print(f"  Total chunks        : {len(all_chunks)}")
    print(f"  Hybrid              : {h_count}")
    print(f"  Fixed fallback      : {f_count}")
    print(f"  Avg words per chunk : {sum(wc)/len(wc):.1f}")
    print(f"  Min / Max           : {min(wc)} / {max(wc)}")
    print(f"  Oversized (>{max_words}w)   : {len(over)}  ← should be 0")
    print(f"\n  Saved → {CHUNKS_FILE}")


if __name__ == "__main__":
    process_transcripts()