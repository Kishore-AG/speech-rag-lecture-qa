# preprocess.py — Hybrid Semantic + Agentic Chunking with Comprehensive Noise Cleaning
# NOISE TYPES HANDLED (all via pattern rules — zero hardcoded words/names):
# TYPE 1 — Short isolated utterances ("Okay.", "Yeah.", "Hmm.", "Right.")
# TYPE 2 — Repeated sentence blocks (Whisper repeats same sentence 2-5 times)
# TYPE 3 — Non-ASCII / Cyrillic characters (Whisper hallucinates foreign scripts)
# TYPE 4 — Letter-by-letter spelled names ("T-A-R-I-K")
# TYPE 5 — ASR filler words (um, uh, hmm, mhm, etc.)
# TYPE 6 — Consecutive repeated words ("it could be that it could be that")
# NOTE: Garbled names are LLM-handled as proper nouns; production fix = custom ASR vocabulary

import os
import re
import json
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5Tokenizer, T5ForConditionalGeneration


# ── Config ───────────────────────────────────────────────────────────────────

TRANSCRIPT_DIR = "data/transcripts"
OUTPUT_FILE    = "data/chunks/chunks.json"

AUTO_CONFIRM_THRESHOLD = 0.42   # Tier 1: auto-split (clearly different topics)
SEMANTIC_THRESHOLD     = 0.55   # Tier 2: ask Flan-T5 for ambiguous zone

MIN_CHUNK_WORDS          = 40
MAX_CHUNK_WORDS          = 200
MIN_SENTENCES_FOR_HYBRID = 15
FIXED_CHUNK_SIZE         = 180
FIXED_OVERLAP            = 45

# True ASR vocal noise — no content words
FILLER_WORDS = {
    "um", "uh", "hmm", "hm", "ah", "er", "eh", "umm", "uhh", "mhm"
}


# ── Models ───────────────────────────────────────────────────────────────────

print("Loading models...")
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
tokenizer       = T5Tokenizer.from_pretrained("google/flan-t5-large")
llm             = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
print("Models ready.\n")


# ── Noise Cleaning (applied to raw text BEFORE sentence splitting) ────────────

def strip_non_ascii(text):
    # TYPE 3: Remove non-ASCII/Cyrillic chars outside ASCII range (0-127)
    return re.sub(r'[^\x00-\x7F]+', ' ', text)


def strip_spelled_names(text):
    # TYPE 4: Remove letter-by-letter spelled sequences like "T-A-R-I-K" (2+ uppercase letters with hyphens)
    return re.sub(r'\b([A-Z]-){2,}[A-Z]\b', '', text)


def remove_isolated_utterances(text):
    # TYPE 1: Remove standalone utterances ≤ 3 words ("Okay.", "Yeah.", etc.)
    parts    = re.split(r'(?<=[.?!])\s+', text)
    filtered = []
    for part in parts:
        word_count = len(part.split())
        if word_count > 3:
            filtered.append(part)
    return ' '.join(filtered)


def deduplicate_sentences(text):
    # TYPE 2: Remove repeated sentence blocks using sliding window of 5 sentences
    sentences = re.split(r'(?<=[.?!])\s+', text)
    seen_window = []
    deduped     = []
    for sent in sentences:
        normalized = sent.strip().lower()
        if normalized and normalized not in seen_window:
            deduped.append(sent)
            seen_window.append(normalized)
            if len(seen_window) > 5:
                seen_window.pop(0)
    return ' '.join(deduped)


def clean_raw_transcript(raw_text):
    # Apply all raw-text noise cleaners in sequence before sentence splitting
    text = raw_text
    text = strip_non_ascii(text)  # TYPE 3: Remove non-ASCII/Cyrillic
    text = strip_spelled_names(text)  # TYPE 4: Strip spelled names
    text = deduplicate_sentences(text)  # TYPE 2: Deduplicate repeated sentences
    text = remove_isolated_utterances(text)  # TYPE 1: Remove isolated utterances
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text


# ── Sentence-Level Cleaning ───────────────────────────────────────────────────

def split_into_sentences(raw_text):
    """
    Split transcript into sentences, filter fragments < 5 words.
    Applied AFTER clean_raw_transcript() so noise is already removed.
    """
    sentences = re.split(r'(?<=[.?!])\s+', raw_text.strip())
    return [s.strip() for s in sentences if len(s.split()) >= 5]


def clean_sentence(sentence):
    # Clean sentence: lowercase, remove filler words (TYPE 5), deduplicate words (TYPE 6), strip punctuation
    text  = sentence.lower().replace("\n", " ")
    words = text.split()
    cleaned = []
    for word in words:
        word = re.sub(r"[^\w']", "", word)
        if not word or word in FILLER_WORDS:
            continue
        if cleaned and word == cleaned[-1]:
            continue
        cleaned.append(word)
    return " ".join(cleaned).strip()


# ── Stage 1: Semantic Scoring ─────────────────────────────────────────────────

def embed_sentences(sentences):
    return embedding_model.encode(sentences, show_progress_bar=False)


def score_all_boundaries(embeddings):
    # Compute cosine similarity between adjacent sentence pairs
    candidates = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(
            embeddings[i].reshape(1, -1),
            embeddings[i + 1].reshape(1, -1)
        )[0][0]
        if sim < SEMANTIC_THRESHOLD:
            candidates.append((i, round(float(sim), 3)))
    return candidates


# ── Stage 2: Two-Tier Boundary Confirmation ───────────────────────────────────

def agentic_confirm_pair(sent_a, sent_b):
    # Ask Flan-T5 if two sentences discuss different topics
    prompt = f"""Do these two sentences discuss different topics? Answer only yes or no.

Sentence 1: {sent_a}
Sentence 2: {sent_b}

Answer:"""
    inputs  = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    outputs = llm.generate(inputs.input_ids, max_new_tokens=5, num_beams=2, early_stopping=True)
    answer  = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
    return answer.startswith("yes")


def get_confirmed_boundaries(sentences, candidates):
    # Two-tier routing: Tier 1 = auto-confirm (low similarity), Tier 2 = ask Flan-T5 (ambiguous zone)
    confirmed   = []
    tier1_count = 0
    tier2_count = 0

    print(f"    Total candidates  : {len(candidates)}")

    for idx, sim in candidates:
        if sim < AUTO_CONFIRM_THRESHOLD:
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


# ── Stage 3: Build + Guard ────────────────────────────────────────────────────

def build_raw_chunks(sentences, boundaries):
    boundary_set    = set(boundaries)
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
    # Recursively split at word midpoint until chunk ≤ MAX_CHUNK_WORDS
    words = chunk.split()
    if len(words) <= MAX_CHUNK_WORDS:
        return [chunk]
    mid = len(words) // 2
    return (recursive_split(" ".join(words[:mid])) + recursive_split(" ".join(words[mid:])))


def size_guard(chunks):
    # Merge undersized chunks, recursively split oversized chunks
    merged, buffer = [], ""
    for chunk in chunks:
        if buffer:
            combined = (buffer + " " + chunk).strip()
            if len(combined.split()) >= MIN_CHUNK_WORDS:
                merged.append(combined)
                buffer = ""
            else:
                buffer = combined
        else:
            if len(chunk.split()) < MIN_CHUNK_WORDS:
                buffer = chunk
            else:
                merged.append(chunk)
    if buffer:
        if merged: merged[-1] = (merged[-1] + " " + buffer).strip()
        else:      merged.append(buffer)
    guarded = []
    for chunk in merged:
        guarded.extend(recursive_split(chunk))
    return [c.strip() for c in guarded if c.strip()]


# ── Fallback ──────────────────────────────────────────────────────────────────

def fixed_chunk(text):
    # Fixed-size chunking with overlap (fallback method)
    words, chunks, i = text.split(), [], 0
    while i < len(words):
        chunk = " ".join(words[i:i + FIXED_CHUNK_SIZE])
        if chunk.strip():
            chunks.append(chunk)
        i += FIXED_CHUNK_SIZE - FIXED_OVERLAP
    return chunks


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def hybrid_chunk(raw_text, filename=""):
    # Full hybrid pipeline with comprehensive noise cleaning
    cleaned_raw = clean_raw_transcript(raw_text)
    raw_sentences = split_into_sentences(cleaned_raw)
    print(f"  Sentences after noise clean + filter: {len(raw_sentences)}")
    if len(raw_sentences) < MIN_SENTENCES_FOR_HYBRID:
        print(f"  ⚠ Too few sentences — using fixed fallback")
        clean  = " ".join(clean_sentence(s) for s in raw_sentences)
        chunks = fixed_chunk(clean)
        return [{"text": c, "chunk_method": "fixed_fallback", "word_count": len(c.split())} for c in chunks if c.strip()]
    cleaned = [clean_sentence(s) for s in raw_sentences]
    cleaned = [s for s in cleaned if s.strip()]
    print(f"  [Stage 1] Embedding {len(cleaned)} sentences...")
    embeddings = embed_sentences(cleaned)
    candidates = score_all_boundaries(embeddings)
    tier1 = sum(1 for _, sim in candidates if sim < AUTO_CONFIRM_THRESHOLD)
    tier2 = len(candidates) - tier1
    print(f"  [Stage 1] Candidates: {len(candidates)} (Tier1={tier1} auto, Tier2={tier2} agentic)")

    # Stage 2: Two-tier confirmation | Stage 3: Build chunks and apply size guard
    print(f"  [Stage 2] Two-tier confirmation...")
    confirmed = get_confirmed_boundaries(cleaned, candidates)
    raw_chunks   = build_raw_chunks(cleaned, confirmed)
    final_chunks = size_guard(raw_chunks)
    print(f"  [Stage 3] Raw: {len(raw_chunks)} → After size guard: {len(final_chunks)}")
    return [{"text": chunk, "chunk_method": "hybrid", "word_count": len(chunk.split())} for chunk in final_chunks if chunk.strip()]


# ── Process All ───────────────────────────────────────────────────────────────

def process_transcripts():
    all_chunks = []
    chunk_id   = 0

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
                "chunk_id":     chunk_id,
                "source_file":  filename,
                "text":         chunk["text"],
                "chunk_method": chunk["chunk_method"],
                "word_count":   chunk["word_count"],
            })
            chunk_id += 1

    os.makedirs("data/chunks", exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    wc      = [c["word_count"] for c in all_chunks]
    h_count = sum(1 for c in all_chunks if c["chunk_method"] == "hybrid")
    f_count = sum(1 for c in all_chunks if c["chunk_method"] == "fixed_fallback")
    over    = [c for c in all_chunks if c["word_count"] > MAX_CHUNK_WORDS]

    print(f"\n{'═'*60}")
    print(f"CHUNKING SUMMARY")
    print(f"{'═'*60}")
    print(f"  Total chunks        : {len(all_chunks)}")
    print(f"  Hybrid              : {h_count}")
    print(f"  Fixed fallback      : {f_count}")
    print(f"  Avg words per chunk : {sum(wc)/len(wc):.1f}")
    print(f"  Min / Max           : {min(wc)} / {max(wc)}")
    print(f"  Oversized (>{MAX_CHUNK_WORDS}w)   : {len(over)}  ← should be 0")
    print(f"\n  Saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    process_transcripts()