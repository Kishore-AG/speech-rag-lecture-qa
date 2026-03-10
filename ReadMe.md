# Speech RAG Lecture Q&A System

## Project Overview

**Speech RAG Lecture Q&A** is an advanced Retrieval-Augmented Generation (RAG) system designed to answer questions about lecture content using transcribed audio. The system combines automatic speech recognition, semantic chunking, hybrid retrieval (semantic + keyword-based), large language models, and hierarchical summarization to provide accurate, contextually-grounded answers and full lecture summaries.

This project demonstrates a production-ready pipeline for processing lecture transcripts, building searchable knowledge bases, and enabling natural language question-answering and summarization over lecture materials.

---

## What We've Built (Completed Components)

### 1. **Automatic Speech Recognition (ASR)**
- **File**: `asr.py`
- **Technology**: OpenAI Whisper (base model)
- **Functionality**:
  - Converts audio files (WAV, MP3, M4A, FLAC, OGG) from `data/audio/` to text transcripts
  - Saves transcripts to `data/transcripts/`
  - Supports batch processing of multiple audio files
  - Skips files with existing transcripts (incremental)
  - Handles various audio qualities automatically

### 2. **Advanced Text Preprocessing with Hybrid Chunking**
- **File**: `preprocess.py`
- **Key Features**:
  - **Comprehensive Noise Cleaning** (6 types of ASR artifacts removed):
    - Isolated short utterances (e.g., "Okay.", "Yeah.")
    - Repeated sentence blocks (Whisper hallucinations)
    - Non-ASCII/Cyrillic character removal
    - Letter-by-letter spelled names (e.g., "T-A-R-I-K")
    - ASR filler words (um, uh, hmm, mhm)
    - Consecutive word deduplication
  - **Semantic Chunking Pipeline** (3-stage):
    - **Stage 1**: Embeds sentences, identifies low-similarity boundaries using cosine similarity
    - **Stage 2**: Two-tier boundary confirmation (auto-confirm vs. Flan-T5 agentic confirmation)
    - **Stage 3**: Size-guarded chunks with recursive splitting for oversized content
  - **Fallback Strategy**: Fixed-size chunking with overlap for transcripts with insufficient sentences
  - Output: Structured JSON with chunks, source files, and metadata

### 3. **Semantic Vector Indexing**
- **File**: `build_index.py`
- **Technology**:
  - SentenceTransformer (BGE-small-en-v1.5) for embeddings
  - FAISS for efficient vector search
- **Functionality**:
  - Encodes all text chunks into 384-dimensional embeddings
  - Builds FAISS IndexFlatL2 for semantic similarity search
  - Saves index and metadata to `data/indexes/`
  - Optimized for CPU/GPU deployment

### 4. **Hybrid Question-Answering System**
- **File**: `qa.py`
- **Advanced Features**:

  **Query Processing**:
  - Query expansion with keyword extraction and synonym generation
  - Context variants generated from topic keywords
  - Automatic deduplication of variant queries

  **Hybrid Retrieval** (Semantic + Keyword):
  - **Semantic Path**: FAISS vector similarity with L2 normalization
  - **Keyword Path**: BM25 ranking for exact phrase matching
  - **Fusion**: Weighted scoring (90% semantic, 10% keyword) with deduplication
  - Adaptive fallback to semantic-only if BM25 unavailable

  **Reranking & Refinement**:
  - Cross-encoder reranking (ms-marco-MiniLM) for relevance scoring
  - Diversity scoring based on chunk length (prevents repetition)
  - Selects top-15 most relevant chunks

  **Answer Generation**:
  - Causal LLM integration (Meta-Llama-3.1-8B or Microsoft Phi-3-mini with fallback)
  - RAG-grounded answer generation (3-chunk context limit for focus)
  - Strict single-sentence output format
  - Temperature-controlled generation (0.3 for factual answers)

### 5. **Hierarchical Lecture Summarization** *(New)*
- **File**: `summary.py`
- **Technology**: Microsoft Phi-3-mini-4k-instruct
- **Functionality**:
  - Loads all preprocessed chunks and joins them into a full transcript
  - Splits transcript into large token-based sections (~2500 tokens each) to fit model context
  - Summarizes each section independently using strict factual rules
  - Combines section summaries into a single coherent final lecture summary
  - Uses Phi-3's **native chat template** (`apply_chat_template`) for correct instruction following
  - `repetition_penalty` set to `1.0` (disabled) to prevent synonym hallucination artifacts
  - `do_sample=False` for deterministic, factual output
- **Output**: Printed final summary with clear section boundaries

---

## System Architecture

### Overall Pipeline
```
Audio Files
      ↓
  [ASR Module] (whisper)
      ↓
Transcripts (TXT)
      ↓
[Preprocessing] (noise cleaning → semantic chunking)
      ↓
Chunks + Metadata (JSON)
      ↓
[Embedding Generation] (SentenceTransformer)
      ↓
FAISS Index + Metadata (data/indexes/)
      ↓
    ┌──────────────────────┐
    │                      │
[QA System]         [Summarization]
(hybrid retrieval    (hierarchical
→ reranking →        section-based
→ LLM generation)    → Phi-3-mini)
    │                      │
Grounded Answers     Lecture Summary
```

### Component Architecture

#### **1. ASR Component**
```
Audio Input → Whisper Model → Text Transcript
```

#### **2. Preprocessing Pipeline**
```
Raw Transcript
      ↓
[Clean Raw Text]
  ├─ Strip non-ASCII
  ├─ Remove spelled names
  ├─ Deduplicate sentences
  └─ Remove isolated utterances
      ↓
[Sentence Segmentation]
      ↓
[Semantic Chunking]
  ├─ Stage 1: Embedding & boundary scoring
  ├─ Stage 2: Two-tier confirmation (AUTO | AGENTIC)
  └─ Stage 3: Size-guard (merge undersized, split oversized)
      ↓
Chunks with Method Tags (hybrid/fixed_fallback)
```

#### **3. Vector Indexing**
```
Chunks → [SentenceTransformer] → Embeddings (384-dim)
              ↓
[FAISS IndexFlatL2] ← Efficient vector search
              ↓
Saved to data/indexes/ (faiss_index.bin + faiss_metadata.json)
```

#### **4. Question-Answering Architecture**
```
User Question
      ↓
[Query Processing]
  ├─ Keyword extraction
  ├─ Synonym generation (paraphrase)
  └─ Query variants (4-8 unique queries)
      ↓
[Hybrid Retrieval]
  ├─ FAISS semantic search (top-50 per query)
  ├─ BM25 keyword search (top-50 per query)
  └─ Weighted fusion + dedup → combined top-50
      ↓
[Reranking]
  ├─ Cross-encoder relevance scoring
  ├─ Diversity scoring (length distribution)
  └─ Top-15 selected
      ↓
[Answer Generation]
  ├─ Context assembly (3-chunk limit)
  ├─ LLM generation (Llama 3.1 8B or Phi-3-mini)
  └─ Output: single complete sentence
      ↓
Grounded Answer
```

#### **5. Summarization Architecture** *(New)*
```
chunks.json
      ↓
[Token-Based Section Splitting] (~2500 tokens/section)
      ↓
[Per-Section Summarization]
  ├─ Phi-3-mini via native chat template
  ├─ do_sample=False (deterministic)
  └─ repetition_penalty=1.0 (hallucination fix)
      ↓
[Final Summary Generation]
  └─ Combines all section summaries → coherent full summary
      ↓
Printed Final Lecture Summary
```

---

## Data Flow and File Structure
```
data/
├── audio/                    # Input: RAW AUDIO FILES
│   ├── lecture_1.wav
│   ├── lecture_2.wav
│   └── lecture_3.wav
│
├── transcripts/              # GENERATED: ASR OUTPUT
│   ├── lecture_1.txt
│   ├── lecture_2.txt
│   └── lecture_3.txt
│
├── chunks/                   # GENERATED: PREPROCESSED CHUNKS
│   └── chunks.json           # Main chunk database
│
└── indexes/                  # GENERATED: VECTOR INDEX (New)
    ├── faiss_index.bin
    └── faiss_metadata.json

src/
├── asr.py                    # Audio → Transcripts
├── preprocess.py             # Transcripts → Chunks (hybrid semantic chunking)
├── build_index.py            # Chunks → FAISS index
├── qa.py                     # Index → Answers (hybrid retrieval + RAG)
├── summary.py                # Chunks → Lecture Summary (hierarchical, New)
└── config.py                 # Centralized configuration

requirements.txt
README.md
```

---

## Technologies & Models Used

### Core Dependencies
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **ASR** | OpenAI Whisper (base) | Speech-to-text transcription |
| **Embeddings** | BAAI BGE-small-en-v1.5 | 384-dim semantic embeddings |
| **Vector Search** | FAISS (CPU/GPU) | Efficient similarity retrieval |
| **Keyword Search** | Rank-BM25 | Traditional keyword matching |
| **Reranking** | Cross-Encoder (ms-marco-MiniLM) | Relevance scoring |
| **QA LLM** | Meta-Llama-3.1-8B or Phi-3-mini | Answer generation with fallback |
| **Summarization LLM** | Microsoft Phi-3-mini-4k-instruct | Hierarchical lecture summarization |
| **NLP Toolkit** | Hugging Face Transformers | Model loading & inference |
| **Sentence Processing** | SentenceTransformers | Semantic embeddings |

### Hardware Requirements
- **ASR**: GPU recommended (>4GB VRAM for Whisper base)
- **Indexing**: CPU-only or GPU (FAISS supports both)
- **QA Inference**: 8GB+ GPU VRAM (Llama) or 2-4GB (Phi-3-mini)
- **Summarization**: 2-4GB GPU VRAM (Phi-3-mini, `device_map="auto"`)
- **Fallback**: All components support CPU-only inference

---

## Key Design Decisions & Optimizations

### 1. **Hybrid Semantic Chunking**
**Problem**: Fixed-size chunking loses semantic boundaries; naive semantic chunking misses boundaries.
**Solution**: Two-tier confirmation strategy:
- Tier 1: Auto-confirm clear boundaries (similarity < 0.42)
- Tier 2: Use Flan-T5 agentic confirmation for ambiguous zones (0.42 ≤ similarity < 0.55)
- **Benefit**: Balances accuracy vs. LLM inference cost

### 2. **Comprehensive Noise Cleaning**
**Problem**: Whisper produces hallucinations on unclear audio (Cyrillic, repeated sentences, spelled names).
**Solution**: Pattern-based cleaning (no hardcoded word lists):
- Non-ASCII stripping (regex-based Unicode range match)
- Sentence-level deduplication with sliding window
- Isolated utterance removal (structural, not semantic)

### 3. **Hybrid Retrieval (Semantic + Keyword)**
**Problem**: Semantic search alone misses exact phrase matches; keyword search can't handle reformulations.
**Solution**: Weighted fusion with deduplication:
- Semantic results: 1.0 base score, tempered by document ranking position
- Keyword results: 0.4 base score (secondary signal), only added if not already semantic-retrieved

### 4. **Strict Answer Format**
**Problem**: LLMs pad under-constrained prompts with boilerplate.
**Solution**:
- Focused context: top-3 chunks only (~600 words)
- Strict format: "Write exactly one complete sentence"
- 150-token max cap

### 5. **LLM Fallback Strategy**
**Problem**: Llama 3.1 8B requires 8GB+ VRAM; may OOM on limited hardware.
**Solution**: Graceful fallback:
1. Attempt Llama-3.1-8B (bfloat16, 8-bit quantization)
2. On OOM: Fall back to Phi-3-mini (float16, 4K context)

### 6. **Hierarchical Summarization with Chat Template** *(New)*
**Problem**: Full transcripts exceed model context limits; naive summarization hallucinates synonyms.
**Solution**:
- Token-based section splitting (~2500 tokens) so each section fits Phi-3-mini's 4K context
- Native `apply_chat_template` used instead of manual prompt formatting — ensures Phi-3 follows instructions correctly
- `repetition_penalty=1.0` (disabled) — fixes a bug where the model was substituting bizarre synonyms due to over-penalization
- `do_sample=False` — greedy decoding for factual, deterministic summaries
- Two-stage output: per-section summaries → combined final summary

---

## End-to-End Workflow

### Step 1: Transcribe Audio
```bash
python src/asr.py
```
Outputs: `data/transcripts/lecture_N.txt`

### Step 2: Preprocess & Chunk
```bash
python src/preprocess.py
```
Outputs: `data/chunks/chunks.json`

### Step 3: Build FAISS Index
```bash
python src/build_index.py
```
Outputs: `data/indexes/faiss_index.bin`, `data/indexes/faiss_metadata.json`

### Step 4: Run Interactive QA
```bash
python src/qa.py
```
Asks for user question → Returns single-sentence answer grounded in transcripts

### Step 5: Generate Lecture Summary *(New)*
```bash
python src/summary.py
```
Outputs: Full hierarchical lecture summary printed to console

---

## Performance & Scalability

### Current Scale
- **Lectures**: Supports any number of `.txt` transcripts
- **Chunk Database**: ~100-500 chunks per lecture (depends on transcript length)
- **Search Latency**: <100ms (FAISS semantic search)
- **Answer Generation**: 2-5 seconds (LLM inference)
- **Summarization**: ~30-90 seconds per lecture (Phi-3-mini, CPU/GPU)

### Scalability
- **Horizontal**: Add lectures → rerun preprocessing + indexing
- **Retrieval**: FAISS scales to millions of vectors
- **Inference**: Batch question processing available
- **Memory**: Metadata stored in JSON; FAISS index size = #chunks × 384 bytes × 4

---

## Future Enhancements

1. **Incremental Indexing**: Add new transcripts without full reindexing
2. **Multi-Lecture Cross-Search**: Query across multiple lectures with source attribution
3. **Interactive Refinement**: Follow-up questions with conversation context
4. **Fine-tuned Models**: Domain-specific embedding model for lecture corpus
5. **Confidence Scores**: Output relevance & verifiability scores with answers
6. **Web Interface**: Streamlit/Flask UI for non-technical access
7. **Summary Export**: Save summaries to file (TXT/PDF) instead of console-only output

---

## How to Use

### Prerequisites
```bash
pip install -r requirements.txt
```

### Full Pipeline
```bash
python src/asr.py
python srcp/reprocess.py
python src/build_index.py
python src/qa.py        # Interactive Q&A
python src/summary.py   # Lecture summary
```

---

## Common Issues & Troubleshooting

1. **CUDA OOM on Llama**:
   - Automatic fallback to Phi-3-mini enabled
   - Manual override: change `primary_llm` in `config.py`

2. **No Transcripts Found**:
   - Ensure audio files exist in `data/audio/`
   - Supported formats: `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`

3. **Slow FAISS Search**:
   - FAISS CPU uses IndexFlatL2 (exact search)
   - For >1M vectors, switch to IndexIVF or GPU FAISS

4. **Summary Has Weird Synonyms / Repetition**:
   - Ensure `repetition_penalty=1.0` in `summary.py` (already set)
   - This was a known bug — fixed in current version

---

## License & Attribution

- Whisper: OpenAI (MIT License)
- FAISS: Meta (MIT License)
- BGE-small embeddings: BAAI (MIT License)
- Cross-Encoder: Hugging Face (Apache 2.0)
- Llama 3.1: Meta (Llama 2 Community License)
- Phi-3: Microsoft (MIT License)

---

**Last Updated**: March 11, 2026  
**Status**: Production-Ready (v0.47.0)