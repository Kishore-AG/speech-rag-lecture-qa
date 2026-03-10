"""
Configuration file for Speech-RAG Lecture QA System
This file centralizes all hardcoded values to make the system generalized and flexible.
"""

import os

# ============================================================================
# DIRECTORY CONFIGURATION
# ============================================================================
BASE_DATA_DIR = "data"
AUDIO_DIR = os.path.join(BASE_DATA_DIR, "audio")
TRANSCRIPT_DIR = os.path.join(BASE_DATA_DIR, "transcripts")
CHUNKS_DIR = os.path.join(BASE_DATA_DIR, "chunks")
INDEXES_DIR = os.path.join(BASE_DATA_DIR, "indexes")

# Ensure all directories exist
for directory in [AUDIO_DIR, TRANSCRIPT_DIR, CHUNKS_DIR, INDEXES_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# FILE PATHS
# ============================================================================
CHUNKS_FILE = os.path.join(CHUNKS_DIR, "chunks.json")
INDEX_FILE = os.path.join(INDEXES_DIR, "faiss_index.bin")
METADATA_FILE = os.path.join(INDEXES_DIR, "faiss_metadata.json")

# ============================================================================
# AUDIO PROCESSING (ASR) CONFIGURATION
# ============================================================================
ASR_CONFIG = {
    # Supported audio formats (can be extended based on ffmpeg availability)
    "supported_formats": [".wav", ".mp3", ".m4a", ".flac", ".ogg"],
    
    # Whisper model size: tiny, base, small, medium, large
    # Larger = better accuracy but slower and requires more memory
    "model": "base",
    
    # Language: None for auto-detect, or specify language code e.g., "en", "es"
    "language": None,
    
    # Device: "cpu" for CPU-only machines, "cuda" for GPU machines
    "device": "cpu",  # Change to "cuda" if you have GPU
}

# ============================================================================
# PREPROCESSING & CHUNKING CONFIGURATION
# ============================================================================
PREPROCESS_CONFIG = {
    # Semantic similarity thresholds for topic boundary detection
    "auto_confirm_threshold": 0.42,    # Clear topic change (strict)
    "semantic_threshold": 0.55,        # Ambiguous zone - ask LLM
    
    # Chunk size constraints (in words)
    "min_chunk_words": 40,
    "max_chunk_words": 200,
    "min_sentences_for_hybrid": 15,
    "fixed_chunk_size": 180,
    "fixed_overlap": 45,
    
    # Filler words to remove (true ASR noise with no content)
    "filler_words": {
        "um", "uh", "hmm", "hm", "ah", "er", "eh", "umm", "uhh", "mhm",
        "yeah", "ok", "okay", "right", "uh-huh", "mm-hmm"
    },
    
    # Embedding model for semantic similarity
    "embedding_model": "BAAI/bge-small-en-v1.5",
    
    # LLM for topic boundary detection
    "llm_model": "google/flan-t5-large",
}

# ============================================================================
# EMBEDDING & INDEXING CONFIGURATION
# ============================================================================
INDEXING_CONFIG = {
    # Embedding model (must match preprocess config for consistency)
    "embedding_model": "BAAI/bge-small-en-v1.5",
    
    # FAISS index type (IndexFlatL2 for exact L2 distance search)
    "index_type": "IndexFlatL2",
    
    # Number of parallel workers for encoding
    "batch_size": 32,
}

# ============================================================================
# RETRIEVAL & QA CONFIGURATION
# ============================================================================
QA_CONFIG = {
    # Retrieval parameters
    "top_k_retrieval": 50,      # Initial retrieval
    "top_k_rerank": 15,         # After reranking
    "top_k_context": 3,         # Context chunks passed to LLM
    
    # Reranking parameters
    "relevance_weight": 0.9,    # Weight for relevance score
    "diversity_weight": 0.1,    # Weight for diversity score
    
    # Embedding model for retrieval
    "embedding_model": "BAAI/bge-small-en-v1.5",
    
    # Reranker model
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    
    # LLM for QA (with fallback)
    "primary_llm": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "fallback_llm": "microsoft/Phi-3-mini-4k-instruct",
    
    # Generation parameters
    "max_new_tokens": 150,
    "temperature": 0.3,
    "top_p": 0.9,
    "num_beams": 1,
    
    # Question processing
    "stopwords": {
        "what", "is", "the", "a", "an", "in", "of", "to", "are",
        "tell", "me", "can", "you", "please", "describe", "explain",
        "and", "or", "for", "about", "from", "with", "by", "on", "at",
        "as", "be", "been", "being", "do", "does", "did", "will",
        "would", "should", "could", "have", "has", "had", "this", "that"
    },
}

# ============================================================================
# SUMMARIZATION CONFIGURATION
# ============================================================================
# ============================================================================
# SUMMARIZATION CONFIGURATION
# ============================================================================
SUMMARIZATION_CONFIG = {
    # Model for summarization - using Phi-3-mini (same as QA)
    "model": "microsoft/Phi-3-mini-4k-instruct",
    
    # Whether this is an instruction-tuned model (requires chat template)
    "is_instruction_model": True,
    
    # Summarization parameters for individual chunks
    "chunk_max_length": 150,        # Max words per chunk summary
    "chunk_min_length": 40,          # Min words per chunk summary
    
    # Summarization parameters for final summary
    "final_max_length": 400,         # Max words for overall summary
    "final_min_length": 100,          # Min words for overall summary
    
    # Hierarchical summarization (for very long lectures)
    "hierarchical_batch_size": 5,     # Number of chunk summaries to combine at once
    "hierarchical_max_words": 1000,    # Threshold for triggering hierarchical mode
    
    # Generation settings for Phi-3-mini
    "generation_config": {
        "temperature": 0.3,
        "top_p": 0.9,
        "do_sample": False,
        "max_new_tokens": 512,
        "num_beams": 1
    },
    
    # Truncation settings
    "max_chunk_words_for_prompt": 500,  # Truncate chunks longer than this
    
    # Device settings
    "device_map": "auto",
    "load_in_8bit": True,
    "torch_dtype": "float16",
    
    # Fallback settings
    "extractive_fallback": {
        "max_sentences": 3,
        "max_words": 150
    }
}

# ============================================================================
# PROMPT TEMPLATES (Generalized for any lecture topic)
# ============================================================================

# Generic topic boundary detection prompt
TOPIC_BOUNDARY_PROMPT = """You are analyzing a lecture transcript to identify topic boundaries.

Given two consecutive sentences from a lecture, determine if they discuss fundamentally different topics.

Guidelines:
- YES: If the topics are clearly different (e.g., switching from "algae photosynthesis" to "climate change")
- NO: If both sentences continue discussing the same concept or closely related ideas
- NO: If sentence 2 merely elaborates, provides examples, or adds details to sentence 1's topic

Sentence 1: {sent_a}

Sentence 2: {sent_b}

Answer only with: yes or no"""

# Generic chunk summarization prompt
CHUNK_SUMMARIZATION_PROMPT = """
Summarize the following lecture segment clearly.

Focus on:
- key discussion points
- important decisions
- project details

Lecture segment:
{text}

Summary:
"""

# Generic final summary prompt
FINAL_SUMMARY_PROMPT = FINAL_SUMMARY_PROMPT = """
The following are summaries of lecture segments.

Combine them into a clear and coherent lecture summary.

Do not repeat sentences.
Do not invent information.
Only use information from the summaries.

Lecture segment summaries:
{combined}

Final lecture summary:
"""

# Generic QA prompt
QA_PROMPT = """Answer the question based ONLY on the lecture transcript provided.

IMPORTANT GUIDELINES:
- Use ONLY information present in the transcript
- DO NOT use external knowledge or make assumptions
- If the answer is not covered, say "This topic is not covered in the lecture"
- Write exactly ONE complete sentence
- Be specific and reference lecture details when relevant
- DO NOT ramble or provide multiple sentences

Lecture transcript:
{context}

Question: {question}

Answer:"""

# Query expansion prompts for generalized question paraphrasing
QUERY_PARAPHRASE_PROMPT = """Paraphrase this lecture question to improve search accuracy.
Keep the same meaning but use different phrasing.

Original question: {question}

Paraphrased question:"""

CONTEXT_VARIANT_PROMPT = """Given these topic keywords from a lecture question, complete this sentence that a speaker might say:

Keywords: {keywords}

The speaker discusses [COMPLETE WITH 3-4 WORDS]:"""

# ============================================================================
# LOGGING & DEBUG SETTINGS
# ============================================================================
DEBUG = False
VERBOSE = True
LOG_LEVEL = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
