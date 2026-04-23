"""
config.py — Variables de entorno, rutas y configuración global
Proyecto 3 - Chatbot Musical CUC
"""

import os
from pathlib import Path

# ── Rutas base ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "resultados"
CACHE_DIR = DATA_DIR / "embeddings_cache"

# ── Corpus ────────────────────────────────────────────────────────────────────
# Ruta actualizada a tu ubicación específica
CORPUS_CSV = r"C:\Users\98248\Downloads\PYCHAR\chatbot\data\tcc_ceds_music.csv"

# ── APIs (opcionales) ─────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ── Parámetros RAG ────────────────────────────────────────────────────────────
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "estrofa")   # "estrofa" | "cancion"
TOP_K = int(os.getenv("TOP_K", "5"))

# ── Parámetros Fine-Tuning ────────────────────────────────────────────────────
FT_BASE_MODEL = "distilbert-base-multilingual-cased"
FT_EPOCHS = int(os.getenv("FT_EPOCHS", "3"))
FT_BATCH_SIZE = int(os.getenv("FT_BATCH_SIZE", "16"))
FT_MAX_LEN = int(os.getenv("FT_MAX_LEN", "128"))
FT_SEED = 42

# ── App ───────────────────────────────────────────────────────────────────────
APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("APP_PORT", "8050"))
APP_DEBUG = os.getenv("APP_DEBUG", "false").lower() == "true"
