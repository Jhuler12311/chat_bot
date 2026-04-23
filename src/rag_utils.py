"""
rag_utils.py — Pipeline RAG con detección robusta de columnas del dataset
Proyecto 3 - Chatbot Musical CUC

Compatible con tcc_ceds_music.csv (columnas: track_name, artist_name, genre,
release_date, lyrics) y cualquier variante del mismo corpus.
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# ── Rutas de caché ────────────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).parent.parent / "data" / "embeddings_cache"
INDEX_FILE = CACHE_DIR / "faiss_index.idx"
CHUNKS_FILE = CACHE_DIR / "chunks.pkl"
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 5

# ─────────────────────────────────────────────────────────────────────────────
# Detección robusta de columnas del CSV
# ─────────────────────────────────────────────────────────────────────────────

def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Detecta automáticamente las columnas relevantes del dataset.
    Funciona con tcc_ceds_music.csv y variantes.
    """
    cols = {c.lower(): c for c in df.columns}

    def find(candidates):
        for c in candidates:
            if c in cols:
                return cols[c]
        return None

    mapping = {
        "lyrics":  find(["lyrics", "letra", "lyric", "text", "song_text", "content"]),
        "title":   find(["track_name", "title", "song_name", "name", "titulo", "song"]),
        "artist":  find(["artist_name", "artist", "artista", "singer", "performer"]),
        "genre":   find(["genre", "genero", "style", "category"]),
        "year":    find(["release_date", "year", "año", "date", "released"]),
    }

    print("[RAG] Columnas detectadas:")
    for k, v in mapping.items():
        print(f"  {k:8s} → {v or 'NO ENCONTRADA'}")

    if mapping["lyrics"] is None:
        available = list(df.columns)
        raise ValueError(
            f"No se encontró columna de letras. Columnas disponibles: {available}\n"
            "Configura manualmente en rag_utils.py o renombra la columna a 'lyrics'."
        )

    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────────────────────────────────────

def _get_val(row, col, default="?"):
    if col is None:
        return default
    val = row.get(col, default)
    return str(val).strip() if pd.notna(val) else default


def chunk_by_song(row: pd.Series, col_map: Dict) -> List[Dict]:
    """Estrategia A: canción completa = 1 chunk."""
    letra = _get_val(row, col_map["lyrics"], "")
    if not letra or letra in ("nan", "?", "") or len(letra) < 40:
        return []
    return [{
        "texto":   letra[:2500],
        "titulo":  _get_val(row, col_map["title"]),
        "artista": _get_val(row, col_map["artist"]),
        "genero":  _get_val(row, col_map["genre"]),
        "año":     _get_val(row, col_map["year"]),
        "estrategia": "cancion_completa",
    }]


def chunk_by_strophe(row: pd.Series, col_map: Dict, min_chars: int = 50) -> List[Dict]:
    """Estrategia B: cada estrofa = 1 chunk."""
    letra = _get_val(row, col_map["lyrics"], "")
    if not letra or letra in ("nan", "?", "") or len(letra) < 40:
        return []

    # Intentar split por doble newline
    partes = [p.strip() for p in letra.split("\n\n") if len(p.strip()) >= min_chars]
    if not partes:
        # Fallback: dividir en ventanas de ~300 chars
        words = letra.split()
        partes = [" ".join(words[i:i+50]) for i in range(0, len(words), 40)]
        partes = [p for p in partes if len(p) >= min_chars]
    if not partes:
        partes = [letra[:1200]]

    chunks = []
    for i, estrofa in enumerate(partes[:10]):
        chunks.append({
            "texto":      estrofa[:900],
            "titulo":     _get_val(row, col_map["title"]),
            "artista":    _get_val(row, col_map["artist"]),
            "genero":     _get_val(row, col_map["genre"]),
            "año":        _get_val(row, col_map["year"]),
            "estrofa_idx": i,
            "estrategia": "estrofa",
        })
    return chunks


def build_chunks(df: pd.DataFrame, estrategia: str = "estrofa") -> List[Dict]:
    col_map = detect_columns(df)
    chunks = []
    fn = chunk_by_strophe if estrategia == "estrofa" else chunk_by_song

    for _, row in df.iterrows():
        chunk_list = fn(row, col_map) if estrategia == "estrofa" else chunk_by_song(row, col_map)
        # Para estrofas, también guardamos la canción completa como chunk raíz
        # para que búsquedas por artista/título funcionen bien
        if estrategia == "estrofa":
            root = chunk_by_song(row, col_map)
            chunks.extend(root)        # canción completa primero
            chunks.extend(chunk_list)  # luego estrofas
        else:
            chunks.extend(chunk_list)

    # Filtrar chunks vacíos
    chunks = [c for c in chunks if c["texto"] and len(c["texto"]) > 30]

    print(f"[RAG] Total chunks generados: {len(chunks)}")
    # Stats
    con_letra = sum(1 for c in chunks if len(c["texto"]) > 100)
    print(f"[RAG] Chunks con letra sustancial (>100 chars): {con_letra}")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────

_model_cache = None


def get_embed_model():
    global _model_cache
    if _model_cache is None:
        if not ST_AVAILABLE:
            raise ImportError("sentence-transformers no está instalado. "
                              "Ejecuta: pip install sentence-transformers")
        print(f"[RAG] Cargando modelo: {EMBED_MODEL_NAME}")
        _model_cache = SentenceTransformer(EMBED_MODEL_NAME)
    return _model_cache


def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    model = get_embed_model()
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # coseno
        convert_to_numpy=True,
    )
    return vecs.astype("float32")


def embed_query(query: str) -> np.ndarray:
    model = get_embed_model()
    vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    return vec.astype("float32")


# ─────────────────────────────────────────────────────────────────────────────
# FAISS
# ─────────────────────────────────────────────────────────────────────────────

def build_index(embeddings: np.ndarray):
    if not FAISS_AVAILABLE:
        raise ImportError("faiss-cpu no está instalado. Ejecuta: pip install faiss-cpu")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner Product = coseno con vectores normalizados
    index.add(embeddings)
    print(f"[RAG] Índice FAISS: {index.ntotal} vectores, dim={dim}")
    return index


def save_index(index, chunks: List[Dict]):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)
    print(f"[RAG] Caché guardado en {CACHE_DIR}")


def load_index() -> Tuple:
    if not INDEX_FILE.exists() or not CHUNKS_FILE.exists():
        raise FileNotFoundError("No hay índice en caché. Ejecuta build_rag_pipeline primero.")
    index = faiss.read_index(str(INDEX_FILE))
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    print(f"[RAG] Índice cargado: {index.ntotal} vectores, {len(chunks)} chunks")
    return index, chunks


def index_exists() -> bool:
    return INDEX_FILE.exists() and CHUNKS_FILE.exists()


# ─────────────────────────────────────────────────────────────────────────────
# BÚSQUEDA SEMÁNTICA
# ─────────────────────────────────────────────────────────────────────────────

def search(query: str, index, chunks: List[Dict],
           top_k: int = TOP_K,
           genero_filter: Optional[str] = None) -> List[Dict]:
    """
    Busca los top_k chunks más relevantes para la query.
    Aplica filtro de género si se especifica.
    Deduplicación: si hay varias estrofas del mismo chunk, prefiere la canción completa.
    """
    q_vec = embed_query(query)
    k_search = min(top_k * 6, len(chunks))
    scores, indices = index.search(q_vec, k_search)

    results = []
    seen_titles = set()

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = dict(chunks[idx])
        chunk["score"] = float(score)

        # Filtro de género
        if genero_filter and genero_filter.lower() not in chunk.get("genero", "").lower():
            continue

        # Deduplicación: preferir canción completa sobre estrofa del mismo título
        title_key = f"{chunk['titulo']}|{chunk['artista']}"
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)

        results.append(chunk)
        if len(results) >= top_k:
            break

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE COMPLETO
# ─────────────────────────────────────────────────────────────────────────────

def build_rag_pipeline(df: pd.DataFrame,
                       estrategia: str = "estrofa",
                       force: bool = False):
    """
    Construye o carga el pipeline RAG.
    Primer run: ~5-15 min según tamaño del corpus.
    Runs siguientes: carga instantánea desde caché.
    """
    if index_exists() and not force:
        print("[RAG] Cargando desde caché (usa force=True para reconstruir)...")
        return load_index()

    print("[RAG] Construyendo pipeline desde cero...")
    chunks = build_chunks(df, estrategia)
    if not chunks:
        raise ValueError("El corpus no produjo chunks. Verifica la columna 'lyrics'.")

    texts = [c["texto"] for c in chunks]
    print(f"[RAG] Generando embeddings para {len(texts)} chunks...")
    embeddings = embed_texts(texts)

    index = build_index(embeddings)
    save_index(index, chunks)
    return index, chunks
