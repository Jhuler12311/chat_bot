"""
finetuning_utils.py — Clasificador de género por fine-tuning (DistilBERT multilingüe)
Proyecto 3 - Chatbot Musical CUC
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional

MODEL_DIR = Path(__file__).parent.parent / "models"
RESULTS_DIR = Path(__file__).parent.parent / "resultados"
BASE_MODEL = "distilbert-base-multilingual-cased"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
SEED = 42

# Géneros válidos (del dataset tcc_ceds_music)
VALID_GENRES = ["pop", "rock", "hip hop", "country", "jazz",
                "blues", "reggae", "metal", "classical", "other"]


# ─────────────────────────────────────────────────────────────────────────────
# Preparación del dataset
# ─────────────────────────────────────────────────────────────────────────────

def prepare_dataset(df: pd.DataFrame,
                    min_per_class: int = 50,
                    max_per_class: int = 500) -> pd.DataFrame:
    """Limpia y balancea el dataset para clasificación de género."""
    col_lyrics = "lyrics" if "lyrics" in df.columns else "letra"
    col_genre = "genre" if "genre" in df.columns else "genero"

    sub = df[[col_lyrics, col_genre]].dropna()
    sub = sub[sub[col_lyrics].str.len() > 50].copy()
    sub.columns = ["lyrics", "genre"]
    sub["genre"] = sub["genre"].str.lower().str.strip()

    # Normalizar géneros poco comunes
    top_genres = sub["genre"].value_counts()
    top_genres = top_genres[top_genres >= min_per_class].index.tolist()[:8]
    sub = sub[sub["genre"].isin(top_genres)]

    # Balanceo
    balanced = []
    for g in top_genres:
        part = sub[sub["genre"] == g].sample(
            min(max_per_class, len(sub[sub["genre"] == g])), random_state=SEED)
        balanced.append(part)
    result = pd.concat(balanced).sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Label encoding
    genres_sorted = sorted(result["genre"].unique())
    genre2id = {g: i for i, g in enumerate(genres_sorted)}
    result["label"] = result["genre"].map(genre2id)

    print(f"[FT] Dataset: {len(result)} filas, {len(genres_sorted)} géneros: {genres_sorted}")
    return result, genre2id


def split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split 70/15/15 con seed fijo."""
    from sklearn.model_selection import train_test_split
    train, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=SEED)
    print(f"[FT] Train={len(train)}, Val={len(val)}, Test={len(test)}")
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# Fine-Tuning con HuggingFace Trainer
# ─────────────────────────────────────────────────────────────────────────────

def run_finetuning(df_train, df_val, df_test, genre2id: Dict) -> Dict:
    """Entrena el clasificador y devuelve métricas."""
    try:
        from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                                  TrainingArguments, Trainer)
        from datasets import Dataset
        import evaluate
        import torch
    except ImportError as e:
        print(f"[FT] Dependencia faltante: {e}. Saltando fine-tuning.")
        return {}

    id2genre = {v: k for k, v in genre2id.items()}
    num_labels = len(genre2id)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize(batch):
        return tokenizer(batch["lyrics"], truncation=True, padding="max_length",
                         max_length=MAX_LEN)

    def make_hf_dataset(df):
        return Dataset.from_pandas(df[["lyrics", "label"]]).map(tokenize, batched=True)

    ds_train = make_hf_dataset(df_train)
    ds_val = make_hf_dataset(df_val)
    ds_test = make_hf_dataset(df_test)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=num_labels)

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        # Usamos sklearn directamente para evitar el error de '.size'
        from sklearn.metrics import accuracy_score, f1_score

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro")

        return {
            "accuracy": float(acc),
            "f1_macro": float(f1)
        }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        seed=SEED,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics,
    )

    print("[FT] Iniciando entrenamiento...")
    trainer.train()

    # Evaluación en test
    test_results = trainer.evaluate(ds_test)
    print(f"[FT] Test accuracy={test_results.get('eval_accuracy', '?'):.4f}, "
          f"F1={test_results.get('eval_f1_macro', '?'):.4f}")

    # Guardar modelo
    model.save_pretrained(str(MODEL_DIR / "classifier"))
    tokenizer.save_pretrained(str(MODEL_DIR / "classifier"))

    # Matriz de confusión
    preds_out = trainer.predict(ds_test)
    preds = np.argmax(preds_out.predictions, axis=-1)
    labels = preds_out.label_ids
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds,
                                   target_names=[id2genre[i] for i in range(num_labels)],
                                   output_dict=True)

    metrics = {
        "accuracy": test_results.get("eval_accuracy", 0),
        "f1_macro": test_results.get("eval_f1_macro", 0),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "genre2id": genre2id,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "metricas_classifier.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Guardar genre2id aparte
    with open(MODEL_DIR / "genre2id.pkl", "wb") as f:
        pickle.dump(genre2id, f)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Inferencia (clasificador cargado)
# ─────────────────────────────────────────────────────────────────────────────

_clf_cache = None
_tok_cache = None
_g2id_cache = None


def load_classifier():
    global _clf_cache, _tok_cache, _g2id_cache
    clf_path = MODEL_DIR / "classifier"
    g2id_path = MODEL_DIR / "genre2id.pkl"

    if not clf_path.exists():
        return None, None, None

    if _clf_cache is None:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            _tok_cache = AutoTokenizer.from_pretrained(str(clf_path))
            _clf_cache = AutoModelForSequenceClassification.from_pretrained(str(clf_path))
            _clf_cache.eval()
            if g2id_path.exists():
                with open(g2id_path, "rb") as f:
                    _g2id_cache = pickle.load(f)
            print("[FT] Clasificador cargado.")
        except Exception as e:
            print(f"[FT] No se pudo cargar el clasificador: {e}")
            return None, None, None

    return _clf_cache, _tok_cache, _g2id_cache


def predict_genre(text: str) -> Optional[str]:
    """Predice el género de un texto. Devuelve None si no hay modelo."""
    import torch
    model, tokenizer, genre2id = load_classifier()
    if model is None:
        return None
    id2genre = {v: k for k, v in genre2id.items()}
    inputs = tokenizer(text[:512], return_tensors="pt", truncation=True,
                       padding=True, max_length=MAX_LEN)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = logits.argmax(-1).item()
    return id2genre.get(pred_id, "desconocido")


def classifier_available() -> bool:
    return (MODEL_DIR / "classifier").exists()
