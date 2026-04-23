# 🎵 MúsicBot — Chatbot Musical Inteligente
### Proyecto 3 · Minería de Textos · CUC · Continuación de Proyectos 1 & 2

Agente conversacional con **RAG + Fine-Tuning** sobre el corpus de letras de canciones (`tcc_ceds_music.csv`).

---

## ⚡ Inicio Rápido

```bash
# 1. Clonar (o descomprimir) el proyecto
git clone https://github.com/Jhuler12311/proyecto3_chatbot_musical
cd proyecto3_chatbot_musical

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU
# pip install torch  # GPU

# 4. Colocar el corpus
# Copia tu tcc_ceds_music.csv en data/
cp /ruta/a/tcc_ceds_music.csv data/

# 5. Lanzar el chatbot
python app/chatbot_app.py
```

Abrir navegador en: **http://127.0.0.1:8050/**

---

## 🏗️ Arquitectura

```
proyecto3_chatbot_musical/
├── app/
│   ├── chatbot_app.py        ← Aplicación Plotly Dash (punto de entrada)
│   └── config.py             ← Variables de entorno y rutas
├── src/
│   ├── rag_utils.py          ← Chunking, embeddings, FAISS, búsqueda
│   ├── finetuning_utils.py   ← Dataset, Trainer HF, evaluación
│   └── chatbot_engine.py     ← Clase MusicChatbot (memoria + prompt + generación)
├── notebooks/
│   ├── 02_rag_pipeline.ipynb
│   ├── 03_finetuning_clasificador.ipynb
│   └── 04_chatbot_completo.ipynb
├── data/
│   ├── tcc_ceds_music.csv    ← Corpus (mismo que Proyectos 1 y 2)
│   └── embeddings_cache/     ← Cache automático (se genera la primera vez)
├── models/
│   └── classifier/           ← Modelo fine-tuneado (después del entrenamiento)
├── resultados/               ← Métricas JSON, matrices de confusión
├── requirements.txt
├── README.md
└── USO_DE_IA.md
```

---

## 🔧 Configuración

### Variables de entorno (opcionales)

```bash
# Usar Claude API como generador (en vez de Flan-T5 local)
export ANTHROPIC_API_KEY="sk-ant-..."

# Usar OpenAI
export OPENAI_API_KEY="sk-..."

# Cambiar ruta del corpus
export CORPUS_CSV="/ruta/a/mi_corpus.csv"

# Cambiar estrategia de chunking: "estrofa" o "cancion"
export CHUNKING_STRATEGY="estrofa"
```

O crear un archivo `.env` en la raíz:
```
ANTHROPIC_API_KEY=sk-ant-...
```

### Modo local (sin API)

El sistema funciona **completamente offline** usando `google/flan-t5-base` como generador.
Se descarga automáticamente de HuggingFace Hub la primera vez (~1 GB).

---

## 🎯 Pipeline RAG

1. **Chunking** — letras divididas por estrofa (doble newline) con metadatos preservados
2. **Embeddings** — `paraphrase-multilingual-MiniLM-L12-v2` (384 dims, multilingüe)
3. **FAISS** — `IndexFlatIP` con vectores normalizados (búsqueda coseno)
4. **Caché** — embeddings guardados en `data/embeddings_cache/` (se reutilizan)
5. **Generación** — Flan-T5 local o API según configuración

---

## 🤖 Fine-Tuning

Clasificador de género musical basado en `distilbert-base-multilingual-cased`.

```bash
# Ejecutar desde Jupyter o Python
from src.finetuning_utils import prepare_dataset, split_dataset, run_finetuning
import pandas as pd

df = pd.read_csv('data/tcc_ceds_music.csv')
df_labeled, genre2id = prepare_dataset(df)
df_train, df_val, df_test = split_dataset(df_labeled)
metrics = run_finetuning(df_train, df_val, df_test, genre2id)
```

> Recomendado: Google Colab con GPU T4 para ~20 min de entrenamiento.

---

## 📊 Evaluación

Los resultados se guardan automáticamente en `resultados/`:
- `metricas.json` — 10+ conversaciones de prueba con/sin RAG
- `metricas_classifier.json` — accuracy, F1-macro, matriz de confusión
- `chunking_comparison.png` — distribución de longitudes por estrategia

---

## ❓ Preguntas de ejemplo

| Tipo | Ejemplo |
|------|---------|
| Factual | `¿Qué canciones hablan de amor?` |
| Comparativa | `¿Qué diferencia al hip-hop del pop?` |
| Seguimiento | `Dame otra del mismo género` |
| Género | `Canciones de blues de los 60s` |
| Fuera de dominio | `¿Cuánto cuesta un vuelo?` → el bot dice que no sabe |

---

## 📦 Dependencias principales

| Paquete | Uso |
|---------|-----|
| `sentence-transformers` | Embeddings multilingüe |
| `faiss-cpu` | Índice vectorial |
| `transformers` | Fine-tuning y Flan-T5 |
| `dash` + `dash-bootstrap-components` | Interfaz web |
| `plotly` | Visualizaciones |

---

## 👤 Autor

Brayton · CUC · Minería de Textos · Profesor: Osvaldo González Chaves
