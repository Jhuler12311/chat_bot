# 🎵 MúsicBot — Chatbot Musical Inteligente
### Proyecto 3 · Minería de Textos · CUC · Continuación de Proyectos 1 & 2

Agente conversacional con **RAG + Fine-Tuning** sobre el corpus de letras de canciones (`tcc_ceds_music.csv`).  
Generador local: **Mistral via Ollama** — 100% sin API, sin costo.

---

## ⚡ Inicio Rápido

```bash
# 1. Clonar el repositorio
git clone https://github.com/Jhuler12311/chat_bot
cd chat_bot

# 2. Crear entorno virtual e instalar dependencias
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
pip install torch

# 3. Colocar el corpus
# Descarga tcc_ceds_music.csv de Kaggle y colócalo en data/
# https://www.kaggle.com/datasets/saurabhshahane/music-dataset-1950-to-2019

# 4. Instalar Ollama (generador local)
# Descargar desde: https://ollama.com/download
# Luego ejecutar en terminal:
ollama pull mistral

# 5. Lanzar el chatbot (Ollama debe estar corriendo en segundo plano)
python app/chatbot_app.py
```

Abrir navegador en: **http://127.0.0.1:8050/**

> **Nota**: Ollama se inicia automáticamente al instalarlo en Windows.  
> Si hay error de puerto, ya está corriendo — no hace falta ejecutar `ollama serve`.

---

## 🏗️ Arquitectura

```
chat_bot/
├── app/
│   ├── chatbot_app.py        ← Aplicación Plotly Dash (punto de entrada)
│   └── config.py             ← Variables de entorno y rutas
├── src/
│   ├── rag_utils.py          ← Chunking, embeddings, FAISS, búsqueda
│   ├── finetuning_utils.py   ← Dataset, Trainer HF, evaluación
│   └── chatbot_engine.py     ← Clase MusicChatbot (memoria + prompt + Ollama)
├── notebooks/
│   ├── 01_exploracion_corpus.ipynb       ← Análisis exploratorio del corpus
│   ├── 02_rag_pipeline.ipynb             ← Chunking + embeddings + FAISS
│   ├── 03_finetuning_clasificador.ipynb  ← Fine-tuning DistilBERT
│   └── 04_chatbot_completo.ipynb         ← Pruebas integradas del chatbot
├── data/
│   ├── tcc_ceds_music.csv    ← Corpus (mismo que Proyectos 1 y 2)
│   └── embeddings_cache/     ← Cache automático (se genera la primera vez)
├── models/
│   └── classifier/           ← Modelo fine-tuneado (DistilBERT)
├── resultados/
│   ├── metricas.json                ← 10+ conversaciones de prueba
│   ├── metricas_classifier.json     ← Accuracy, F1, matriz de confusión
│   ├── confusion_matrix.png
│   ├── chunking_comparison.png
│   └── corpus_distribucion.png
├── requirements.txt
├── README.md
└── USO_DE_IA.md
```

---

## 🤖 Generador: Ollama + Mistral (local)

El sistema usa **Mistral 7B via Ollama** como generador de lenguaje natural.  
Corre 100% en tu máquina, sin API keys ni costos.

```bash
# Instalar modelo (solo una vez, ~4GB)
ollama pull mistral

# Verificar que Ollama está activo
ollama list
```

El chatbot detecta Ollama automáticamente al arrancar.  
Si no está disponible, cae a modo **rules-based** que responde directamente de los chunks del corpus.

---

## 🎯 Pipeline RAG

```
Pregunta del usuario
      │
      ├─► Clasificador DistilBERT → detecta género (pop/rock/jazz...)
      │
      ├─► FAISS → recupera top-5 chunks más relevantes
      │
      ├─► Filtro → solo chunks del género detectado
      │
      └─► Mistral (Ollama) → genera respuesta con contexto real del corpus
```

| Componente | Tecnología | Detalle |
|---|---|---|
| Chunking | Por estrofa | Mayor precisión semántica |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 | 384 dims, multilingüe |
| Vector Store | FAISS IndexFlatIP | Búsqueda coseno |
| Clasificador | DistilBERT fine-tuned | 7 géneros musicales |
| Generador | Mistral via Ollama | 100% local |
| Memoria | Historial 5 turnos | Preguntas de seguimiento |
| Interfaz | Plotly Dash | Puerto 8050 |

---

## 🎸 Fine-Tuning del Clasificador

Modelo base: `distilbert-base-multilingual-cased`  
Tarea: clasificar letras en 7 géneros (`blues, country, hip hop, jazz, pop, reggae, rock`)

| Parámetro | Valor |
|---|---|
| Épocas | 5 |
| Learning rate | 2e-5 |
| Batch size | 16 |
| Max por clase | 700 |
| Split | 70/15/15 |
| Seed | 42 |

El clasificador filtra los chunks recuperados por FAISS para que el chatbot responda con canciones del género relevante a la pregunta.

---

## 📊 Resultados

Los resultados completos están en `resultados/`:

- `metricas_classifier.json` — accuracy, F1-macro, matriz de confusión, comparación vs baseline
- `metricas.json` — 10+ conversaciones de prueba (factuales, comparativas, seguimiento, fuera de dominio)
- Gráficas PNG de distribución del corpus, matriz de confusión y comparación de chunking

---

## ❓ Preguntas de ejemplo

| Tipo | Ejemplo |
|---|---|
| Letra | `¿Qué dice la letra de Hotel California?` |
| Artista | `Háblame sobre Bob Dylan` |
| Recomendación | `Dame una canción triste de los 80s` |
| Género | `¿Qué diferencia al blues del jazz?` |
| Seguimiento | `Dame otra del mismo género` |
| Fuera de dominio | `¿Cuánto cuesta un vuelo a Madrid?` → el bot dice que no sabe |

---

## 📦 Dependencias principales

```bash
pip install -r requirements.txt
pip install torch
```

| Paquete | Uso |
|---|---|
| `sentence-transformers` | Embeddings multilingüe |
| `faiss-cpu` | Índice vectorial |
| `transformers` + `datasets` | Fine-tuning DistilBERT |
| `dash` + `dash-bootstrap-components` | Interfaz web |
| `plotly` | Visualizaciones |
| `requests` | Comunicación con Ollama |

---

## 🗂️ Corpus

**Dataset**: [Music Dataset: Lyrics and Metadata 1950–2019](https://www.kaggle.com/datasets/saurabhshahane/music-dataset-1950-to-2019)  
**Archivo**: `tcc_ceds_music.csv`  
**Tamaño**: 28,372 canciones  
**Géneros**: Pop, Rock, Hip-Hop, Country, Jazz, Blues, Reggae  
**Período**: 1950–2019  

El mismo corpus usado en los Proyectos 1 (POS Tagging) y 2 (BoW/Word2Vec/BETO).

---

## 👤 Autor

Brayton · CUC · Minería de Textos · Profesor: Osvaldo González Chaves
