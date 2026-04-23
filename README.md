# MúsicBot — Chatbot Musical Inteligente
### Proyecto 3 · Minería de Textos · CUC · Continuación de Proyectos 1 & 2

Agente conversacional con **RAG + Fine-Tuning** sobre el corpus de letras de canciones (`tcc_ceds_music.csv`).  
Generador local: **Mistral via Ollama** — 100% sin API, sin costo.

---

## Inicio Rápido

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

## Arquitectura

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

## Generador: Ollama + Mistral (local)

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

## Pipeline RAG

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

## Fine-Tuning del Clasificador

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

## Resultados

Los resultados completos están en `resultados/`:

- `metricas_classifier.json` — accuracy, F1-macro, matriz de confusión, comparación vs baseline
- `metricas.json` — 10+ conversaciones de prueba (factuales, comparativas, seguimiento, fuera de dominio)
- Gráficas PNG de distribución del corpus, matriz de confusión y comparación de chunking

---

## Preguntas de ejemplo

| Tipo | Ejemplo |
|---|---|
| Letra | `¿Qué dice la letra de Hotel California?` |
| Artista | `Háblame sobre Bob Dylan` |
| Recomendación | `Dame una canción triste de los 80s` |
| Género | `¿Qué diferencia al blues del jazz?` |
| Seguimiento | `Dame otra del mismo género` |
| Fuera de dominio | `¿Cuánto cuesta un vuelo a Madrid?` → el bot dice que no sabe |

---

## Dependencias principales

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

## Corpus

**Dataset**: [Music Dataset: Lyrics and Metadata 1950–2019](https://www.kaggle.com/datasets/saurabhshahane/music-dataset-1950-to-2019)  
**Archivo**: `tcc_ceds_music.csv`  
**Tamaño**: 28,372 canciones  
**Géneros**: Pop, Rock, Hip-Hop, Country, Jazz, Blues, Reggae  
**Período**: 1950–2019  

El mismo corpus usado en los Proyectos 1 (POS Tagging) y 2 (BoW/Word2Vec/BETO).

---

## Limitaciones conocidas y mejoras propuestas

### 1. Clasificador de género con sesgo hacia la clase mayoritaria
El clasificador fine-tuneado tiende a predecir "hip hop" en la mayoría de las consultas, independientemente del contenido de la pregunta. Esto ocurre porque el modelo fue entrenado sobre letras de canciones, pero en producción recibe preguntas en lenguaje natural — un dominio muy diferente al de entrenamiento.

**Mejora propuesta:** Recolectar un dataset de intenciones reales — preguntas escritas por personas como "dame una canción de jazz" o "qué diferencia al rock del blues" — y hacer un segundo fine-tuning de clasificación de intención sobre esas frases. Alternativamente, usar un modelo de zero-shot classification como `facebook/bart-large-mnli` para detectar el género mencionado directamente en la pregunta sin necesidad de reentrenamiento.

---

### 2. Preguntas fuera de dominio no son rechazadas correctamente
Cuando RAG está activo, el chatbot responde preguntas completamente ajenas a la música (como "¿cuánto cuesta un vuelo a Madrid?" o "¿quién ganó el mundial 2022?") devolviendo canciones vagamente relacionadas por similitud semántica superficial, en lugar de indicar que la pregunta está fuera de su dominio.

**Mejora propuesta:** Implementar un paso de detección de dominio antes del pipeline RAG, usando un clasificador binario ligero (dentro/fuera de dominio musical) o un threshold mínimo de similitud coseno: si el score del chunk más relevante es menor a un umbral definido (por ejemplo, 0.25), el chatbot responde con un mensaje predefinido de "no tengo información sobre eso".

---

### 3. Letras preprocesadas en lugar de texto original
El corpus `tcc_ceds_music.csv` contiene letras que ya fueron lematizadas y limpiadas (sin signos de puntuación, palabras en forma base). Esto hace que los fragmentos citados en las respuestas se vean poco naturales y difíciles de leer para el usuario final.

**Mejora propuesta:** Enriquecer el dataset con una versión paralela de las letras en su forma original, manteniendo el texto lematizado solo para embeddings y búsqueda, pero citando el texto original en las respuestas. Esto requeriría cruzar el corpus con una fuente de letras completas como Genius API o LyricsGenius.

---

### 4. Dependencia de Ollama en segundo plano
El modo de generación de calidad requiere que Ollama esté corriendo localmente con el modelo Mistral descargado (~4 GB). Si el servicio no está activo, el sistema cae a un modo rules-based que simplemente concatena fragmentos del corpus sin generar lenguaje natural fluido. Esto puede sorprender al usuario si no leyó las instrucciones de instalación.

**Mejora propuesta:** Añadir un mensaje de advertencia claro en la interfaz cuando Ollama no está disponible, e incluir un script de verificación (`setup_check.py` ya existe en el repo) que se ejecute automáticamente antes de lanzar la app y guíe al usuario paso a paso si falta alguna dependencia.

---

### 5. Corpus en inglés con interfaz en español
La mayoría de las canciones del corpus son en inglés, pero el chatbot responde en español. Esto genera una experiencia inconsistente: el bot explica en español pero cita fragmentos de letras en inglés sin traducción ni contexto.

**Mejora propuesta:** Añadir un campo de idioma detectado por canción al momento de construir los chunks, y opcionalmente ofrecer una traducción automática de los fragmentos citados usando `Helsinki-NLP/opus-mt-en-es` (modelo ligero de traducción inglés→español disponible en Hugging Face) antes de mostrarlos al usuario.

---

## Autor

Brayton · CUC · Minería de Textos · Profesor: Osvaldo González Chaves
