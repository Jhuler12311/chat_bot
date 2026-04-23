# Uso de IA en el Proyecto 3

## Herramientas utilizadas

- **Claude (Anthropic), Sonnet 4.6** — asistente principal para arquitectura y código
- **GitHub Copilot** en VS Code — autocompletado durante desarrollo

---

## Áreas de uso

- Diseño de la arquitectura del pipeline RAG (chunking → embeddings → FAISS → generación)
- Estructura del layout Plotly Dash (callbacks, stores, styled components)
- Depuración del `Trainer` de HuggingFace (gestión de memoria, `eval_strategy`)
- Generación de prompts base para el sistema del chatbot
- Estructura de los notebooks de evaluación
- Redacción inicial del README.md

---

## Prompts representativos

1. **"¿Cómo implemento búsqueda semántica con FAISS y sentence-transformers sobre un CSV de letras de canciones?"**  
   → Se usó para entender el flujo completo de embeddings + IndexFlatIP con normalización coseno.

2. **"El Trainer de HuggingFace lanza OOM con batch_size=16 en CPU, ¿qué parámetros ajustar?"**  
   → Se ajustó a `gradient_accumulation_steps=4` y `fp16=False` para modo CPU.

3. **"Dame la estructura de callbacks de Plotly Dash para un chatbot con historial, toggle RAG y visualización de chunks recuperados."**  
   → Se adaptó el código sugerido al diseño visual del proyecto.

4. **"¿Cómo cachear embeddings en disco con numpy para no regenerarlos en cada ejecución?"**  
   → Se implementó con `np.save()` / `np.load()` + pickle para chunks.

---

## Cómo validamos las respuestas

- Cada sugerencia de código fue ejecutada localmente antes de incorporarse al proyecto.
- Se contrastaron las recomendaciones de HuggingFace con la documentación oficial de `transformers` y `datasets`.
- Los callbacks de Dash se probaron manualmente con datos de prueba antes de conectar el chatbot real.
- La estrategia de chunking se evaluó empíricamente comparando longitudes y resultados de búsqueda.

---

## Lo que NO delegamos

- La elección de la línea de investigación (Línea A: Género musical).
- La decisión de usar estrategia de chunking por estrofa vs canción completa (se probaron las dos y se analizaron los resultados).
- La interpretación de la matriz de confusión y las métricas del clasificador.
- El diseño visual final de la interfaz (paleta de colores, tipografía, layout).
- La selección de preguntas de prueba para la evaluación del chatbot.
- Las conclusiones sobre las limitaciones del sistema y posibles mejoras.

---

## Reflexión sobre el uso de IA

El uso de asistentes de IA aceleró la implementación de componentes técnicos complejos (FAISS, HuggingFace Trainer, callbacks Dash). Sin embargo, **las decisiones de diseño del sistema, la elección de hiperparámetros y la evaluación crítica de los resultados fueron responsabilidad del equipo**. La IA actuó como un asistente de codificación, no como tomador de decisiones.
