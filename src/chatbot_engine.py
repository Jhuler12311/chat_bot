"""
chatbot_engine.py — Motor del chatbot con Ollama (Mistral local)
Proyecto 3 - Chatbot Musical CUC
"""

import os
import re
import requests
from typing import List, Dict, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# PERSONALIDAD
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Eres MusicBot, un critico musical apasionado y experto en letras de canciones.

Tu personalidad:
- Hablas con entusiasmo genuino sobre musica
- Siempre citas fragmentos REALES de letras cuando las tienes disponibles en el contexto
- Eres especifico: mencionas el artista, la cancion y el ano
- Si el usuario pregunta que dice una cancion, reproduces versos reales de la letra del contexto
- Si no encuentras info exacta, lo dices honestamente: "No tengo esa cancion en mi corpus"
- NUNCA inventas letras. Solo usas lo que aparece en el CONTEXTO que se te da
- Puedes comparar estilos, generos, epocas con criterio
- Si el usuario dice "hola" o saluda, te presentas brevemente y preguntas que quiere explorar
- Respondes en espanol siempre, a menos que el usuario escriba en ingles

Tu corpus: ~28,000 canciones de Pop, Rock, Hip-Hop, Country, Jazz, Blues, Reggae, 1950-2019."""

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "mistral"

MAX_HISTORY = 10

# ─────────────────────────────────────────────────────────────────────────────
# Detección de intención
# ─────────────────────────────────────────────────────────────────────────────

INTENT_PATTERNS = {
    "letra": [
        r"\bqu[eé]\s+(dice|habla|trata|significa)\b",
        r"\bletra\b", r"\blyrics?\b", r"\bverso\b", r"\bestrofa\b",
        r"\bcanta\b", r"\bde qu[eé] trata\b", r"\bwhat.*about\b",
        r"\bwhat does .+ say\b",
    ],
    "artista": [
        r"\bartista\b", r"\bcantante\b", r"\bsinger\b", r"\bband\b",
        r"\bgrupo\b", r"\bquién\s+(es|canta|hizo)\b",
        r"\bháblame\s+de\b", r"\btell me about\b",
        r"\bsobre\s+[A-Z]",   # "sobre The Beatles"
        r"\bhistoria\s+de\b",
    ],
    "recomendacion": [
        r"\brecomien\w+\b", r"\bsugier\w+\b", r"\bdam[eé]\b",
        r"\bquiero\s+(oir|escuchar)\b", r"\bparecida\b", r"\bsimilar\b",
        r"\bsuggest\b", r"\bgive me\b", r"\bqué\s+cancion",
        r"\balgo\s+(triste|alegre|romántico|para)\b",
    ],
    "genero": [
        r"\bgénero\b", r"\bdiferencia\b", r"\bestilo\b",
        r"\brock\b", r"\bpop\b", r"\bhip.?hop\b", r"\bjazz\b",
        r"\bblues\b", r"\bcountry\b", r"\breggae\b", r"\bmetal\b",
        r"\bcompara\b", r"\bcompare\b",
    ],
    "epoca": [
        r"\b(19[5-9]\d|20[01]\d)s?\b", r"\bdécada\b", r"\baños\s+\d",
        r"\bépoca\b", r"\bevoluci[oó]n\b", r"\bclásico\b",
    ],
}


def detect_intent(query: str) -> str:
    q = query.lower()
    scores = {intent: 0 for intent in INTENT_PATTERNS}
    for intent, patterns in INTENT_PATTERNS.items():
        for p in patterns:
            if re.search(p, q):
                scores[intent] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


# ─────────────────────────────────────────────────────────────────────────────
# Construcción del prompt
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(query: str, chunks: List[Dict], history: List[Dict], intent: str) -> str:
    """Construye el prompt de usuario con contexto real del corpus."""

    context_lines = []
    for i, c in enumerate(chunks[:5], 1):
        titulo  = c.get("titulo",  "?")
        artista = c.get("artista", "?")
        genero  = c.get("genero",  "?")
        año     = c.get("año",     "?")
        texto   = c.get("texto",   "")

        if intent in ("letra", "artista"):
            # Letra completa para preguntas de contenido
            context_lines.append(
                f'[{i}] "{titulo}" — {artista} ({genero}, {año})\n'
                f'LETRA:\n"""\n{texto[:900]}\n"""'
            )
        else:
            context_lines.append(
                f'[{i}] "{titulo}" — {artista} | {genero} | {año}\n'
                f'{texto[:300]}'
            )

    context_block = "\n\n".join(context_lines) if context_lines else \
        "No se encontraron canciones en el corpus para esta consulta."

    intent_instructions = {
        "letra":        "El usuario pregunta sobre la LETRA de una canción. Cita versos reales del contexto. Si no tienes esa canción exacta, dilo.",
        "artista":      "El usuario pregunta sobre un ARTISTA. Describe su estilo con canciones concretas del contexto. Cita algún verso.",
        "recomendacion":"El usuario pide RECOMENDACIONES. Sugiere canciones del contexto con artista y año. Explica brevemente por qué.",
        "genero":       "El usuario pregunta sobre GÉNEROS. Compara con ejemplos concretos del contexto.",
        "epoca":        "El usuario pregunta sobre una ÉPOCA. Usa las canciones del contexto para ilustrar tendencias.",
        "general":      "Responde con personalidad musical. Si es un saludo, preséntate y pregunta qué quiere explorar.",
    }
    instruction = intent_instructions.get(intent, intent_instructions["general"])

    prompt = (
        f"INSTRUCCIÓN: {instruction}\n\n"
        f"CANCIONES DEL CORPUS (usa estos datos reales en tu respuesta):\n"
        f"{context_block}\n\n"
        f"PREGUNTA DEL USUARIO: {query}\n\n"
        f"Responde como MúsicBot con personalidad. Cita letras reales si las tienes. "
        f"Sé específico con artistas y títulos del contexto."
    )
    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# Clase principal
# ─────────────────────────────────────────────────────────────────────────────

class MusicChatbot:
    def __init__(self, index=None, chunks=None):
        self.index  = index
        self.chunks = chunks
        self.history: List[Dict] = []
        self._api_mode = self._detect_api()
        print(f"[BOT] Modo generador: {self._api_mode}")

    # ── Detección ─────────────────────────────────────────────────────────────

    def _detect_api(self) -> str:
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        # Ollama local
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=3)
            if r.status_code == 200:
                print("[BOT] Ollama detectado ")
                return "ollama"
        except Exception:
            pass
        return "rules"

    # ── Generadores ───────────────────────────────────────────────────────────

    def _generate_ollama(self, system: str, user_prompt: str) -> str:
        """Llama a Ollama (Mistral) corriendo localmente."""
        messages = [{"role": "system", "content": system}]

        # Solo los 2 turnos mas recientes para reducir tokens
        for turn in self.history[-2:]:
            messages.append({
                "role": turn["role"],
                "content": turn["content"][:200],
            })

        # Recortar el prompt para no saturar el contexto
        messages.append({"role": "user", "content": user_prompt[:1500]})

        payload = {
            "model":  OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 300,
                "num_ctx": 2048,
            },
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    def _generate_anthropic(self, system: str, user_prompt: str) -> str:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=800,
            system=system,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return msg.content[0].text

    def _generate_openai(self, system: str, user_prompt: str) -> str:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=800,
            temperature=0.7,
        )
        return resp.choices[0].message.content

    def _generate_rules(self, chunks: List[Dict], query: str, intent: str) -> str:
        """Fallback sin ningún modelo: plantillas + datos del corpus."""
        q = query.strip().lower()

        # Saludo
        if len(query.strip()) < 15 or q in ("hola", "hi", "hello", "buenas", "hey"):
            return (
                "¡Hola! :) Soy **MúsicBot**, tu experto en letras de canciones.\n\n"
                "Tengo en mi corpus más de 28,000 canciones de Pop, Rock, Hip-Hop, "
                "Country, Jazz, Blues y Reggae (1950–2019).\n\n"
                "Puedo ayudarte con:\n"
                "• La letra de una canción específica\n"
                "• Info sobre un artista\n"
                "• Recomendaciones por mood o género\n"
                "• Comparar estilos o épocas\n\n"
                "¿Qué quieres explorar? "
            )

        if not chunks:
            return (
                ":( No encontré canciones relacionadas con tu búsqueda.\n"
                "Prueba con el nombre de un artista, canción o género específico."
            )

        if intent == "letra":
            c = chunks[0]
            letra = c.get("texto", "")
            return (
                f" **\"{c['titulo']}\"** — {c['artista']} ({c['genero']}, {c['año']})\n\n"
                f"Fragmento de la letra:\n\n"
                f"{letra[:700]}\n\n"
                f"{'...(letra continúa en el corpus)' if len(letra) > 700 else ''}"
            )

        elif intent == "artista":
            artista = chunks[0].get("artista", "?")
            canciones = []
            seen = set()
            for c in chunks:
                if c["titulo"] not in seen:
                    seen.add(c["titulo"])
                    canciones.append(c)
            lines = [f" **{artista}** en el corpus:\n"]
            for c in canciones[:4]:
                lines.append(f"• **\"{c['titulo']}\"** ({c['genero']}, {c['año']})")
                if c.get("texto"):
                    lines.append(f"  _{c['texto'][:150].strip()}..._")
            return "\n".join(lines)

        elif intent == "recomendacion":
            lines = [" Te recomiendo estas canciones del corpus:\n"]
            seen = set()
            for c in chunks[:4]:
                if c["titulo"] not in seen:
                    seen.add(c["titulo"])
                    lines.append(
                        f"🎵 **\"{c['titulo']}\"** — {c['artista']} ({c['genero']}, {c['año']})\n"
                        f"   _{c.get('texto','')[:130].strip()}..._\n"
                    )
            return "\n".join(lines)

        else:
            lines = [" .-. Encontré estas canciones relacionadas:\n"]
            seen = set()
            for c in chunks[:4]:
                if c["titulo"] not in seen:
                    seen.add(c["titulo"])
                    lines.append(f"• **\"{c['titulo']}\"** — {c['artista']} ({c['genero']}, {c['año']})")
            return "\n".join(lines)

    def generate(self, prompt: str, chunks: List[Dict] = None,
                 query: str = "", intent: str = "general") -> str:
        try:
            if self._api_mode == "anthropic":
                return self._generate_anthropic(SYSTEM_PROMPT, prompt)
            elif self._api_mode == "openai":
                return self._generate_openai(SYSTEM_PROMPT, prompt)
            elif self._api_mode == "ollama":
                return self._generate_ollama(SYSTEM_PROMPT, prompt)
            else:
                return self._generate_rules(chunks or [], query, intent)
        except Exception as e:
            # Si Ollama falla, caer a rules
            print(f"[BOT] Error en generador ({self._api_mode}): {e}")
            return self._generate_rules(chunks or [], query, intent)

    # ── Memoria ───────────────────────────────────────────────────────────────

    def add_to_history(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > MAX_HISTORY:
            self.history = self.history[-MAX_HISTORY:]

    def reset_history(self):
        self.history = []

    def set_index(self, index, chunks):
        self.index  = index
        self.chunks = chunks

    # ── Chat principal ────────────────────────────────────────────────────────

    def chat(self, query: str, use_rag: bool = True,
             genero_filter: Optional[str] = None) -> Tuple[str, List[Dict]]:

        retrieved = []
        intent = detect_intent(query)

        if use_rag and self.index is not None and self.chunks is not None:
            from src.rag_utils import search
            top_k = 6 if intent in ("letra", "artista") else 4
            retrieved = search(query, self.index, self.chunks,
                               top_k=top_k, genero_filter=genero_filter)

        detected_genre = None
        try:
            from src.finetuning_utils import predict_genre, classifier_available
            if classifier_available():
                detected_genre = predict_genre(query)
        except Exception:
            pass

        prompt   = build_prompt(query, retrieved, self.history, intent)
        response = self.generate(prompt, chunks=retrieved, query=query, intent=intent)

        # Fuentes
        if retrieved and intent != "general":
            seen, unique = set(), []
            for c in retrieved:
                key = f"{c['titulo']}|{c['artista']}"
                if key not in seen:
                    seen.add(key)
                    unique.append(f'"{c["titulo"]}" — {c["artista"]}')
            if unique:
                response += f"\n\n *Fuentes: {' · '.join(unique[:3])}*"

        if detected_genre and intent != "general":
            response += f"\n  *Género detectado: {detected_genre}*"

        self.add_to_history("user",      query)
        self.add_to_history("assistant", response)

        return response, retrieved

    def chat_no_rag(self, query: str) -> str:
        intent   = detect_intent(query)
        prompt   = build_prompt(query, [], self.history, intent)
        response = self.generate(prompt, chunks=[], query=query, intent=intent)
        self.add_to_history("user",      query)
        self.add_to_history("assistant", response)
        return response