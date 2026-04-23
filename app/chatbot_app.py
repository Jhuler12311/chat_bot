"""
chatbot_app.py — Aplicación Plotly Dash del Chatbot Musical
Proyecto 3 - Chatbot Musical CUC

Ejecutar:
    python app/chatbot_app.py
    
Abrir navegador en: http://127.0.0.1:8050/
"""

import sys
import os
import json
import time
import threading
import pandas as pd
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import dash
from dash import dcc, html, Input, Output, State, ctx, callback_context, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px

from app.config import (CORPUS_CSV, TOP_K, CHUNKING_STRATEGY,
                        APP_HOST, APP_PORT, APP_DEBUG)

# ─────────────────────────────────────────────────────────────────────────────
# Estado global (se inicializa en background)
# ─────────────────────────────────────────────────────────────────────────────
chatbot = None
rag_index = None
rag_chunks = None
df_corpus = None
STATUS = {"ready": False, "message": "Iniciando sistema..."}


def load_corpus() -> pd.DataFrame:
    p = Path(CORPUS_CSV)
    if not p.exists():
        # Buscar en data/
        alts = list((ROOT / "data").glob("*.csv"))
        if alts:
            p = alts[0]
        else:
            raise FileNotFoundError(
                f"No se encontró el corpus. Coloca tu CSV en {CORPUS_CSV}")
    df = pd.read_csv(p, low_memory=False)
    # Normalizar nombres de columnas
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    print(f"[APP] Corpus cargado: {len(df)} filas, columnas: {list(df.columns)}")
    return df


def init_system():
    global chatbot, rag_index, rag_chunks, df_corpus, STATUS
    try:
        STATUS["message"] = "Cargando corpus..."
        df_corpus = load_corpus()

        STATUS["message"] = "Construyendo índice RAG (puede tardar 1-2 min la primera vez)..."
        from src.rag_utils import build_rag_pipeline
        rag_index, rag_chunks = build_rag_pipeline(df_corpus, estrategia=CHUNKING_STRATEGY)

        STATUS["message"] = "Inicializando chatbot..."
        from src.chatbot_engine import MusicChatbot
        chatbot = MusicChatbot(index=rag_index, chunks=rag_chunks)

        STATUS["ready"] = True
        STATUS["message"] = "✅ Sistema listo"
        print("[APP] Sistema completamente inicializado.")
    except Exception as e:
        STATUS["message"] = f"❌ Error: {e}"
        print(f"[APP] Error en inicialización: {e}")
        import traceback; traceback.print_exc()


# Lanzar en background
threading.Thread(target=init_system, daemon=True).start()

# ─────────────────────────────────────────────────────────────────────────────
# Gráficas auxiliares
# ─────────────────────────────────────────────────────────────────────────────

def make_genre_chart():
    if df_corpus is None:
        return go.Figure()
    col = "genre" if "genre" in df_corpus.columns else "genero"
    if col not in df_corpus.columns:
        return go.Figure()
    counts = df_corpus[col].value_counts().head(8)
    fig = px.bar(
        x=counts.values, y=counts.index, orientation="h",
        color=counts.values,
        color_continuous_scale=["#1a1a2e", "#e94560", "#f5a623"],
        labels={"x": "Canciones", "y": "Género"},
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0", family="Space Mono, monospace"),
        margin=dict(l=10, r=10, t=20, b=10),
        coloraxis_showscale=False,
        showlegend=False,
        height=220,
    )
    fig.update_traces(marker_line_width=0)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    return fig


def make_decade_chart():
    if df_corpus is None:
        return go.Figure()
    col = "release_date" if "release_date" in df_corpus.columns else "year"
    if col not in df_corpus.columns:
        return go.Figure()
    years = pd.to_numeric(df_corpus[col], errors="coerce").dropna()
    decades = (years // 10 * 10).value_counts().sort_index()
    fig = px.line(
        x=decades.index.astype(int), y=decades.values,
        markers=True,
        labels={"x": "Década", "y": "Canciones"},
        color_discrete_sequence=["#e94560"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0", family="Space Mono, monospace"),
        margin=dict(l=10, r=10, t=20, b=10),
        height=180,
    )
    fig.update_traces(line_width=2, marker_size=6)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zeroline=False)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Layout Plotly Dash
# ─────────────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Bebas+Neue&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="MúsicBot — CUC",
)
server = app.server

# ── Estilos inline ────────────────────────────────────────────────────────────
COLORS = {
    "bg":       "#0d0d1a",
    "surface":  "#141428",
    "card":     "#1a1a35",
    "accent":   "#e94560",
    "accent2":  "#f5a623",
    "text":     "#e0e0e0",
    "muted":    "#7a7a9a",
    "border":   "#2a2a4a",
}

FONT_MONO = "Space Mono, Courier New, monospace"
FONT_DISPLAY = "Bebas Neue, Impact, sans-serif"

STYLE_PAGE = {
    "backgroundColor": COLORS["bg"],
    "minHeight": "100vh",
    "fontFamily": FONT_MONO,
    "color": COLORS["text"],
    "padding": "0",
}

STYLE_HEADER = {
    "background": f"linear-gradient(135deg, {COLORS['bg']} 0%, #1a0a2e 100%)",
    "borderBottom": f"2px solid {COLORS['accent']}",
    "padding": "16px 24px",
    "display": "flex",
    "alignItems": "center",
    "gap": "16px",
}

STYLE_CHAT_AREA = {
    "backgroundColor": COLORS["surface"],
    "borderRadius": "12px",
    "border": f"1px solid {COLORS['border']}",
    "height": "500px",
    "overflowY": "auto",
    "padding": "16px",
    "marginBottom": "12px",
    "scrollbarWidth": "thin",
    "scrollbarColor": f"{COLORS['accent']} {COLORS['bg']}",
}

STYLE_INPUT = {
    "backgroundColor": COLORS["card"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "8px",
    "color": COLORS["text"],
    "fontFamily": FONT_MONO,
    "fontSize": "13px",
    "padding": "12px 16px",
    "flex": "1",
    "outline": "none",
}

STYLE_BTN_SEND = {
    "backgroundColor": COLORS["accent"],
    "border": "none",
    "borderRadius": "8px",
    "color": "white",
    "fontFamily": FONT_DISPLAY,
    "fontSize": "16px",
    "padding": "12px 24px",
    "cursor": "pointer",
    "letterSpacing": "1px",
    "whiteSpace": "nowrap",
}

STYLE_BTN_CLEAR = {
    "backgroundColor": "transparent",
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "8px",
    "color": COLORS["muted"],
    "fontFamily": FONT_MONO,
    "fontSize": "11px",
    "padding": "6px 14px",
    "cursor": "pointer",
}

STYLE_CARD = {
    "backgroundColor": COLORS["card"],
    "borderRadius": "12px",
    "border": f"1px solid {COLORS['border']}",
    "padding": "16px",
    "marginBottom": "12px",
}

STYLE_CHUNK_CARD = {
    "backgroundColor": "#1e1e3a",
    "borderRadius": "8px",
    "border": f"1px solid {COLORS['accent']}33",
    "padding": "10px 12px",
    "marginBottom": "8px",
    "fontSize": "11px",
    "color": COLORS["muted"],
    "borderLeft": f"3px solid {COLORS['accent']}",
}


def render_message(role: str, content: str) -> html.Div:
    """Renderiza un mensaje del chat."""
    is_bot = role == "assistant"
    bubble_style = {
        "backgroundColor": COLORS["card"] if is_bot else COLORS["accent"] + "22",
        "borderRadius": "12px 12px 12px 4px" if is_bot else "12px 12px 4px 12px",
        "border": f"1px solid {COLORS['border']}" if is_bot else f"1px solid {COLORS['accent']}44",
        "padding": "12px 16px",
        "maxWidth": "80%",
        "fontSize": "13px",
        "lineHeight": "1.6",
        "whiteSpace": "pre-wrap",
    }
    avatar = "🎵" if is_bot else "👤"
    return html.Div([
        html.Span(avatar, style={"fontSize": "18px", "marginTop": "4px",
                                  "flexShrink": "0"}),
        html.Div(content, style=bubble_style),
    ], style={
        "display": "flex",
        "gap": "10px",
        "marginBottom": "12px",
        "flexDirection": "row" if is_bot else "row-reverse",
        "alignItems": "flex-start",
    })


# ── Preguntas de ejemplo ──────────────────────────────────────────────────────
EXAMPLE_QUESTIONS = [
    "¿Qué canciones hablan de amor?",
    "Dame una canción de rock sobre libertad",
    "¿Qué diferencia al hip-hop del pop?",
    "Canciones tristes de los 80s",
    "¿Quién canta sobre el desamor?",
    "Tell me about jazz songs in my corpus",
]

app.layout = html.Div([
    # ── Estado del sistema ────────────────────────────────────────────────────
    dcc.Store(id="chat-store", data=[]),
    dcc.Store(id="rag-toggle", data=True),
    dcc.Interval(id="status-interval", interval=2000, n_intervals=0,
                 disabled=False),

    # ── Header ────────────────────────────────────────────────────────────────
    html.Div([
        html.Span("🎵", style={"fontSize": "32px"}),
        html.Div([
            html.H1("MúsicBot", style={
                "fontFamily": FONT_DISPLAY,
                "fontSize": "36px",
                "color": COLORS["accent"],
                "margin": "0",
                "letterSpacing": "3px",
            }),
            html.P("Chatbot Musical Inteligente · RAG + Fine-Tuning · CUC",
                   style={"margin": "0", "fontSize": "11px", "color": COLORS["muted"]}),
        ]),
        html.Div(id="status-badge", style={"marginLeft": "auto"}),
    ], style=STYLE_HEADER),

    # ── Body ──────────────────────────────────────────────────────────────────
    html.Div([
        dbc.Row([
            # ── Columna izquierda: sidebar ────────────────────────────────────
            dbc.Col([
                # Stats corpus
                html.Div([
                    html.H6("📊 CORPUS", style={
                        "fontFamily": FONT_DISPLAY,
                        "color": COLORS["accent2"],
                        "letterSpacing": "2px",
                        "marginBottom": "10px",
                        "fontSize": "14px",
                    }),
                    html.Div(id="corpus-stats"),
                    dcc.Graph(id="genre-chart", figure=go.Figure(),
                              config={"displayModeBar": False}),
                    dcc.Graph(id="decade-chart", figure=go.Figure(),
                              config={"displayModeBar": False}),
                ], style=STYLE_CARD),

                # Toggle RAG
                html.Div([
                    html.H6("⚙️ CONFIGURACIÓN", style={
                        "fontFamily": FONT_DISPLAY,
                        "color": COLORS["accent2"],
                        "letterSpacing": "2px",
                        "marginBottom": "10px",
                        "fontSize": "14px",
                    }),
                    html.Div([
                        html.Label("Usar RAG", style={"fontSize": "12px",
                                                       "color": COLORS["text"],
                                                       "marginRight": "10px"}),
                        dcc.Checklist(
                            id="rag-checkbox",
                            options=[{"label": "", "value": "rag"}],
                            value=["rag"],
                            style={"display": "inline"},
                        ),
                    ], style={"marginBottom": "8px", "display": "flex",
                              "alignItems": "center"}),
                    html.P("Activa RAG para respuestas fundamentadas en el corpus.",
                           style={"fontSize": "10px", "color": COLORS["muted"],
                                  "marginBottom": "10px"}),

                    # Filtro de género
                    html.Label("Filtrar por género:", style={"fontSize": "11px",
                                                              "color": COLORS["muted"]}),
                    dcc.Dropdown(
                        id="genre-filter",
                        options=[{"label": "Todos", "value": ""}],
                        value="",
                        clearable=False,
                        style={
                            "backgroundColor": COLORS["surface"],
                            "color": COLORS["text"],
                            "border": f"1px solid {COLORS['border']}",
                            "borderRadius": "6px",
                            "fontSize": "12px",
                        },
                    ),
                ], style=STYLE_CARD),

                # Preguntas de ejemplo
                html.Div([
                    html.H6("💡 EJEMPLOS", style={
                        "fontFamily": FONT_DISPLAY,
                        "color": COLORS["accent2"],
                        "letterSpacing": "2px",
                        "marginBottom": "10px",
                        "fontSize": "14px",
                    }),
                    *[html.Button(q, id=f"example-{i}", n_clicks=0, style={
                        "backgroundColor": "transparent",
                        "border": f"1px solid {COLORS['border']}",
                        "borderRadius": "6px",
                        "color": COLORS["muted"],
                        "fontFamily": FONT_MONO,
                        "fontSize": "10px",
                        "padding": "6px 10px",
                        "cursor": "pointer",
                        "display": "block",
                        "width": "100%",
                        "textAlign": "left",
                        "marginBottom": "6px",
                        "transition": "all 0.2s",
                    }) for i, q in enumerate(EXAMPLE_QUESTIONS)],
                ], style=STYLE_CARD),

            ], width=3, style={"padding": "16px 8px 16px 16px"}),

            # ── Columna central: chat ─────────────────────────────────────────
            dbc.Col([
                # Área de mensajes
                html.Div(id="chat-messages",
                         children=[render_message("assistant",
                            "¡Hola! Soy MúsicBot 🎵\n"
                            "Soy un experto en letras de canciones de tu corpus.\n"
                            "Puedes preguntarme sobre artistas, géneros, letras o temáticas.\n"
                            "¿Qué quieres explorar hoy?")],
                         style=STYLE_CHAT_AREA),

                # Input row
                html.Div([
                    dcc.Input(
                        id="user-input",
                        type="text",
                        placeholder="Pregúntame sobre canciones, géneros o artistas...",
                        n_submit=0,
                        debounce=False,
                        style=STYLE_INPUT,
                    ),
                    html.Button("ENVIAR ▶", id="send-btn", n_clicks=0,
                                style=STYLE_BTN_SEND),
                    html.Button("🗑 Limpiar", id="clear-btn", n_clicks=0,
                                style=STYLE_BTN_CLEAR),
                ], style={"display": "flex", "gap": "8px", "alignItems": "center"}),

            ], width=6, style={"padding": "16px 8px"}),

            # ── Columna derecha: chunks recuperados ───────────────────────────
            dbc.Col([
                html.Div([
                    html.H6("🔍 CHUNKS RECUPERADOS", style={
                        "fontFamily": FONT_DISPLAY,
                        "color": COLORS["accent2"],
                        "letterSpacing": "2px",
                        "marginBottom": "10px",
                        "fontSize": "14px",
                    }),
                    html.Div(id="retrieved-chunks",
                             children=[html.P("Los fragmentos recuperados por RAG aparecerán aquí.",
                                              style={"color": COLORS["muted"],
                                                     "fontSize": "11px"})]),
                ], style=STYLE_CARD),

                # Info del sistema
                html.Div([
                    html.H6("ℹ️ SISTEMA", style={
                        "fontFamily": FONT_DISPLAY,
                        "color": COLORS["accent2"],
                        "letterSpacing": "2px",
                        "marginBottom": "10px",
                        "fontSize": "14px",
                    }),
                    html.Div(id="system-info"),
                ], style=STYLE_CARD),

            ], width=3, style={"padding": "16px 16px 16px 8px"}),
        ]),
    ], style={"maxWidth": "1600px", "margin": "0 auto"}),

], style=STYLE_PAGE)


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    [Output("status-badge", "children"),
     Output("corpus-stats", "children"),
     Output("genre-chart", "figure"),
     Output("decade-chart", "figure"),
     Output("genre-filter", "options"),
     Output("system-info", "children"),
     Output("status-interval", "disabled")],
    Input("status-interval", "n_intervals"),
)
def update_status(n):
    ready = STATUS["ready"]
    msg = STATUS["message"]

    badge_color = COLORS["accent2"] if ready else COLORS["muted"]
    badge = html.Span(msg, style={
        "backgroundColor": badge_color + "22",
        "border": f"1px solid {badge_color}",
        "borderRadius": "20px",
        "padding": "4px 12px",
        "fontSize": "11px",
        "color": badge_color,
        "fontFamily": FONT_MONO,
    })

    if not ready:
        return badge, no_update, no_update, no_update, no_update, no_update, False

    # Corpus stats
    n_songs = len(df_corpus) if df_corpus is not None else 0
    col_g = "genre" if "genre" in (df_corpus.columns if df_corpus is not None else []) else "genero"
    n_genres = df_corpus[col_g].nunique() if df_corpus is not None and col_g in df_corpus.columns else 0
    n_chunks = len(rag_chunks) if rag_chunks else 0

    stats = html.Div([
        html.Div([
            html.Span(f"{n_songs:,}", style={"color": COLORS["accent"], "fontSize": "20px",
                                              "fontFamily": FONT_DISPLAY}),
            html.Span(" canciones", style={"fontSize": "10px", "color": COLORS["muted"]}),
        ]),
        html.Div([
            html.Span(f"{n_genres}", style={"color": COLORS["accent2"], "fontSize": "20px",
                                             "fontFamily": FONT_DISPLAY}),
            html.Span(" géneros", style={"fontSize": "10px", "color": COLORS["muted"]}),
        ]),
        html.Div([
            html.Span(f"{n_chunks:,}", style={"color": "#7bc8f6", "fontSize": "20px",
                                               "fontFamily": FONT_DISPLAY}),
            html.Span(" chunks RAG", style={"fontSize": "10px", "color": COLORS["muted"]}),
        ]),
    ], style={"marginBottom": "12px", "display": "flex", "gap": "16px",
              "flexWrap": "wrap"})

    genre_fig = make_genre_chart()
    decade_fig = make_decade_chart()

    # Genre filter options
    genre_opts = [{"label": "Todos", "value": ""}]
    if df_corpus is not None and col_g in df_corpus.columns:
        for g in sorted(df_corpus[col_g].dropna().unique()):
            genre_opts.append({"label": g.title(), "value": g})

    # System info
    from src.finetuning_utils import classifier_available
    clf_status = "✅ Disponible" if classifier_available() else "⚠️ No entrenado"
    api_mode = chatbot._api_mode if chatbot else "?"
    sys_info = html.Div([
        html.Div([html.Span("Generador: ", style={"color": COLORS["muted"]}),
                  html.Span(api_mode, style={"color": COLORS["accent2"]})],
                 style={"fontSize": "11px", "marginBottom": "4px"}),
        html.Div([html.Span("Clasificador: ", style={"color": COLORS["muted"]}),
                  html.Span(clf_status, style={"color": COLORS["text"]})],
                 style={"fontSize": "11px", "marginBottom": "4px"}),
        html.Div([html.Span("Chunking: ", style={"color": COLORS["muted"]}),
                  html.Span(CHUNKING_STRATEGY, style={"color": COLORS["text"]})],
                 style={"fontSize": "11px"}),
    ])

    return badge, stats, genre_fig, decade_fig, genre_opts, sys_info, True   # disable interval


@app.callback(
    [Output("chat-messages", "children"),
     Output("chat-store", "data"),
     Output("user-input", "value"),
     Output("retrieved-chunks", "children")],
    [Input("send-btn", "n_clicks"),
     Input("user-input", "n_submit"),
     Input("clear-btn", "n_clicks"),
     *[Input(f"example-{i}", "n_clicks") for i in range(len(EXAMPLE_QUESTIONS))]],
    [State("user-input", "value"),
     State("chat-store", "data"),
     State("rag-checkbox", "value"),
     State("genre-filter", "value")],
    prevent_initial_call=True,
)
def handle_chat(send_clicks, input_submit, clear_clicks,
                *args):
    # Unpack examples + states
    example_clicks = args[:len(EXAMPLE_QUESTIONS)]
    user_text, history, rag_val, genre_filter = args[len(EXAMPLE_QUESTIONS):]

    triggered = [c["prop_id"] for c in callback_context.triggered]

    # Clear
    if any("clear-btn" in t for t in triggered):
        if chatbot:
            chatbot.reset_history()
        welcome = render_message("assistant",
            "¡Historial limpiado! 🎵 ¿Sobre qué quieres preguntar ahora?")
        return [welcome], [], "", [html.P("Esperando pregunta...",
                                           style={"color": COLORS["muted"],
                                                  "fontSize": "11px"})]

    # Example button clicked
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        if any(f"example-{i}.n_clicks" in t for t in triggered):
            user_text = q
            break

    if not user_text or not user_text.strip():
        return no_update, no_update, no_update, no_update

    if not STATUS["ready"] or chatbot is None:
        msgs = list(history or []) + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": "⏳ El sistema aún se está inicializando. Por favor espera un momento."},
        ]
        return _render_history(msgs), msgs, "", no_update

    use_rag = bool(rag_val)

    # Usar género del dropdown si está seleccionado,
    # si no, usar el clasificador fine-tuneado automáticamente
    genre_f = genre_filter if genre_filter else None
    if not genre_f:
        try:
            from src.finetuning_utils import predict_genre, classifier_available
            if classifier_available():
                from src.chatbot_engine import detect_intent
                intent_check = detect_intent(user_text)
                if intent_check != "general":
                    genre_f = predict_genre(user_text)
        except Exception:
            pass

    try:
        response, retrieved = chatbot.chat(user_text, use_rag=use_rag,
                                           genero_filter=genre_f)
    except Exception as e:
        response = f"Error: {e}"
        retrieved = []

    # Update history
    new_history = list(history or [])
    new_history.append({"role": "user", "content": user_text})
    new_history.append({"role": "assistant", "content": response})

    # Render chunks
    chunks_ui = []
    if retrieved:
        for c in retrieved:
            score_pct = int(c.get("score", 0) * 100)
            chunks_ui.append(html.Div([
                html.Div([
                    html.Span(c["titulo"], style={"color": COLORS["text"],
                                                   "fontWeight": "bold"}),
                    html.Span(f" — {c['artista']}",
                               style={"color": COLORS["accent"]}),
                ], style={"fontSize": "11px", "marginBottom": "4px"}),
                html.Div(f"🎸 {c['genero']} · 📅 {c['año']} · 🎯 {score_pct}%",
                         style={"fontSize": "10px", "color": COLORS["muted"],
                                "marginBottom": "4px"}),
                html.Div(c["texto"][:180] + "...",
                         style={"fontSize": "10px", "color": COLORS["muted"],
                                "fontStyle": "italic", "lineHeight": "1.4"}),
            ], style=STYLE_CHUNK_CARD))
    else:
        chunks_ui = [html.P("No se recuperaron chunks relevantes.",
                             style={"color": COLORS["muted"], "fontSize": "11px"})]

    return _render_history(new_history), new_history, "", chunks_ui


def _render_history(history: list) -> list:
    msgs = []
    for turn in history:
        msgs.append(render_message(turn["role"], turn["content"]))
    return msgs


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  🎵  MúsicBot — Chatbot Musical CUC  🎵")
    print("═" * 60)
    print(f"  URL: http://{APP_HOST}:{APP_PORT}/")
    print(f"  Corpus: {CORPUS_CSV}")
    print(f"  Chunking: {CHUNKING_STRATEGY}")
    print("  (El índice RAG se construye en background)")
    print("═" * 60 + "\n")
    app.run(host=APP_HOST, port=APP_PORT, debug=APP_DEBUG)
