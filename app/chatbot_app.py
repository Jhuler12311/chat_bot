"""
chatbot_app.py — Aplicación Plotly Dash del Chatbot Musical
Proyecto 3 - Chatbot Musical CUC

Ejecutar:
    python app/chatbot_app.py

Abrir navegador: http://127.0.0.1:8050/
"""

import sys
import os
import threading
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px

from app.config import CORPUS_CSV, CHUNKING_STRATEGY, APP_HOST, APP_PORT, APP_DEBUG

# ─────────────────────────────────────────────────────────────────────────────
# Estado global
# ─────────────────────────────────────────────────────────────────────────────
chatbot = None
rag_index = None
rag_chunks = None
df_corpus = None
STATUS = {"ready": False, "message": "Iniciando sistema...", "error": False}


def load_corpus() -> pd.DataFrame:
    p = Path(CORPUS_CSV)
    if not p.exists():
        alts = list((ROOT / "data").glob("*.csv"))
        if alts:
            p = alts[0]
            print(f"[APP] Corpus encontrado: {p}")
        else:
            raise FileNotFoundError(
                f"Coloca tu CSV en: {ROOT / 'data'}\n"
                "Debe ser tcc_ceds_music.csv u otro con columna 'lyrics'."
            )
    df = pd.read_csv(p, low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    print(f"[APP] Corpus: {len(df)} filas | columnas: {list(df.columns)[:8]}...")
    return df


def init_system():
    global chatbot, rag_index, rag_chunks, df_corpus, STATUS
    try:
        STATUS["message"] = "Cargando corpus..."
        df_corpus = load_corpus()

        STATUS["message"] = "Construyendo índice RAG..."
        from src.rag_utils import build_rag_pipeline
        rag_index, rag_chunks = build_rag_pipeline(df_corpus, estrategia=CHUNKING_STRATEGY)

        STATUS["message"] = "Inicializando chatbot..."
        from src.chatbot_engine import MusicChatbot
        chatbot = MusicChatbot(index=rag_index, chunks=rag_chunks)

        STATUS["ready"] = True
        STATUS["message"] = "Sistema listo"
    except Exception as e:
        STATUS["message"] = f"{e}"
        STATUS["error"] = True
        import traceback; traceback.print_exc()


threading.Thread(target=init_system, daemon=True).start()

# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────

def make_genre_chart():
    if df_corpus is None:
        return go.Figure().update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                          plot_bgcolor="rgba(0,0,0,0)")
    col = next((c for c in ["genre", "genero"] if c in df_corpus.columns), None)
    if not col:
        return go.Figure()
    counts = df_corpus[col].value_counts().head(8)
    fig = px.bar(x=counts.values, y=counts.index, orientation="h",
                 color_discrete_sequence=["#c8a96e"])
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#888", family="'DM Mono', monospace", size=10),
        margin=dict(l=0, r=0, t=4, b=0),
        coloraxis_showscale=False,
        height=180,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)", zeroline=False, showticklabels=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.03)", tickfont=dict(size=10))
    fig.update_traces(marker_opacity=0.75)
    return fig


def make_decade_chart():
    if df_corpus is None:
        return go.Figure()
    col = next((c for c in ["release_date", "year", "año"] if c in df_corpus.columns), None)
    if not col:
        return go.Figure()
    years = pd.to_numeric(df_corpus[col], errors="coerce").dropna()
    decades = (years // 10 * 10).value_counts().sort_index()
    fig = px.area(x=decades.index.astype(int), y=decades.values,
                  color_discrete_sequence=["#c8a96e"])
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#888", family="'DM Mono', monospace", size=10),
        margin=dict(l=0, r=0, t=4, b=0),
        height=120,
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.03)", zeroline=False, title=None)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.03)", zeroline=False, title=None, showticklabels=False)
    fig.update_traces(fillcolor="rgba(200,169,110,0.08)", line_width=1.5)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Design tokens
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "bg":      "#0e0e0e",
    "surface": "#141414",
    "card":    "#181818",
    "border":  "#272727",
    "gold":    "#c8a96e",
    "text":    "#e0e0e0",
    "muted":   "#555",
    "dimmed":  "#333",
    "green":   "#5a9e6f",
    "red":     "#9e5a5a",
}
MONO  = "'DM Mono', 'Courier New', monospace"
SERIF = "'Cormorant Garamond', Georgia, serif"

INTENT_COLORS = {
    "letra":         "#8ab4c8",
    "artista":       C["gold"],
    "recomendacion": C["green"],
    "genero":        "#a08ab4",
    "epoca":         "#c88a6e",
    "general":       C["muted"],
}
INTENT_LABELS = {
    "letra":         "letra",
    "artista":       "artista",
    "recomendacion": "recomendación",
    "genero":        "género",
    "epoca":         "época",
    "general":       "general",
}

EXAMPLE_QUESTIONS = [
    "¿Qué dice la letra de 'Hotel California'?",
    "Háblame sobre The Beatles",
    "¿Qué diferencia al blues del jazz?",
    "Recomiéndame canciones tristes de los 80s",
    "¿Qué canciones hablan de amor y pérdida?",
    "Tell me about hip-hop songs about the streets",
]

# ─────────────────────────────────────────────────────────────────────────────
# Render de mensajes
# ─────────────────────────────────────────────────────────────────────────────

def render_message(role: str, content: str, intent: str = "general") -> html.Div:
    is_bot = role == "assistant"

    if is_bot and intent != "general":
        badge_color = INTENT_COLORS.get(intent, C["muted"])
        badge = html.Span(INTENT_LABELS.get(intent, intent), style={
            "color": badge_color,
            "fontSize": "9px",
            "letterSpacing": "0.12em",
            "textTransform": "uppercase",
            "marginBottom": "5px",
            "display": "block",
            "fontFamily": MONO,
        })
    else:
        badge = None

    bubble_style = {
        "padding": "12px 16px",
        "maxWidth": "82%",
        "fontSize": "13px",
        "lineHeight": "1.75",
        "whiteSpace": "pre-wrap",
        "color": C["text"] if is_bot else "#bbb",
        "fontFamily": MONO,
        "borderRadius": "2px",
        "backgroundColor": C["card"] if is_bot else "transparent",
        "border": f"1px solid {C['border']}" if is_bot else "none",
        "borderLeft": f"2px solid {C['gold']}" if is_bot else f"2px solid {C['dimmed']}",
    }

    return html.Div([
        html.Div([
            badge,
            html.Div(content, style=bubble_style),
        ], style={
            "maxWidth": "82%",
        }),
    ], style={
        "display": "flex",
        "marginBottom": "18px",
        "justifyContent": "flex-start" if is_bot else "flex-end",
    })


# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400&family=Cormorant+Garamond:wght@300;400;600&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="MúsicBot",
)
server = app.server

# ── Sidebar section helper ────────────────────────────────────────────────────
def sidebar_section(title, children, style_extra=None):
    s = {
        "marginBottom": "1px",
        "borderBottom": f"1px solid {C['border']}",
        "padding": "18px 20px",
    }
    if style_extra:
        s.update(style_extra)
    return html.Div([
        html.Div(title, style={
            "fontSize": "9px",
            "letterSpacing": "0.18em",
            "textTransform": "uppercase",
            "color": C["muted"],
            "marginBottom": "14px",
            "fontFamily": MONO,
        }),
        *children,
    ], style=s)


app.layout = html.Div([
    dcc.Store(id="chat-store", data=[]),
    dcc.Store(id="intent-store", data="general"),
    dcc.Interval(id="status-interval", interval=2000, n_intervals=0),

    # ── LAYOUT PRINCIPAL: sidebar izq + chat + sidebar der ────────────────────
    html.Div([

        # ── SIDEBAR IZQUIERDA ─────────────────────────────────────────────────
        html.Div([

            # Logo / título
            html.Div([
                html.Div("músicbot", style={
                    "fontFamily": SERIF,
                    "fontWeight": "300",
                    "fontSize": "22px",
                    "color": C["text"],
                    "letterSpacing": "0.05em",
                }),
                html.Div("RAG · Fine-tuning · CUC 2025", style={
                    "fontSize": "9px",
                    "color": C["muted"],
                    "fontFamily": MONO,
                    "letterSpacing": "0.1em",
                    "marginTop": "2px",
                }),
            ], style={"padding": "24px 20px 20px", "borderBottom": f"1px solid {C['border']}"}),

            # Estado del sistema
            sidebar_section("estado", [
                html.Div(id="header-status"),
            ]),

            # API Key
            sidebar_section("generador", [
                dcc.Input(
                    id="api-key-input",
                    type="password",
                    placeholder="sk-ant-... o sk-...",
                    style={
                        "backgroundColor": "transparent",
                        "border": "none",
                        "borderBottom": f"1px solid {C['border']}",
                        "borderRadius": "0",
                        "color": C["text"],
                        "fontFamily": MONO,
                        "fontSize": "11px",
                        "padding": "6px 0",
                        "width": "100%",
                        "marginBottom": "10px",
                        "outline": "none",
                    }
                ),
                html.Button("aplicar", id="apply-api-btn", n_clicks=0, style={
                    "backgroundColor": "transparent",
                    "border": f"1px solid {C['border']}",
                    "borderRadius": "2px",
                    "color": C["muted"],
                    "fontFamily": MONO,
                    "fontSize": "10px",
                    "letterSpacing": "0.1em",
                    "padding": "5px 14px",
                    "cursor": "pointer",
                    "transition": "all 0.2s",
                }),
                html.Div(id="api-status", style={
                    "fontSize": "10px",
                    "marginTop": "8px",
                    "color": C["muted"],
                    "fontFamily": MONO,
                }),
            ]),

            # Corpus stats
            sidebar_section("corpus", [
                html.Div(id="corpus-stats", style={"marginBottom": "14px"}),
                dcc.Graph(id="genre-chart", figure=go.Figure(),
                          config={"displayModeBar": False},
                          style={"marginLeft": "-8px"}),
                dcc.Graph(id="decade-chart", figure=go.Figure(),
                          config={"displayModeBar": False},
                          style={"marginLeft": "-8px", "marginTop": "8px"}),
            ]),

            # Filtro género
            sidebar_section("filtrar", [
                html.Label("Género", style={"fontSize": "10px", "color": C["muted"],
                                             "fontFamily": MONO, "marginBottom": "6px",
                                             "display": "block"}),
                dcc.Dropdown(
                    id="genre-filter",
                    options=[{"label": "Todos", "value": ""}],
                    value="",
                    clearable=False,
                    style={"fontSize": "11px"},
                ),
            ], style_extra={"borderBottom": "none"}),

        ], style={
            "width": "230px",
            "flexShrink": "0",
            "backgroundColor": C["surface"],
            "borderRight": f"1px solid {C['border']}",
            "overflowY": "auto",
            "height": "100vh",
            "position": "sticky",
            "top": "0",
        }),

        # ── CHAT CENTRAL ──────────────────────────────────────────────────────
        html.Div([

            # Mensajes
            html.Div(
                id="chat-messages",
                children=[render_message("assistant",
                    "Hola. Soy MúsicBot.\n\n"
                    "Puedo ayudarte con letras, artistas, géneros musicales y recomendaciones.\n"
                    "El sistema usa RAG para buscar en el corpus real de canciones.\n\n"
                    "¿Qué quieres explorar?"
                )],
                style={
                    "flex": "1",
                    "overflowY": "auto",
                    "padding": "32px 40px",
                }
            ),

            # Sugerencias
            html.Div([
                html.Span(q, id=f"eg-{i}", n_clicks=0, style={
                    "border": f"1px solid {C['dimmed']}",
                    "borderRadius": "2px",
                    "padding": "4px 10px",
                    "fontSize": "10px",
                    "color": C["muted"],
                    "cursor": "pointer",
                    "display": "inline-block",
                    "marginRight": "6px",
                    "marginBottom": "6px",
                    "fontFamily": MONO,
                    "letterSpacing": "0.02em",
                    "transition": "all 0.15s",
                }) for i, q in enumerate(EXAMPLE_QUESTIONS)
            ], style={
                "padding": "0 40px 12px",
                "lineHeight": "2.2",
            }),

            # Input row
            html.Div([
                dcc.Input(
                    id="user-input",
                    type="text",
                    n_submit=0,
                    placeholder="Escribe tu pregunta...",
                    style={
                        "backgroundColor": "transparent",
                        "border": "none",
                        "borderTop": f"1px solid {C['border']}",
                        "color": C["text"],
                        "fontFamily": MONO,
                        "fontSize": "13px",
                        "padding": "18px 20px",
                        "flex": "1",
                        "outline": "none",
                    }
                ),
                html.Button("↑", id="send-btn", n_clicks=0, style={
                    "backgroundColor": C["gold"],
                    "border": "none",
                    "borderTop": f"1px solid {C['border']}",
                    "color": C["bg"],
                    "fontFamily": MONO,
                    "fontSize": "18px",
                    "fontWeight": "bold",
                    "padding": "0 24px",
                    "cursor": "pointer",
                    "transition": "opacity 0.2s",
                }),
                html.Button("✕", id="clear-btn", n_clicks=0, style={
                    "backgroundColor": "transparent",
                    "border": "none",
                    "borderTop": f"1px solid {C['border']}",
                    "borderLeft": f"1px solid {C['border']}",
                    "color": C["muted"],
                    "fontFamily": MONO,
                    "fontSize": "13px",
                    "padding": "0 18px",
                    "cursor": "pointer",
                }),
            ], style={
                "display": "flex",
                "borderTop": f"1px solid {C['border']}",
            }),

        ], style={
            "flex": "1",
            "display": "flex",
            "flexDirection": "column",
            "height": "100vh",
            "backgroundColor": C["bg"],
        }),

        # ── PANEL DERECHO: RAG chunks + info ─────────────────────────────────
        html.Div([

            # Título panel
            html.Div("fuentes recuperadas", style={
                "fontSize": "9px",
                "letterSpacing": "0.18em",
                "textTransform": "uppercase",
                "color": C["muted"],
                "fontFamily": MONO,
                "padding": "24px 20px 14px",
                "borderBottom": f"1px solid {C['border']}",
            }),

            # Chunks
            html.Div(
                id="retrieved-chunks",
                children=[
                    html.Div("Los fragmentos recuperados por RAG aparecerán aquí.",
                             style={"color": C["muted"], "fontSize": "11px",
                                    "lineHeight": "1.7", "fontFamily": MONO})
                ],
                style={
                    "padding": "16px 20px",
                    "overflowY": "auto",
                    "flex": "1",
                    "borderBottom": f"1px solid {C['border']}",
                }
            ),

            # Info sistema
            html.Div([
                html.Div("sistema", style={
                    "fontSize": "9px",
                    "letterSpacing": "0.18em",
                    "textTransform": "uppercase",
                    "color": C["muted"],
                    "fontFamily": MONO,
                    "marginBottom": "14px",
                }),
                html.Div(id="system-info"),
            ], style={"padding": "18px 20px"}),

        ], style={
            "width": "240px",
            "flexShrink": "0",
            "backgroundColor": C["surface"],
            "borderLeft": f"1px solid {C['border']}",
            "height": "100vh",
            "display": "flex",
            "flexDirection": "column",
            "overflowY": "auto",
        }),

    ], style={
        "display": "flex",
        "height": "100vh",
        "overflow": "hidden",
        "backgroundColor": C["bg"],
        "color": C["text"],
        "fontFamily": MONO,
    }),

], style={"backgroundColor": C["bg"]})


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    [Output("header-status", "children"),
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
    msg   = STATUS["message"]
    err   = STATUS["error"]

    dot_color = C["green"] if ready else (C["red"] if err else C["gold"])
    badge = html.Div([
        html.Span("●", style={"color": dot_color, "marginRight": "6px", "fontSize": "10px"}),
        html.Span(msg, style={"fontSize": "10px", "color": C["muted"], "fontFamily": MONO}),
    ], style={"display": "flex", "alignItems": "center"})

    if not ready:
        return badge, no_update, no_update, no_update, [{"label": "todos", "value": ""}], no_update, False

    n_songs  = len(df_corpus) if df_corpus is not None else 0
    col_g    = next((c for c in ["genre", "genero"] if c in (df_corpus.columns if df_corpus is not None else [])), None)
    n_genres = df_corpus[col_g].nunique() if df_corpus is not None and col_g else 0
    n_chunks = len(rag_chunks) if rag_chunks else 0
    col_l    = next((c for c in ["lyrics", "letra"] if c in (df_corpus.columns if df_corpus is not None else [])), None)
    has_lyrics = col_l is not None and df_corpus is not None and df_corpus[col_l].notna().sum() > 100

    def stat_row(value, label, color):
        return html.Div([
            html.Span(f"{value:,}" if isinstance(value, int) else str(value),
                      style={"color": color, "fontSize": "18px", "fontFamily": SERIF,
                             "fontWeight": "300"}),
            html.Span(f" {label}", style={"fontSize": "10px", "color": C["muted"],
                                           "fontFamily": MONO}),
        ], style={"marginBottom": "4px"})

    stats = html.Div([
        stat_row(n_songs,  "canciones", C["text"]),
        stat_row(n_genres, "géneros",   C["gold"]),
        stat_row(n_chunks, "chunks",    "#8ab4c8"),
        html.Div("letras disponibles" if has_lyrics else "sin columna 'lyrics'",
                 style={"fontSize": "9px", "color": C["green"] if has_lyrics else C["red"],
                        "fontFamily": MONO, "marginTop": "6px", "letterSpacing": "0.05em"}),
    ])

    genre_opts = [{"label": "todos", "value": ""}]
    if df_corpus is not None and col_g:
        for g in sorted(df_corpus[col_g].dropna().unique()):
            genre_opts.append({"label": str(g).lower(), "value": str(g)})

    from src.finetuning_utils import classifier_available
    clf_ok   = classifier_available()
    api_mode = chatbot._api_mode if chatbot else "—"

    sys_info = html.Div([
        _info_row("generador",    api_mode),
        _info_row("clasificador", "activo" if clf_ok else "no entrenado"),
        _info_row("chunking",     CHUNKING_STRATEGY),
        _info_row("letras",       "sí" if has_lyrics else "no"),
    ])

    return badge, stats, make_genre_chart(), make_decade_chart(), genre_opts, sys_info, True


def _info_row(label, value):
    return html.Div([
        html.Span(label, style={"color": C["muted"], "fontSize": "10px",
                                 "marginRight": "6px", "fontFamily": MONO}),
        html.Span(value, style={"color": C["text"], "fontSize": "10px",
                                 "fontFamily": MONO}),
    ], style={"marginBottom": "5px"})


@app.callback(
    Output("api-status", "children"),
    Input("apply-api-btn", "n_clicks"),
    State("api-key-input", "value"),
    prevent_initial_call=True,
)
def apply_api_key(n, key):
    if not key or not key.strip():
        return "ingresa una key primero"
    key = key.strip()
    if key.startswith("sk-ant-"):
        os.environ["ANTHROPIC_API_KEY"] = key
        if chatbot:
            chatbot._api_mode = "anthropic"
        return "claude api activa"
    elif key.startswith("sk-"):
        os.environ["OPENAI_API_KEY"] = key
        if chatbot:
            chatbot._api_mode = "openai"
        return "openai api activa"
    return "formato no reconocido"


@app.callback(
    [Output("chat-messages", "children"),
     Output("chat-store", "data"),
     Output("user-input", "value"),
     Output("retrieved-chunks", "children"),
     Output("intent-store", "data")],
    [Input("send-btn", "n_clicks"),
     Input("user-input", "n_submit"),
     Input("clear-btn", "n_clicks"),
     *[Input(f"eg-{i}", "n_clicks") for i in range(len(EXAMPLE_QUESTIONS))]],
    [State("user-input", "value"),
     State("chat-store", "data")],
)
def handle_chat(send, n_submit, clear, *args):
    n_eg           = len(EXAMPLE_QUESTIONS)
    example_clicks = args[:n_eg]
    user_text      = args[n_eg]
    history        = args[n_eg + 1] if len(args) > n_eg + 1 else []
    triggered           = [c["prop_id"] for c in callback_context.triggered]

    if any("clear-btn" in t for t in triggered):
        if chatbot:
            chatbot.reset_history()
        return (
            [render_message("assistant", "Historial limpiado. ¿Sobre qué quieres preguntar?")],
            [], "",
            [html.Div("Esperando consulta...", style={"color": C["muted"], "fontSize": "11px",
                                                       "fontFamily": MONO})],
            "general"
        )

    for i, q in enumerate(EXAMPLE_QUESTIONS):
        if any(f"eg-{i}" in t for t in triggered):
            user_text = q
            break

    if not user_text or not user_text.strip():
        return no_update, no_update, no_update, no_update, no_update

    if not STATUS["ready"] or chatbot is None:
        msgs = list(history or [])
        msgs.append({"role": "user",      "content": user_text})
        msgs.append({"role": "assistant", "content": "El sistema aún está iniciando. Espera un momento.",
                     "intent": "general"})
        return _render_history(msgs), msgs, "", no_update, "general"

    try:
        response, retrieved = chatbot.chat(user_text, use_rag=True)
        from src.chatbot_engine import detect_intent
        intent = detect_intent(user_text)
    except Exception as e:
        response  = f"Error: {e}"
        retrieved = []
        intent    = "general"

    new_history = list(history or [])
    new_history.append({"role": "user",      "content": user_text, "intent": "general"})
    new_history.append({"role": "assistant", "content": response,  "intent": intent})

    # Chunks UI
    if retrieved:
        chunks_ui = []
        for c in retrieved:
            score_pct    = int(c.get("score", 0) * 100)
            texto_preview = c.get("texto", "")
            chunks_ui.append(html.Div([
                html.Div([
                    html.Span(c["titulo"],
                              style={"color": C["text"], "fontSize": "11px",
                                     "fontFamily": MONO}),
                    html.Span(f"  {c['artista']}",
                              style={"color": C["gold"], "fontSize": "11px",
                                     "fontFamily": MONO}),
                ], style={"marginBottom": "4px"}),
                html.Div(
                    f"{c['genero']} · {c['año']} · {score_pct}%",
                    style={"fontSize": "9px", "color": C["muted"], "marginBottom": "6px",
                           "fontFamily": MONO, "letterSpacing": "0.05em"}
                ),
                html.Div(
                    texto_preview[:220] + ("…" if len(texto_preview) > 220 else ""),
                    style={"fontSize": "10px", "color": "#666", "fontStyle": "italic",
                           "lineHeight": "1.6", "whiteSpace": "pre-wrap", "fontFamily": MONO}
                ),
            ], style={
                "borderLeft": f"2px solid {C['gold']}55",
                "paddingLeft": "12px",
                "marginBottom": "16px",
            }))
    else:
        chunks_ui = [html.Div("Sin chunks relevantes.", style={
            "color": C["muted"], "fontSize": "11px", "fontFamily": MONO})]

    return _render_history(new_history), new_history, "", chunks_ui, intent


def _render_history(history: list) -> list:
    out = []
    for turn in history:
        intent = turn.get("intent", "general")
        out.append(render_message(turn["role"], turn["content"], intent))
    return out


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'─'*50}")
    print("  músicbot — chatbot musical CUC")
    print(f"{'─'*50}")
    print(f"  url   : http://{APP_HOST}:{APP_PORT}/")
    print(f"  csv   : {CORPUS_CSV}")
    print(f"  rag   : chunking por {CHUNKING_STRATEGY}")
    print(f"{'─'*50}\n")
    app.run(host=APP_HOST, port=APP_PORT, debug=APP_DEBUG)