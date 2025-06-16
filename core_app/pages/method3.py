from dash import html, dcc, Input, Output
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import torch
from core_app.app import app
from ..tools.dictionary_learning import sae_heatmap_plotting


layout = dbc.Container([

    html.H2("🧠 SAE Saliency Heatmap", className="mb-4"),

    # Model & Tokenizer row
    dbc.Row([
        dbc.Col([
            html.Label("Model ID/Path"),
            dcc.Input(id="model-id", value="DHL3B/DHL3B-model", className="form-control"),
        ], width=6),
        dbc.Col([
            html.Label("Tokenizer ID/Path"),
            dcc.Input(id="tokenizer-id", value="DHL3B/DHL3B-tokenizer", className="form-control"),
        ], width=6),
    ], className="mb-3"),

    # Prompt row
    dbc.Row([
        dbc.Col([
            html.Label("Prompt"),
            dcc.Textarea(
                id="input-text",
                placeholder="Your prompt here: e.g., What is y if y=2*2-4+(3*2)",
                value="the cat cat is on the mat mat",
                className="form-control",
                style={"height": "80px"}
            ),
        ], width=12),
    ], className="mb-3"),
    # Start / End / Top-K row
    dbc.Row([
        dbc.Col([
            html.Label("Tokens per row"),
            dcc.Input(id="tokens-per-row", type="number", value=0, className="form-control"),
        ], width=3),
        dbc.Col([
            html.Label("Eval Mode"),
            dcc.Checklist(
                id="eval-mode",
                options=[{"label": "Eval Mode", "value": True}],
                value=[True],
                inputStyle={"margin-right":"8px"}
            ),
        ], width=3),
        dbc.Col([
            html.Label("Top-K"),
            dcc.Input(id="top-k", type="number", value=5, className="form-control"),
        ], width=3),
        dbc.Col([
            html.Label("Deterministic Backend"),
            dcc.Checklist(
                id="deterministic",
                options=[{"label": "Deterministic SAE", "value": True}],
                value=[],
                inputStyle={"margin-right":"8px"}
            ),
        ], width=3),
    ], className="mb-3"),

    # Metric selector + Plot button row
    dbc.Row([
        dbc.Col([
            html.Label("bnb_config"),
            dcc.Dropdown(
                id="model-precision",
                options=[
                    {"label":"None","value":""},
                    {"label":"PTDQ 8-bit","value":"ptdq8bit"},
                    {"label":"PTDQ 4-bit","value":"ptdq4bit"},
                    {"label":"PTSQ 8-bit","value":"ptsq8bit"},
                    {"label":"PTSQ 4-bit","value":"ptsq4bit"},
                ],
                value="",
                clearable=False,
                className="form-select"
            ),
        ], width=8),

        dbc.Col([
            html.Button("Plot", id="plot-btn", n_clicks=0, className="btn btn-primary mt-3"),
        ], width=4, className="d-flex align-items-start justify-content-end"),
    ], className="mb-4"),

    # Advanced settings header
    html.H4("Advanced settings", className="mb-3"),

    # block_step / font sizes row
    dbc.Row([
        dbc.Col([
            html.Label("block_step"), 
            dcc.Input(id="block-step", type="number", value=1, className="form-control"),
        ], width=4),
        dbc.Col([
            html.Label("token_font_size"), 
            dcc.Input(id="token-font-size", type="number", value=12, className="form-control"),
        ], width=4),
        dbc.Col([
            html.Label("label_font_size"), 
            dcc.Input(id="label-font-size", type="number", value=20, className="form-control"),
        ], width=4),
    ], className="mb-3"),

    # Flags checklist row
    dbc.Row([
        dbc.Col([
            dcc.Checklist(
                id="flags",
                options=[
                    {"label":"Mean Top-K","value":"topk_mean"},
                ],
                value=["topk_mean"],
                inline=True,
                inputStyle={"margin-right":"8px","margin-left":"16px"}
            )
        ], width=12),
    ], className="mb-3"),

    # decoder layers / precision row
    dbc.Row([
        dbc.Col([], width=3),
        dbc.Col([
            html.Label("target_layers"),
            dcc.Dropdown(
                id="target-layers",
                options=[{"label":nm,"value":nm} for nm in [1, 5, 10, 20, 25]],
                value=[1, 5, 10, 20, 25],
                multi=True,
                className="form-select"
            ),
        ], width=6),
        dbc.Col([], width=3),
    ], className="mb-4"),

    # Graph row
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="sae-saliency-graph", style={"height":"700px"}),
        ], width=12),
    ]),

], fluid=True)


@app.callback(
    Output("sae-saliency-graph", "figure"),
    [
        Input("plot-btn",          "n_clicks"),
        Input("model-id",          "value"),
        Input("tokenizer-id",      "value"),
        Input("input-text",        "value"),
        Input("top-k",             "value"),
        Input("tokens-per-row",    "value"),
        Input("token-font-size",   "value"),
        Input("label-font-size",   "value"),
        Input("flags",             "value"),
        Input("target-layers",     "value"),
        Input("model-precision",   "value"),
        Input("eval-mode",         "value"),
        Input("deterministic",     "value"),
    ]
)
def update_sae_saliency(
    n_clicks, model_id, tok_id, text,
    top_k, tokens_per_row,
    token_font_size, label_font_size,
    flags, target_layers, model_precision,
    eval_mode, deterministic
):
    if not n_clicks or not text:
        return go.Figure()

    topk_mean         = "topk_mean" in flags
    model_to_eval     = True if eval_mode else False
    deterministic_sae = True if deterministic else False
    tokens_per_row    = tokens_per_row or 25
    top_k             = top_k or 5
    target_layers     = target_layers or []

    try:
        return sae_heatmap_plotting.plot_sae_heatmap(
            model_path        = model_id,
            tokenizer_path    = tok_id,
            inputs            = text,
            model_precision   = model_precision,
            top_k             = top_k,
            #topk_mean         = topk_mean,
            tokens_per_row    = tokens_per_row,
            target_layers     = target_layers,
            model_to_eval     = model_to_eval,
            deterministic_sae = deterministic_sae,
            token_font_size   = token_font_size,
            label_font_size   = label_font_size
        ) or go.Figure()
    except Exception as e:
        return go.Figure().add_annotation(
            text=f"Error:\n{e}",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="red", size=14)
        )