import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from ..tools.embedding_tool.sentence_embedding_drift_plotter import plot_sentence_embedding_drift, load_model_tokenizer


layout = dbc.Container([
    html.H2("📐 Sentence Embedding Drift Comparison", className="mb-4"),

    dbc.Row([
        dbc.Col([
            html.Label("Enter sentences (one per line):"),
            dcc.Textarea(id="sd-sentences", className="form-control", value="The king rules.\nThe queen leads."),
        ], width=12)
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            html.Label("Pooling Method"),
            dcc.Dropdown(
                id="sd-method",
                options=[
                    {"label": "Mean Pooling", "value": "mean"},
                    {"label": "[CLS] Token", "value": "cls"},
                ],
                value="mean"
            )
        ], width=4),

        dbc.Col([
            html.Label("Transformer Layer (e.g. -1 for last)"),
            dcc.Input(id="sd-layer", type="number", value=-1, className="form-control"),
        ], width=3),
        
        dbc.Col([
            html.Label("Tokenizer Path (shared)"),
            dcc.Input(id="sd-tokenizer", value="DHL3B/DHL3B-tokenizer", className="form-control")
        ], width=4),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            html.Label("Model 1 Path"),
            dcc.Input(id="sd-model-fp", value="DHL3B/DHL3B-model", className="form-control")
        ]),
        dbc.Col([
            html.Label("Model 2 Path"),
            dcc.Input(id="sd-model-q", value="DHL3B/DHL3B-model", className="form-control")
        ]),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            html.Label("BnB Config Model 1"),
            dcc.Dropdown(
                id="model-precision-1",
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
        ], width=6),
        dbc.Col([
            html.Label("BnB Config Model 2"),
            dcc.Dropdown(
                id="model-precision-2",
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
        ], width=6),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            html.Button("Compare Embeddings", id="sd-run", className="btn btn-primary")
        ])
    ]),

    html.Hr(),

    dcc.Loading(
        id="sd-loading",
        type="default",
        children=dcc.Graph(id="sd-graph")
    )
])


@callback(
    Output("sd-graph", "figure"),
    Input("sd-run", "n_clicks"),
    State("sd-sentences", "value"),
    State("sd-method", "value"),
    State("sd-layer", "value"),
    State("sd-model-fp", "value"),
    State("sd-model-q", "value"),
    State("sd-tokenizer", "value"),
    State("model-precision-1", "value"),
    State("model-precision-2", "value"),
)
def run_sentence_embedding_drift(n_clicks, raw_sentences, method, layer, model_fp_path, model_q_path, tokenizer_path, model_precision_1, model_precision_2):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    sentences = [s.strip() for s in raw_sentences.strip().split('\n') if s.strip()]
    model_fp, tokenizer_fp = load_model_tokenizer(model_fp_path, tokenizer_path, model_precision_1)
    model_q, tokenizer_q = load_model_tokenizer(model_q_path, tokenizer_path, model_precision_2)

    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

    fig = plot_sentence_embedding_drift(sentences, model_fp, tokenizer_fp, model_q, tokenizer_q, top_layer=layer, method=method)
    return fig
