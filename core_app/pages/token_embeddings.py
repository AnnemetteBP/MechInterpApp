import dash
from dash import html, dcc, Input, Output, State
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import torch
import numpy as np
from sklearn.decomposition import PCA
from core_app.app import app
from ..tools.embedding_tool.token_embedding_plotter import load_model_tokenizer, visualize_token_embedding_arithmetic



layout = dbc.Container([
    html.H2("🔬 Embedding Arithmetic Visualizer", className="mb-4"),

    # === Add model and tokenizer input fields ===
    dbc.Row([
        dbc.Col([
            html.Label("Model Path / Hugging Face ID"),
            dcc.Input(id="model-id", value="DHL3B/DHL3B-model", className="form-control"),
        ], width=5),
        dbc.Col([
            html.Label("Tokenizer Path / Hugging Face ID"),
            dcc.Input(id="tokenizer-id", value="DHL3B/DHL3B-tokenizer", className="form-control"),
        ], width=4),
        dbc.Col([
            html.Label("BnB Config"),
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
        ], width=3),
    ], className="mb-3"),

    # === Existing expression and top-N fields ===
    dbc.Row([
        dbc.Col([
            html.Label("Token Arithmetic (e.g. king - man + woman)"),
            dcc.Input(id="embedding-expression", value="king - man + woman", className="form-control"),
        ], width=8),
        dbc.Col([
            html.Label("Top N Neighbors"),
            dcc.Input(id="top-n-neighbors", type="number", value=5, className="form-control"),
        ], width=4)
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            html.Button("Visualize Embedding", id="embedding-submit", className="btn btn-primary")
        ])
    ]),

    html.Hr(),

    dcc.Loading(
        id="loading-embedding-fig",
        type="default",
        children=dcc.Graph(id="embedding-graph")
    )
])


@app.callback(
    Output("embedding-graph", "figure"),
    Input("embedding-submit", "n_clicks"),
    State("embedding-expression", "value"),
    State("top-n-neighbors", "value"),
    State("model-id", "value"),
    State("tokenizer-id", "value"),
    State("model-precision", "value"),
)
def update_embedding_graph(n_clicks, expression, top_n, model_id, tokenizer_id, model_precision):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    model, tokenizer = load_model_tokenizer(model_id, tokenizer_id, model_precision)

    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

    fig = visualize_token_embedding_arithmetic(model, tokenizer, expression, top_n)
    return fig
