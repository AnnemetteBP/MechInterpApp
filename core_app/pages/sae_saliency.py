
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import torch
from core_app.app import app
from ..tools.dictionary_learning.sae_heatmap_plotting import plot_sae_heatmap


layout = dbc.Container([
    html.H2("🧠 Single-Model SAE Heatmap", className="mb-4"),

    dbc.Row([
        dbc.Col([
            html.Label("Input Text"),
            dcc.Input(id="sm-text", className="form-control", value="The king led the troops into battle."),
        ])
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            html.Label("Target Layers (comma-separated)"),
            dcc.Input(id="sm-layers", className="form-control", value="5,10,15"),
        ], width=6),
        dbc.Col([
            html.Label("Top K Concepts"),
            dcc.Input(id="sm-topk", type="number", value=5, className="form-control"),
        ], width=2),
        dbc.Col([
            html.Label("Tokens per Row"),
            dcc.Input(id="sm-tpr", type="number", value=12, className="form-control"),
        ], width=2)
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            html.Label("Model Path"),
            dcc.Input(id="sm-model", value="DHL3B/DHL3B-model", className="form-control")
        ]),
        dbc.Col([
            html.Label("Tokenizer Path"),
            dcc.Input(id="sm-tokenizer", value="DHL3B/DHL3B-tokenizer", className="form-control")
        ]),
        dbc.Col([
            html.Label("BnB Config"),
            dcc.Dropdown(
                id="sm-precision",
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
        ])
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            html.Label("Eval Mode"),
            dcc.Checklist(
                id="sm-eval-mode",
                options=[{"label": "Eval Mode", "value": True}],
                value=[True],
                inputStyle={"margin-right":"8px"}
            ),
        ], width=3),
        dbc.Col([
            html.Label("token_font_size"), 
            dcc.Input(id="sm-token-font-size", type="number", value=12, className="form-control"),
        ], width=3),
        dbc.Col([
            html.Label("label_font_size"), 
            dcc.Input(id="sm-label-font-size", type="number", value=14, className="form-control"),
        ], width=3),
        dbc.Col([
            html.Label("Deterministic Backend"),
            dcc.Checklist(
                id="sm-deterministic",
                options=[{"label": "Deterministic SAE", "value": True}],
                value=[],
                inputStyle={"margin-right":"8px"}
            ),
        ], width=3),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            html.Button("Run SAE", id="sm-run", className="btn btn-primary")
        ], width=12)
    ], className="mb-3"),

    html.Hr(),

    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id="sm-loading",
                type="default",
                children=html.Div(id="sm-graph")
            )
        ])
    ])
], fluid=True)


@app.callback(
    Output("sm-graph", "children"),
    Input("sm-run", "n_clicks"),
    State("sm-text", "value"),
    State("sm-layers", "value"),
    State("sm-topk", "value"),
    State("sm-tpr", "value"),
    State("sm-model", "value"),
    State("sm-tokenizer", "value"),
    State("sm-precision", "value"),
    State("sm-eval-mode", "value"),
    State("sm-token-font-size", "value"),
    State("sm-label-font-size", "value"),
    State("sm-deterministic", "value"),
)
def run_single_model_sae(n_clicks, text, layer_str, top_k, tpr, model_path, tokenizer_path, precision,
                         eval_mode, token_font_size, label_font_size, deterministic):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    target_layers = [int(x.strip()) for x in layer_str.split(",") if x.strip().isdigit()]


    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

    return plot_sae_heatmap(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        inputs=text,
        model_precision=precision,
        top_k=top_k,
        tokens_per_row=tpr,
        target_layers=target_layers,
        model_to_eval=eval_mode,
        deterministic_sae=deterministic,
        token_font_size=token_font_size,
        label_font_size=label_font_size,
    )
