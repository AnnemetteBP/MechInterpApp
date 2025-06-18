import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import torch
from core_app.app import app
from ..tools.dictionary_learning.sae_comparing_plotting import plot_comparing_heatmap, load_model_tokenizer


layout = dbc.Container([
    html.H2("🧠 SAE Heatmap Comparison", className="mb-4"),

    dbc.Row([
        dbc.Col([
            html.Label("Input Text"),
            dcc.Input(id="sh-text", className="form-control", value="The king led the troops into battle."),
        ])
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            html.Label("Target Layers (comma-separated)"),
            dcc.Input(id="sh-layers", className="form-control", value="5,10,15"),
        ], width=6),
        dbc.Col([
            html.Label("Top K Concepts"),
            dcc.Input(id="sh-topk", type="number", value=5, className="form-control"),
        ], width=2),
        dbc.Col([
            html.Label("Tokens per Row"),
            dcc.Input(id="sh-tpr", type="number", value=12, className="form-control"),
        ], width=2)
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            html.Label("Model 1 Path"),
            dcc.Input(id="sh-model-fp", value="DHL3B/DHL3B-model", className="form-control")
        ]),
        dbc.Col([
            html.Label("Model 2 Path"),
            dcc.Input(id="sh-model-q", value="DHL3B/DHL3B-model", className="form-control")
        ]),
        dbc.Col([
            html.Label("Tokenizer Path"),
            dcc.Input(id="sh-tokenizer", value="DHL3B/DHL3B-tokenizer", className="form-control")
        ])
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
    ], className="mb-4"),

    dbc.Row([
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
            html.Label("token_font_size"), 
            dcc.Input(id="token-font-size", type="number", value=12, className="form-control"),
        ], width=3),
        dbc.Col([
            html.Label("label_font_size"), 
            dcc.Input(id="label-font-size", type="number", value=14, className="form-control"),
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

    dbc.Row([
        dbc.Col([
            html.Button("Compare SAE", id="sh-run", className="btn btn-primary")
        ], width=12)
    ], className="mb-3"),

    html.Hr(),

    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id="sh-loading",
                type="default",
                children=html.Div(id="sh-graph")  # ← not dcc.Graph anymore!
            )
        ])
    ])
])


@callback(
    Output("sh-graph", "children"),
    Input("sh-run", "n_clicks"),
    State("sh-text", "value"),
    State("sh-layers", "value"),
    State("sh-topk", "value"),
    State("sh-tpr", "value"),
    State("sh-model-fp", "value"),
    State("sh-model-q", "value"),
    State("sh-tokenizer", "value"),
    State("model-precision-1", "value"),
    State("model-precision-2", "value"),
    State("eval-mode", "value"),
    State("token-font-size", "value"),
    State("label-font-size", "value"),
    State("deterministic", "value"),
)
def run_sae_heatmap(n_clicks, text, layer_str, top_k, tpr, model_fp_path, model_q_path, tokenizer_path,
                    model_precision_1, model_precision_2, eval_mode, token_font_size, label_font_size, deterministic):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    target_layers = [int(x.strip()) for x in layer_str.split(",") if x.strip().isdigit()]

    model_fp, tokenizer = load_model_tokenizer(model_fp_path, tokenizer_path, model_precision_1)
    model_q, _ = load_model_tokenizer(model_q_path, tokenizer_path, model_precision_2)

    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

    fig = plot_comparing_heatmap(
        models=(model_fp, model_q),
        tokenizer=tokenizer,
        inputs=text,
        top_k=top_k,
        tokens_per_row=tpr,
        target_layers=target_layers,
        model_to_eval=eval_mode,
        deterministic_sae=deterministic,
        token_font_size=token_font_size,
        label_font_size=label_font_size,
    )
    return fig
