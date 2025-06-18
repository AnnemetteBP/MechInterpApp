import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from ..tools.embedding_tool.neighbor_drift_plotter import plot_fp_vs_quantized_neighbors, load_model_tokenizer


layout = dbc.Container([
    html.H2("🔁 Neighbor Drift: FP vs Quantized", className="mb-4"),

    dbc.Row([
        dbc.Col([
            html.Label("Token Arithmetic (e.g. king - man + woman)"),
            dcc.Input(id="nd-expression", value="king - man + woman", className="form-control"),
        ], width=6),
        dbc.Col([
            html.Label("Top N Neighbors"),
            dcc.Input(id="nd-top-n", type="number", value=5, className="form-control"),
        ], width=2),
        dbc.Col([
            html.Label("Tokenizer Path (shared)"),
            dcc.Input(id="nd-tokenizer", value="DHL3B/DHL3B-tokenizer", className="form-control")
        ], width=4),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            html.Label("Model 1 Path"),
            dcc.Input(id="nd-model-fp", value="DHL3B/DHL3B-model", className="form-control")
        ]),
        dbc.Col([
            html.Label("Model 2 Path"),
            dcc.Input(id="nd-model-q", value="DHL3B/DHL3B-model", className="form-control")
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
            html.Button("Compare", id="nd-run", className="btn btn-primary")
        ])
    ]),

    html.Hr(),

    dcc.Loading(
        id="nd-loading",
        type="default",
        children=dcc.Graph(id="nd-graph")
    )
])


@callback(
    Output("nd-graph", "figure"),
    Input("nd-run", "n_clicks"),
    State("nd-expression", "value"),
    State("nd-top-n", "value"),
    State("nd-model-fp", "value"),
    State("nd-model-q", "value"),
    State("nd-tokenizer", "value"),
    State("model-precision-1", "value"),
    State("model-precision-2", "value"),
)
def run_neighbor_drift(n_clicks, expr, top_n, model_fp_path, model_q_path, tokenizer_path, model_precision_1, model_precision_2):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    model_fp, tokenizer_fp = load_model_tokenizer(model_fp_path, tokenizer_path, model_precision_1)
    model_q, tokenizer_q = load_model_tokenizer(model_q_path, tokenizer_path, model_precision_2)

    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

    fig = plot_fp_vs_quantized_neighbors(expr, model_fp, tokenizer_fp, model_q, tokenizer_q, top_n)
    return fig
