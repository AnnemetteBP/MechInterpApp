import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
from ..tools.attention_tool.attention_analysis import (
    get_attention_figure, 
    load_model_tokenizer,
    get_attention_weights,
    plot_attention_heatmap,
    compute_attention_entropy,
    plot_entropy_heatmap,
    plot_qk_embeddings)

layout = dbc.Container([
    html.H2("🔍 Transformer Attention Head Explorer", className="mb-4"),

    dbc.Row([
        dbc.Col([html.Label("Model Path"), dcc.Input(id="attn-model", value="DHL3B/DHL3B-model", className="form-control")]),
        dbc.Col([html.Label("Tokenizer Path"), dcc.Input(id="attn-tokenizer", value="DHL3B/DHL3B-tokenizer", className="form-control")]),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([html.Label("Input Text"), dcc.Input(id="attn-text", value="The quick brown fox jumps over the lazy dog.", className="form-control")])
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([html.Label("Layer Index"), dcc.Input(id="attn-layer", type="number", value=0, className="form-control")], width=3),
        dbc.Col([html.Label("Head Index"), dcc.Input(id="attn-head", type="number", value=0, className="form-control")], width=3),
        dbc.Col([html.Label("Use Softmax"), dcc.Checklist(id="attn-softmax", options=[{"label": "Apply Softmax", "value": "yes"}], value=["yes"])], width=3),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            html.Label("Visualization Type"),
            dcc.Dropdown(
                id="attn-vis-mode",
                options=[
                    {"label": "Attention Heatmap", "value": "heatmap"},
                    {"label": "Top-k Heatmap", "value": "topk"},
                    {"label": "Entropy", "value": "entropy"},
                    {"label": "Q/K Projections", "value": "qk"},
                ],
                value="heatmap",
                clearable=False,
                className="form-select"
            )
        ], width=4),

        dbc.Col([
            html.Label("Top-k Tokens"),
            dcc.Slider(id="attn-topk", min=1, max=20, step=1, value=10, marks=None, tooltip={"placement": "bottom"})
        ], width=6),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(html.Button("Visualize Attention", id="attn-run", className="btn btn-primary"), width=3)
    ]),

    html.Hr(),

    dbc.Row([
        dbc.Col([
            dcc.Loading(children=html.Div(id="attn-output"), type="default")
        ])
    ])
])


@callback(
    Output("attn-output", "children"),
    Input("attn-run", "n_clicks"),
    State("attn-text", "value"),
    State("attn-model", "value"),
    State("attn-tokenizer", "value"),
    State("attn-layer", "value"),
    State("attn-head", "value"),
    State("attn-softmax", "value"),
    State("attn-vis-mode", "value"),
    State("attn-topk", "value")
)
def update_attention(n_clicks, text, model_path, tokenizer_path, layer_idx, head_idx,
                     softmax_val, vis_mode, top_k):

    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    model, tokenizer = load_model_tokenizer(model_path, tokenizer_path, None)
    use_softmax = "yes" in softmax_val

    if vis_mode == "heatmap":
        fig = get_attention_figure(model, tokenizer, text, layer_idx, head_idx, use_softmax)

    elif vis_mode == "topk":
        attn, tokens = get_attention_weights(model, tokenizer, text, layer_idx, head_idx)
        fig = plot_attention_heatmap(attn, tokens, top_k=top_k)

    elif vis_mode == "entropy":
        attn, tokens = get_attention_weights(model, tokenizer, text, layer_idx, head_idx)
        entropy = compute_attention_entropy(attn)
        fig = plot_entropy_heatmap(entropy, tokens)

    elif vis_mode == "qk":
        fig = plot_qk_embeddings(model, tokenizer, text, layer=layer_idx, mode="pca")

    else:
        return html.Div("Invalid visualization mode.")

    return dcc.Graph(figure=fig)

