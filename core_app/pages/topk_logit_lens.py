from dash import html, dcc, Input, Output
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import torch
from core_app.app import app
from ..tools.logit_lens import topk_lens_plotter

layout = dbc.Container([

    html.H2("🔍 TopK-N Logit Lens", className="mb-4"),

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
            html.Label("Start Index"),
            dcc.Input(id="start-ix", type="number", value=0, className="form-control"),
        ], width=4),
        dbc.Col([
            html.Label("End Index"),
            dcc.Input(id="end-ix", type="number", value=10, className="form-control"),
        ], width=4),
        dbc.Col([
            html.Label("Top-K"),
            dcc.Input(id="top-k", type="number", value=5, className="form-control"),
        ], width=4),
    ], className="mb-3"),

    # Metric selector + Plot button row
    dbc.Row([
        dbc.Col([
            html.Label("Metric"),
            dcc.Dropdown(
                id="metric-type",
                options=[
                    {"label":"Logits","value":"logits"},
                    {"label":"Probs","value":"probs"},
                    {"label":"Entropy","value":"entropy"},
                    {"label":"KL Div","value":"kl"},
                    {"label":"Cosine Sim","value":"cosine_sims"},
                    {"label":"KL layer","value":"kl_layerwise"},
                    {"label":"Tok variety","value":"token_variety"},
                    {"label":"Ranks","value":"ranks"},
                ],
                value="logits",
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
        ], width=3),
        dbc.Col([
            html.Label("token_font_size"), 
            dcc.Input(id="token-font-size", type="number", value=12, className="form-control"),
        ], width=3),
        dbc.Col([
            html.Label("label_font_size"), 
            dcc.Input(id="label-font-size", type="number", value=20, className="form-control"),
        ], width=3),
        dbc.Col([
            html.Label("use_deterministic_backend"),
            dcc.Checklist(
                id="deterministic",
                options=[{"label": "Deterministic Backend", "value": False}],
                value=[],
                inputStyle={"margin-right":"8px"}
            ),
        ], width=3),
    ], className="mb-3"),

    # Flags checklist row
    dbc.Row([
        dbc.Col([
            dcc.Checklist(
                id="flags",
                options=[
                    {"label":"include_input","value":"include_input"},
                    {"label":"force_include_output","value":"force_include_output"},
                    {"label":"include_subblocks","value":"include_subblocks"},
                    {"label":"top_down","value":"top_down"},
                    {"label":"verbose","value":"verbose"},
                    {"label":"pad_to_max_length","value":"pad_to_max_length"},
                    {"label":"Mean Top-K","value":"topk_mean"},
                ],
                value=["include_input","force_include_output","topk_mean"],
                inline=True,
                inputStyle={"margin-right":"8px","margin-left":"16px"}
            )
        ], width=12),
    ], className="mb-3"),

    # decoder layers / precision row
    dbc.Row([
        dbc.Col([
            html.Label("decoder_layer_names"),
            dcc.Dropdown(
                id="decoder-layers",
                options=[{"label":nm,"value":nm} for nm in ["norm","lm_head"]],
                value=["norm","lm_head"],
                multi=True,
                className="form-select"
            ),
        ], width=6),
        dbc.Col([
            html.Label("bnb_config"),
            dcc.Dropdown(
                id="model-precision",
                options=[
                    {"label":"None","value":""},
                    {"label":"PTDQ 8-bit","value":"ptdq8bit"},
                    {"label":"PTDQ 4-bit","value":"ptdq4bit"},
                    {"label":"PTSQ 1.58-bit","value":"158ptdq"},
                    {"label":"PTSQ 1.58-bit","value":"158ptsq"},
                ],
                value="",
                clearable=False,
                className="form-select"
            ),
        ], width=6),
    ], className="mb-4"),

    # Graph row
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="logit-lens-graph", style={"height":"700px"}),
        ], width=12),
    ]),

], fluid=True)


@app.callback(
    Output("logit-lens-graph", "figure"),
    [
      Input("plot-btn",     "n_clicks"),
      Input("model-id",        "value"),
      Input("tokenizer-id",    "value"),
      Input("input-text",      "value"),
      Input("start-ix",        "value"),
      Input("end-ix",          "value"),
      Input("top-k",           "value"),
      Input("metric-type",     "value"),
      Input("block-step",      "value"),
      Input("token-font-size", "value"),
      Input("label-font-size", "value"),
      Input("flags",           "value"),
      Input("decoder-layers",  "value"),
      Input("model-precision", "value"),
      Input("deterministic",   "value"),
    ]
)
def update_logit_lens(
    n, model_id, tok_id, text,
    start_ix, end_ix, top_k, metric,
    block_step, token_font_size, label_font_size,
    flags, decoder_layers, model_precision, deterministic
):
    if not n or not text:
        return go.Figure()

    # ensure end_ix ≥ start_ix+1
    if end_ix is None or end_ix <= start_ix:
        end_ix = start_ix + 1

    include_input        = "include_input" in flags
    force_include_output = "force_include_output" in flags
    include_subblocks    = "include_subblocks" in flags
    top_down             = "top_down" in flags
    verbose              = "verbose" in flags
    pad_to_max_length    = "pad_to_max_length" in flags
    topk_mean            = "topk_mean" in flags
    use_deterministic    = True if deterministic else False

    try:
        return topk_lens_plotter.plot_topk_logit_lens(
            model_path           = model_id,
            tokenizer_path       = tok_id,
            inputs               = text,
            start_ix             = start_ix,
            end_ix               = end_ix,
            topk                 = top_k,
            topk_mean            = topk_mean,
            entropy              = (metric=="entropy"),
            probs                = (metric=="probs"),
            kl                   = (metric=="kl"),
            cosine_sims          = (metric=="cosine_sims"),
            kl_layerwise         = (metric=="kl_layerwise"),
            token_variety        = (metric=="token_variety"),
            ranks                = (metric=="ranks"),
            block_step           = block_step,
            token_font_size      = token_font_size,
            label_font_size      = label_font_size,
            include_input        = include_input,
            force_include_output = force_include_output,
            include_subblocks    = include_subblocks,
            decoder_layer_names  = decoder_layers,
            top_down             = top_down,
            verbose              = verbose,
            pad_to_max_length    = pad_to_max_length,
            model_precision      = model_precision,
            use_deterministic_backend = use_deterministic
        ) or go.Figure()
    except Exception as e:
        return go.Figure().add_annotation(
            text=f"Error:\n{e}",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="red", size=14)
        )