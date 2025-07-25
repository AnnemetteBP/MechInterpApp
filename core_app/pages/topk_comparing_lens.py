from dash import html, dcc, Input, Output
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import torch
from core_app.app import app
from ..tools.logit_lens import topk_comparing_plotter


PROMPT = """Richmond Football Club Richmond began 2017 with 5 straight wins, a feat it had not achieved since 1995. A series of close losses hampered the Tigers throughout the middle of the season, including a 5-point loss to the Western Bulldogs, 2-point loss to Fremantle, and a 3-point loss to the Giants. Richmond ended the season strongly with convincing victories over Fremantle and St Kilda in the final two rounds, elevating the club to 3rd on the ladder. Richmond's first final of the season against the Cats at the MCG attracted a record qualifying final crowd of 95,028; the Tigers won by 51 points. Having advanced to the first preliminary finals for the first time since 2001, Richmond defeated Greater Western Sydney by 36 points in front of a crowd of 94,258 to progress to the Grand Final against Adelaide, their first Grand Final appearance since 1982. The attendance was 100,021, the largest crowd to a grand final since 1986. The Crows led at quarter time and led by as many as 13, but the Tigers took over the game as it progressed and scored seven straight goals at one point. They eventually would win by 48 points – 16.12 (108) to Adelaide's 8.12 (60) – to end their 37-year flag drought.[22] Dustin Martin also became the first player to win a Premiership medal, the Brownlow Medal and the Norm Smith Medal in the same season, while Damien Hardwick was named AFL Coaches Association Coach of the Year. Richmond's jump from 13th to premiers also marked the biggest jump from one AFL season to the next."""

MODELS = ['DHL3B/DHL3B-model', 'LI8B/LI-model', 'HF1BitLLM/HF1BitLLM-model']
TOKS = ['DHL3B/DHL3B-tokenizer', 'LI8B/LI-tokenizer', 'HF1BitLLM/HF1BitLLM-tokenizer']


layout = dbc.Container([

    html.H2("🔍 TopK-N Comparing Lens", className="mb-4"),

    # Model & Tokenizer row
    dbc.Row([
        dbc.Col([
            html.Label("Model ID/Path"),
            dcc.Input(id="model-id-1", value=MODELS[2], className="form-control"),
            html.Label("Model ID/Path"),
            dcc.Input(id="model-id-2", value=MODELS[1], className="form-control"),
        ], width=6),
        dbc.Col([
            html.Label("Tokenizer ID/Path"),
            dcc.Input(id="tokenizer-id-1", value=TOKS[2], className="form-control"),
            html.Label("Tokenizer ID/Path"),
            dcc.Input(id="tokenizer-id-2", value=TOKS[1], className="form-control"),
        ], width=6),
    ], className="mb-3"),

    # Prompt row
    dbc.Row([
        dbc.Col([
            html.Label("Prompt"),
            dcc.Textarea(
                id="input-text-0",
                placeholder="Your prompt here: e.g., What is y if y=2*2-4+(3*2)",
                #value="the cat cat is on the mat mat",
                value=PROMPT,
                className="form-control",
                style={"height": "80px"}
            ),
        ], width=12),
    ], className="mb-3"),

    # Start / End / Top-K row
    dbc.Row([
        dbc.Col([
            html.Label("Start Index"),
            dcc.Input(id="start-ix-0", type="number", value=0, className="form-control"),
        ], width=3),
        dbc.Col([
            html.Label("End Index"),
            dcc.Input(id="end-ix-0", type="number", value=10, className="form-control"),
        ], width=3),
        dbc.Col([
            html.Label("Top-K"),
            dcc.Input(id="top-k-0", type="number", value=5, className="form-control"),
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

    # Metric selector + Plot button row
    dbc.Row([
        dbc.Col([
            html.Label("Metric"),
            dcc.Dropdown(
                id="metric-type-0",
                options=[
                    {"label":"JS Divergence","value":"js"},
                    {"label":"Normalized Wasserstein","value":"nwd"},
                ],
                value="nwd",
                clearable=False,
                className="form-select"
            ),
        ], width=8),
        dbc.Col([
            html.Button("Plot", id="plot-btn-0", n_clicks=0, className="btn btn-primary mt-3"),
        ], width=4, className="d-flex align-items-start justify-content-end"),
    ], className="mb-4"),

    # Advanced settings header
    html.H4("Advanced settings", className="mb-3"),

    # block_step / font sizes row
    dbc.Row([
        dbc.Col([
            html.Label("block_step"), 
            dcc.Input(id="block-step-0", type="number", value=1, className="form-control"),
        ], width=4),
        dbc.Col([
            html.Label("token_font_size"), 
            dcc.Input(id="token-font-size-0", type="number", value=12, className="form-control"),
        ], width=4),
        dbc.Col([
            html.Label("label_font_size"), 
            dcc.Input(id="label-font-size-0", type="number", value=20, className="form-control"),
        ], width=4),
    ], className="mb-3"),

    # Flags checklist row
    dbc.Row([
        dbc.Col([
            dcc.Checklist(
                id="flags-0",
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
                id="decoder-layers-0",
                options=[{"label":nm,"value":nm} for nm in ["norm","lm_head"]],
                value=["norm","lm_head"],
                multi=True,
                className="form-select"
            ),
        ], width=4),
        dbc.Col([
            html.Label("bnb_config"),
            dcc.Dropdown(
                id="model-precision-0",
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
        ], width=4),
        dbc.Col([
            html.Label("bnb_config"),
            dcc.Dropdown(
                id="model-precision-00",
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
        ], width=4),
    ], className="mb-4"),

    # Graph row
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="logit-lens-graph-0", style={"height":"700px"}),
        ], width=12),
    ]),

], fluid=True)


@app.callback(
    Output("logit-lens-graph-0", "figure"),
    [
      Input("plot-btn-0",     "n_clicks"),
      Input("model-id-1",        "value"),
      Input("model-id-2",        "value"),
      Input("tokenizer-id-1",    "value"),
      Input("tokenizer-id-2",    "value"),
      Input("input-text-0",      "value"),
      Input("start-ix-0",        "value"),
      Input("end-ix-0",          "value"),
      Input("top-k-0",           "value"),
      Input("metric-type-0",     "value"),
      Input("block-step-0",      "value"),
      Input("token-font-size-0", "value"),
      Input("label-font-size-0", "value"),
      Input("flags-0",           "value"),
      Input("decoder-layers-0",  "value"),
      Input("model-precision-0", "value"),
      Input("model-precision-00","value"),
      Input("deterministic",     "value"),
    ]
)
def update_logit_lens(
    n,
    model_id_1, model_id_2,
    tokenizer_id_1, tokenizer_id_2,
    text,
    start_ix, end_ix, top_k, metric,
    block_step, token_font_size, label_font_size,
    flags, decoder_layers, model_precision_1, model_precision_2,
    deterministic
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
        return topk_comparing_plotter.plot_topk_comparing_lens(
            model_1              = model_id_1,
            model_2              = model_id_2,
            tokenizer_1          = tokenizer_id_1,
            tokenizer_2          = tokenizer_id_2,
            inputs               = text,
            start_ix             = start_ix,
            end_ix               = end_ix,
            topk                 = top_k,
            topk_mean            = topk_mean,
            js                   = (metric=="js"),
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
            model_precision_1    = model_precision_1,
            model_precision_2    = model_precision_2,
            use_deterministic_backend = use_deterministic
        ) or go.Figure()
    except Exception as e:
        return go.Figure().add_annotation(
            text=f"Error:\n{e}",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="red", size=14)
        )
    