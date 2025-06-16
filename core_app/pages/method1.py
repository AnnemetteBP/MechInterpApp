from dash import html, dcc, Input, Output
import plotly.graph_objects as go
from core_app.app import app
import torch

# import the module (so we call module.function)
from ..tools.logit_lens import topk_lens_plotter

layout = html.Div([
    html.H2("The TopK-N Logit Lens"),
    dcc.Input(id="model-id", placeholder="model (e.g. LLaMA / OLMo)", value="NousResearch/DeepHermes-3-Llama-3-3B-Preview"),
    dcc.Input(id="tokenizer-id", placeholder="tokenizer", value="NousResearch/DeepHermes-3-Llama-3-3B-Preview"),
    #dcc.Textarea(id="input-text", placeholder="Your prompt here: e.g., What is y if y=2*2-4+(3*2)", style={"width":"100%","height":80}),
    # Prompt and slicing
    html.Div([
        html.Label("Prompt:"),
        dcc.Textarea(id="input-text", placeholder="Your prompt here: e.g., What is y if y=2*2-4+(3*2)", style={"width":"100%","height":80}),
    ], style={"margin-bottom":"1em"}),
    html.Div([
        html.Label("Start Index:"), 
        dcc.Input(id="start-ix",  type="number", value=0, style={"width":"6em"}),
        html.Label("End Index:", style={"margin-left":"2rem"}), 
        dcc.Input(id="end-ix",    type="number", value=10, style={"width":"6em"}),
        html.Label("Top-K:", style={"margin-left":"2rem"}), 
        dcc.Input(id="top-k",      type="number", value=5, style={"width":"6em"}),
    ], style={"margin-bottom":"1em"}),

    # Metric selector
    html.Div([
        html.Label("Metric:"),
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
            style={"width":"300px"}
        )
    ], style={"margin-bottom":"1em"}),

    # Advanced parameters
    html.H4("Advanced settings"),
    html.Div([
        html.Label("block_step:"), dcc.Input(id="block-step", type="number", value=1, style={"width":"4em"}),
        html.Label(" token_font_size:", style={"margin-left":"1em"}), dcc.Input(id="token-font-size", type="number", value=12, style={"width":"4em"}),
        html.Label(" label_font_size:", style={"margin-left":"1em"}), dcc.Input(id="label-font-size", type="number", value=20, style={"width":"4em"}),
    ], style={"margin-bottom":"0.5em"}),

    html.Div([
        dcc.Checklist(
            id="flags",
            options=[
                {"label":"include_input","value":"include_input"},
                {"label":"force_include_output","value":"force_include_output"},
                {"label":"include_subblocks","value":"include_subblocks"},
                {"label":"top_down","value":"top_down"},
                {"label":"verbose","value":"verbose"},
                {"label":"pad_to_max_length","value":"pad_to_max_length"},
                {"label":"Mean Top-K","value":"topk_mean"},            # ← new
            ],
            value=["include_input","force_include_output","topk_mean"],  # default includes it
            inline=True,
            inputStyle={"margin-right":"4px","margin-left":"12px"}
        )
    ], style={"margin-bottom":"1em"}),

    html.Div([
        html.Label("decoder_layer_names:"),
        dcc.Dropdown(
            id="decoder-layers",
            options=[{"label":nm,"value":nm} for nm in ["norm","lm_head"]],
            value=["norm","lm_head"],
            multi=True,
            style={"width":"300px"}
        ),
        html.Label("model_precision:", style={"margin-left":"2rem"}),
        dcc.Dropdown(
            id="model-precision",
            options=[
                {"label":"None","value":""},
                {"label":"float16","value":"float16"},
                {"label":"bfloat16","value":"bfloat16"},
                {"label":"float32","value":"float32"},
            ],
            value="",
            clearable=False,
            style={"width":"150px"}
        )
    ], style={"margin-bottom":"1em"}),

    html.Button("Plot", id="plot-btn", n_clicks=0),
    html.Hr(),
    dcc.Graph(id="logit-lens-graph", style={"height":"700px"}),
])

@app.callback(
    Output("logit-lens-graph", "figure"),
    [
      Input("plot-btn",        "n_clicks"),
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
    ]
)
def update_logit_lens(
    n, model_id, tok_id, text,
    start_ix, end_ix, top_k, metric,
    block_step, token_font_size, label_font_size,
    flags, decoder_layers, model_precision
):
    if not n or not text:
        return go.Figure()
    end_ix = start_ix + len(text.split())

    # translate flags list → booleans
    include_input        = "include_input" in flags
    force_include_output = "force_include_output" in flags
    include_subblocks    = "include_subblocks" in flags
    top_down             = "top_down" in flags
    verbose              = "verbose" in flags
    pad_to_max_length    = "pad_to_max_length" in flags
    topk_mean = "topk_mean" in flags

    # parse precision
    precision = None
    if model_precision:
        precision = getattr(torch, model_precision)

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
            model_precision      = precision
        ) or go.Figure()
    except Exception as e:
        return go.Figure().add_annotation(
            text=f"Error:\n{e}",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="red", size=14)
        )