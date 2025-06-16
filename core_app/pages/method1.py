from dash import html, dcc, Input, Output
import plotly.graph_objects as go
from core_app.app import app

# import the module (so we call module.function)
from ..tools.logit_lens import topk_lens_plotter

layout = html.Div([
    html.H2("The TopK-N Logit Lens"),
    dcc.Input(id="model-id",     placeholder="model (e.g. LLaMA / OLMo)", value="NousResearch/DeepHermes-3-Llama-3-3B-Preview"),
    dcc.Input(id="tokenizer-id", placeholder="tokenizer",       value="NousResearch/DeepHermes-3-Llama-3-3B-Preview"),
    dcc.Textarea(id="input-text", placeholder="Your prompt here: e.g., What is y if y=2*2-4+(3*2)", style={"width":"100%","height":80}),
    html.Div([
        "start_ix:", dcc.Input(id="start-ix", type="number", value=0, style={"width":"4em"}),
        " top_k:",   dcc.Input(id="top-k",    type="number", value=5, style={"width":"4em"}),
    ], style={"margin":"1em 0"}),
    dcc.Dropdown(
        id="metric-type",
        options=[
            {"label":"Logits","value":"logits"},
            {"label":"Probs","value":"probs"},
            {"label":"Entropy","value":"entropy"},
            {"label":"KL Divergence","value":"kl"},
            {"label":"Cosine","value":"cosine_sims"},
            {"label":"KL Layer-wise","value":"kl_layerwise"},
            {"label":"Token Variety","value":"token_variety"},
            {"label":"Ranks","value":"ranks"},
        ],
        value="logits", clearable=False, style={"width":"200px"}
    ),
    html.Button("Plot", id="plot-btn", n_clicks=0),
    dcc.Graph(id="logit-lens-graph", style={"height":"600px"}),
])

@app.callback(
    Output("logit-lens-graph", "figure"),
    Input("plot-btn",      "n_clicks"),
    Input("model-id",      "value"),
    Input("tokenizer-id",  "value"),
    Input("input-text",    "value"),
    Input("start-ix",      "value"),
    Input("top-k",         "value"),
    Input("metric-type",   "value"),
)
def update_logit_lens(n, model_id, tok_id, text, start_ix, top_k, metric):
    if not n or not text:
        return go.Figure()
    end_ix = start_ix + len(text.split())

    try:
        # **One** call to the public plotter:
        return topk_lens_plotter.plot_topk_logit_lens(
            model_path      = model_id,
            tokenizer_path  = tok_id,
            inputs          = text,
            start_ix        = start_ix,
            end_ix          = end_ix,
            topk            = top_k,
            # flags from metric dropdown:
            entropy         = (metric=="entropy"),
            probs           = (metric=="probs"),
            kl              = (metric=="kl"),
            cosine_sims     = (metric=="cosine_sims"),
            kl_layerwise    = (metric=="kl_layerwise"),
            token_variety   = (metric=="token_variety"),
            ranks           = (metric=="ranks"),
        ) or go.Figure()
    except Exception as e:
        return go.Figure().add_annotation(
            text=f"Error:\n{e}",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="red", size=14)
        )