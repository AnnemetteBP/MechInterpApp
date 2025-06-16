from dash import html, dcc, Input, Output
import plotly.graph_objects as go
from core_app.app import app
from ..tools.dictionary_learning import heatmap_comparing_plotting

layout = html.Div([
    html.H2("The SAE Comparing Heatmap"),
    dcc.Textarea(id="input-text-1", style={"width":"100%","height":100}),
    dcc.Input(id="start-ix-1", type="number", value=0),
    dcc.Input(id="top-k-1",   type="number", value=5),
    html.Button("Plot", id="plot-btn-1"),
    dcc.Graph(id="graph-1", style={"height":"600px"}),
])

@app.callback(
    Output("graph-1", "figure"),
    Input("plot-btn-1", "n_clicks"),
    Input("input-text-1", "value"),
    Input("start-ix-1",    "value"),
    Input("top-k-1",       "value"),
)
def update_graph1(n_clicks, text, start_ix, top_k):
    if not n_clicks:
        return go.Figure()
    #metrics = collect_metrics_method1(inputs=text, start_ix=start_ix, topk=top_k)
    #return make_figure_method1(**metrics, start_ix=start_ix, top_k=top_k)