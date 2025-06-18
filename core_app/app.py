from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.FLATLY],
)
server = app.server
app.title = "Mech Interp Dashboard"

import core_app.pages.dashboard
import core_app.pages.topk_logit_lens
import core_app.pages.topk_comparing_lens
import core_app.pages.sae_saliency
import core_app.pages.sae_comparison
import core_app.pages.token_embeddings
import core_app.pages.neighbor_drift
import core_app.pages.sentence_embedding_drift

routes = {
    "Home":                                          "/",
    "TopK-N Logit Lens":              "/topk_logit_lens",
    "TopK-N Comparing Lens":      "/topk_comparing_lens",
    "SAE Saliency Heatmap":              "/sae_saliency",
    "SAE Comparison Heatmap":          "/sae_comparison",
    "PCA Token Embedding":  "/visualize_token_embedding",
    "PCA Neighbor Drift":    "/visualize_neighbor_drift",
    "PCA Sentence Drift":    "/visualize_sentence_drift",
}

views = {
    "/":                                        core_app.pages.dashboard.layout,
    "/topk_logit_lens":                   core_app.pages.topk_logit_lens.layout,
    "/topk_comparing_lens":           core_app.pages.topk_comparing_lens.layout,
    "/sae_saliency":                         core_app.pages.sae_saliency.layout,
    "/sae_comparison":                     core_app.pages.sae_comparison.layout,
    "/visualize_token_embedding":        core_app.pages.token_embeddings.layout,
    "/visualize_neighbor_drift":           core_app.pages.neighbor_drift.layout,
    "/visualize_sentence_drift": core_app.pages.sentence_embedding_drift.layout,
}

nav = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink(label, href=path, active="exact"))
        for label, path in routes.items()
    ],
    brand="🛠️ Mech. Interp. Toolkit",
    color="dark",
    dark=True,
    fluid=True,
)

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    nav,
    html.Div(id="page-content", style={"padding": "2rem"})
])


@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def render_page(pathname):
    path = (pathname or "/").rstrip("/") or "/"
    print("render_page called with:", repr(path))
    print("available views:", list(views.keys()))
    # Return the matched layout or a 404 message
    return views.get(
        path,
        html.H1(f"404: No view for {path}", className="text-danger")
    )