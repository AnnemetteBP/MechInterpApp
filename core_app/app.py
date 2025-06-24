from dash import Dash, dcc, html, clientside_callback
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import os


#ASSETS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    #external_stylesheets=[dbc.themes.FLATLY],
    #assets_folder="assets",
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
import core_app.pages.attention_viewer

routes = {
    "Home":                                          "/",
    "TopK-N Logit Lens":              "/topk_logit_lens",
    "TopK-N Comparing Lens":      "/topk_comparing_lens",
    "SAE Saliency Heatmap":              "/sae_saliency",
    "SAE Comparison Heatmap":          "/sae_comparison",
    "PCA Token Embedding":  "/visualize_token_embedding",
    "PCA Neighbor Drift":    "/visualize_neighbor_drift",
    "PCA Sentence Drift":    "/visualize_sentence_drift",
    "Attention Viewer":              "/attention_viewer",
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
    "/attention_viewer":                 core_app.pages.attention_viewer.layout,
}

nav = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/", active="exact")),

        dbc.DropdownMenu(
            label="Logit Lens",
            nav=True,
            in_navbar=True,
            children=[
                dbc.DropdownMenuItem("TopK-N Logit Lens", href="/topk_logit_lens"),
                dbc.DropdownMenuItem("TopK-N Comparing Lens", href="/topk_comparing_lens"),
            ],
        ),

        dbc.DropdownMenu(
            label="SAE",
            nav=True,
            in_navbar=True,
            children=[
                dbc.DropdownMenuItem("SAE Saliency Heatmap", href="/sae_saliency"),
                dbc.DropdownMenuItem("SAE Comparison Heatmap", href="/sae_comparison"),
            ],
        ),

        dbc.DropdownMenu(
            label="PCA",
            nav=True,
            in_navbar=True,
            children=[
                dbc.DropdownMenuItem("PCA Token Embedding", href="/visualize_token_embedding"),
                dbc.DropdownMenuItem("PCA Neighbor Drift", href="/visualize_neighbor_drift"),
                dbc.DropdownMenuItem("PCA Sentence Drift", href="/visualize_sentence_drift"),
            ],
        ),

        dbc.NavItem(dbc.NavLink("Attention Viewer", href="/attention_viewer", active="exact")),
        dbc.NavItem(
            dbc.Switch(
                id="theme-switch",
                label="Light mode",
                value=False,
                style={"marginLeft": "1rem", "marginTop": "0.3rem"},
            )
        ),
    ],
    brand="🛠️ Mech. Interp. Toolkit",
    color="dark",
    dark=True,
    fluid=True,
    brand_href="/",
)

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    nav,
    html.Div(id="page-content", style={"padding": "2rem"})
])

clientside_callback(
    """
    function(switchOn) {
        document.documentElement.setAttribute('data-bs-theme', switchOn ? 'light' : 'dark');
        return window.dash_clientside.no_update;
    }
    """,
    Output("theme-switch", "id"),  # dummy output
    Input("theme-switch", "value")
)

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