from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.FLATLY],
)
server = app.server
app.title = "Interpretability Dashboard"

import core_app.pages.dashboard
import core_app.pages.method1

routes = {
    "Home":               "/",
    "TopK-N Logit Lens":  "/method1",
}

views = {
    "/":        core_app.pages.dashboard.layout,
    "/method1": core_app.pages.method1.layout,
}

nav = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink(label, href=path, active="exact"))
        for label, path in routes.items()
    ],
    brand="Mech. Interp.",
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