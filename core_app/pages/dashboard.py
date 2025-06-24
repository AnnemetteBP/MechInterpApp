from dash import html
import dash_bootstrap_components as dbc

layout = html.Div(
    style={
        "height": "100vh",
        "width": "100vw",
        "backgroundImage": "url('assets/background.jpg')",
        "backgroundSize": "cover",
        "backgroundPosition": "center",
        "backgroundRepeat": "no-repeat",
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "center",
    },
    children=dbc.Container(
        dbc.Card(
            dbc.CardBody([
                html.H1("Welcome to the Mech. Interp. Dashboard!", className="text-white"),
                html.P("Explore Logit Lenses, SAE heatmaps, PCA visualizations, and more.", className="lead text-white"),
                dbc.Button("Get Started", href="/topk_logit_lens", color="primary", size="lg")
            ]),
            style={"backgroundColor": "rgba(0, 0, 0, 0.6)", "padding": "2rem"},
            className="text-center"
        ),
        fluid=True
    )
)

