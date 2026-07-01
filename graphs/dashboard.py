"""
dashboard.py

Interactive Plotly GIS viewer.

Completely replaces matplotlib visualization.
"""

import plotly.graph_objects as go
import numpy as np
from PIL import Image as PILImage


# ---------------------------------------------------------
# Build road traces
# ---------------------------------------------------------

def road_trace(
    G,
    flooded_edges=None,
    healed_edges=None,
):

    flooded_edges = flooded_edges or set()
    healed_edges = healed_edges or set()

    traces = []

    for u, v, data in G.edges(data=True):

        geometry = data.get("geometry")
        if geometry:
            xs = [p[0] for p in geometry]
            ys = [p[1] for p in geometry]
        else:
            ux = G.nodes[u]["x"]
            uy = G.nodes[u]["y"]
            vx = G.nodes[v]["x"]
            vy = G.nodes[v]["y"]
            xs = [ux, vx]
            ys = [uy, vy]

        edge = tuple(sorted((u, v)))

        color = "#BBBBBB"
        width = 2

        if edge in healed_edges:
            color = "#00AAFF"
            width = 3

        if edge in flooded_edges:
            color = "#FF3333"
            width = 4

        traces.append(

            go.Scatter(
                x=xs,
                y=ys,

                mode="lines",

                line=dict(
                    color=color,
                    width=width,
                ),

                hoverinfo="skip",

                showlegend=False,

            )

        )

    return traces

def node_trace(
    G,
    flooded_nodes=None,
    choke_nodes=None,
    seed_nodes=None,
):

    flooded_nodes = set(flooded_nodes or [])
    choke_nodes = set(choke_nodes or [])
    seed_nodes = set(seed_nodes or [])

    groups = {

        "Normal": {
            "x": [],
            "y": [],
            "color": "white",
            "size": 6,
        },

        "Flooded": {
            "x": [],
            "y": [],
            "color": "red",
            "size": 10,
        },

        "Choke": {
            "x": [],
            "y": [],
            "color": "orange",
            "size": 14,
        },

        "Seed": {
            "x": [],
            "y": [],
            "color": "lime",
            "size": 16,
        },

    }

    hover = []

    for node, data in G.nodes(data=True):

        x = data["x"]
        y = data["y"]

        hover_text = (
            f"Node: {node}"
            f"<br>Degree: {G.degree(node)}"
        )

        if node in seed_nodes:

            g = groups["Seed"]

        elif node in choke_nodes:

            g = groups["Choke"]

        elif node in flooded_nodes:

            g = groups["Flooded"]

        else:

            g = groups["Normal"]

        g["x"].append(x)
        g["y"].append(y)

        hover.append(hover_text)

    traces = []

    for name, g in groups.items():

        traces.append(

            go.Scatter(

                x=g["x"],

                y=g["y"],

                mode="markers",

                name=name,

                marker=dict(

                    color=g["color"],

                    size=g["size"],

                    line=dict(width=1,color="black")

                ),

                hovertemplate="%{text}<extra></extra>",

                text=hover,

            )

        )

    return traces

def _normalize_layout_image(image):
    if isinstance(image, np.ndarray):
        return PILImage.fromarray(image)
    return image


def add_background(fig, image):

    image = _normalize_layout_image(image)
    h, w = image.shape[:2] if isinstance(image, np.ndarray) else image.size[::-1]

    fig.add_layout_image(

        dict(

            source=image,

            x=0,

            y=0,

            sizex=w,

            sizey=h,

            xref="x",

            yref="y",

            sizing="stretch",

            layer="below",

        )

    )

def create_dashboard(
    G,
    image=None,
    flooded_nodes=None,
    flooded_edges=None,
    choke_nodes=None,
    healed_edges=None,
    seed_nodes=None,
):

    fig = go.Figure()

    if image is not None:

        add_background(fig,image)

    for t in road_trace(
        G,
        flooded_edges,
        healed_edges,
    ):
        fig.add_trace(t)

    for t in node_trace(
        G,
        flooded_nodes,
        choke_nodes,
        seed_nodes,
    ):
        fig.add_trace(t)

    fig.update_layout(

        template="plotly_dark",

        dragmode="pan",

        hovermode="closest",

        height=850,

        showlegend=True,

        margin=dict(
            l=0,
            r=0,
            b=0,
            t=35,
        ),

    )

    fig.update_yaxes(

        scaleanchor="x",
        autorange="reversed",
        visible=False,

    )

    fig.update_xaxes(
        visible=False,
    )

    return fig