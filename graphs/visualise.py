"""
visualize.py

Visualization utilities.

Unlike NetworkX, this draws the actual road geometry
stored inside each edge.

Supports

- satellite overlay
- flood highlighting
- choke points
- seed nodes
- healed roads
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# --------------------------------------------------------
# Background
# --------------------------------------------------------

def draw_background(ax, image=None):

    if image is None:
        ax.set_facecolor("#111111")
        return

    ax.imshow(
        image,
        origin="upper",
    )


# --------------------------------------------------------
# Draw roads
# --------------------------------------------------------

def draw_roads(

    ax,
    G,

    flooded_edges=None,
    healed_edges=None,

    road_color="#BBBBBB",
    flood_color="#FF3333",
    healed_color="#00AAFF",

    linewidth=2,

):

    flooded_edges = flooded_edges or set()

    healed_edges = healed_edges or set()

    for u, v, data in G.edges(data=True):

        xs = [p[0] for p in data["geometry"]]
        ys = [p[1] for p in data["geometry"]]

        edge = tuple(sorted((u, v)))

        color = road_color

        if edge in healed_edges:

            color = healed_color

        if edge in flooded_edges:

            color = flood_color

        ax.plot(

            xs,
            ys,

            color=color,

            linewidth=linewidth,

            solid_capstyle="round",

            zorder=2,

        )


# --------------------------------------------------------
# Draw nodes
# --------------------------------------------------------

def draw_nodes(

    ax,
    G,

    flooded_nodes=None,

    choke_nodes=None,

    seed_nodes=None,

):

    flooded_nodes = set(flooded_nodes or [])

    choke_nodes = set(choke_nodes or [])

    seed_nodes = set(seed_nodes or [])

    normal_x = []
    normal_y = []

    flood_x = []
    flood_y = []

    choke_x = []
    choke_y = []

    seed_x = []
    seed_y = []

    for node, data in G.nodes(data=True):

        x = data["x"]
        y = data["y"]

        if node in seed_nodes:

            seed_x.append(x)
            seed_y.append(y)

        elif node in choke_nodes:

            choke_x.append(x)
            choke_y.append(y)

        elif node in flooded_nodes:

            flood_x.append(x)
            flood_y.append(y)

        else:

            normal_x.append(x)
            normal_y.append(y)

    ax.scatter(

        normal_x,
        normal_y,

        s=12,

        color="white",

        alpha=0.5,

        zorder=5,

    )

    ax.scatter(

        flood_x,
        flood_y,

        s=25,

        color="red",

        zorder=8,

    )

    ax.scatter(

        choke_x,
        choke_y,

        s=60,

        color="orange",

        edgecolors="black",

        linewidths=0.5,

        zorder=9,

    )

    ax.scatter(

        seed_x,
        seed_y,

        s=80,

        color="lime",

        edgecolors="black",

        linewidths=0.5,

        zorder=10,

    )


# --------------------------------------------------------
# Labels
# --------------------------------------------------------

def annotate_nodes(

    ax,
    G,
    nodes,

):

    for node in nodes:

        if node not in G:

            continue

        x = G.nodes[node]["x"]
        y = G.nodes[node]["y"]

        ax.text(

            x,

            y,

            str(node),

            fontsize=8,

            color="yellow",

        )


# --------------------------------------------------------
# Main Viewer
# --------------------------------------------------------

def show_graph(

    G,

    background=None,

    flooded_nodes=None,

    flooded_edges=None,

    choke_nodes=None,

    healed_edges=None,

    seed_nodes=None,

    figsize=(12,12),

):

    fig, ax = plt.subplots(figsize=figsize)

    draw_background(ax, background)

    draw_roads(

        ax,

        G,

        flooded_edges=flooded_edges,

        healed_edges=healed_edges,

    )

    draw_nodes(

        ax,

        G,

        flooded_nodes,

        choke_nodes,

        seed_nodes,

    )

    ax.set_xticks([])

    ax.set_yticks([])

    ax.set_aspect("equal")

    plt.tight_layout()

    plt.show()