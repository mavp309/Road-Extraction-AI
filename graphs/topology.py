"""
topology.py

Converts a skeleton image into a proper road graph.

Instead of creating a node for every skeleton pixel,
only intersections and dead-ends become graph nodes.

Roads become graph edges with stored geometry.
"""

import networkx as nx
import numpy as np

from .mask_processing import (
    neighbours,
    detect_keypoints,
    OFFSETS_8,
)


# -------------------------------------------------------
# Node indexing
# -------------------------------------------------------

def _cluster_adjacent_pixels(pixels):
    """
    Group junction/dead-end pixels that are 8-connected neighbours of each
    other into a single logical node.

    Even with a clean 1-pixel skeleton, a real intersection commonly shows
    up as a small tight cluster of 2-4 adjacent pixels that each independently
    satisfy degree >= 3 (e.g. a 4-way crossing has a couple of pixels each
    touching several branches). Without this clustering step, every one of
    those pixels becomes its own graph node -- splitting a single real
    intersection into several near-duplicate nodes sitting almost on top of
    each other. This is a second, independent source of node bloat on top of
    the skeletonization issue.
    """
    pixels = set(pixels)
    visited = set()
    clusters = []

    for start in pixels:
        if start in visited:
            continue

        stack = [start]
        visited.add(start)
        cluster = []

        while stack:
            cur = stack.pop()
            cluster.append(cur)

            cy, cx = cur
            for dy, dx in OFFSETS_8:
                neighbour = (cy + dy, cx + dx)
                if neighbour in pixels and neighbour not in visited:
                    visited.add(neighbour)
                    stack.append(neighbour)

        clusters.append(cluster)

    return clusters


def build_node_lookup(keypoints):

    clusters = _cluster_adjacent_pixels(keypoints["junctions"])

    lookup = {}
    node_positions = {}

    for node_id, cluster in enumerate(clusters):

        mean_y = sum(p[0] for p in cluster) / len(cluster)
        mean_x = sum(p[1] for p in cluster) / len(cluster)

        node_positions[node_id] = (mean_x, mean_y)

        for pixel in cluster:
            lookup[pixel] = node_id

    return lookup, node_positions


# -------------------------------------------------------
# Trace a road
# -------------------------------------------------------

def trace_segment(
    skeleton,
    start_pixel,
    first_pixel,
    junction_pixels,
):

    geometry = [start_pixel]

    prev = start_pixel

    current = first_pixel

    length = 0.0

    while True:

        geometry.append(current)

        length += np.hypot(
            current[0] - prev[0],
            current[1] - prev[1],
        )

        if current in junction_pixels:

            return current, geometry, length

        nbrs = neighbours(
            skeleton,
            current[0],
            current[1],
        )

        nbrs = [p for p in nbrs if p != prev]

        if len(nbrs) == 0:

            return None, geometry, length

        prev = current

        current = nbrs[0]


# -------------------------------------------------------
# Build graph
# -------------------------------------------------------

def build_topological_graph(skeleton):

    keypoints = detect_keypoints(skeleton)

    lookup, positions = build_node_lookup(keypoints)

    junction_pixels = set(keypoints["junctions"])

    G = nx.Graph()

    for node_id, (x, y) in positions.items():

        G.add_node(

            node_id,

            x=float(x),

            y=float(y),

            pos=(float(x), float(y)),

        )

    visited = set()

    for pixel in junction_pixels:

        start_node = lookup[pixel]

        nbrs = neighbours(

            skeleton,

            pixel[0],

            pixel[1],

        )

        for n in nbrs:

            if (pixel, n) in visited:

                continue

            end_pixel, geometry, length = trace_segment(

                skeleton,

                pixel,

                n,

                junction_pixels,

            )

            for i in range(len(geometry) - 1):

                visited.add(

                    (geometry[i], geometry[i + 1])

                )

                visited.add(

                    (geometry[i + 1], geometry[i])

                )

            if end_pixel is None:

                continue

            end_node = lookup[end_pixel]

            if start_node == end_node:

                continue

            if G.has_edge(start_node, end_node):

                continue

            geometry_xy = [

                (p[1], p[0])

                for p in geometry

            ]

            G.add_edge(

                start_node,

                end_node,

                weight=length,

                length=length,

                geometry=geometry_xy,

            )

    return G


# -------------------------------------------------------
# Cleanup
# -------------------------------------------------------

def prune_spurs(G, min_length=15.0):
    """
    Remove short dead-end branches.

    Segmentation masks almost always have jagged/noisy boundaries, and those
    show up in the skeleton as short spurs hanging off the real road network.
    This repeatedly strips dead-end edges shorter than min_length (in the
    same units as your edge "length" attribute, i.e. pixels) until none
    remain. Real short roads/cul-de-sacs are typically much longer than mask
    noise, so a modest threshold (start around 10-20px and tune to your
    image resolution) is usually safe.
    """
    G = G.copy()

    changed = True
    while changed:
        changed = False
        dead_ends = [n for n in G.nodes if G.degree(n) == 1]

        for node in dead_ends:
            if node not in G:
                continue
            neighbours_ = list(G.neighbors(node))
            if not neighbours_:
                continue
            other = neighbours_[0]
            length = G.edges[node, other].get("length", 0.0)

            if length < min_length:
                G.remove_node(node)
                changed = True

    return G