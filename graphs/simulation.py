import math
import random
from collections import defaultdict

import networkx as nx
import numpy as np


def resolve_seed_nodes(G, seed_nodes):
    """Validate requested seed nodes and fall back to the first graph node when needed."""
    if not seed_nodes:
        return [next(iter(G.nodes()))], []

    resolved = []
    invalid = []
    for node in seed_nodes:
        if node in G:
            resolved.append(node)
        else:
            invalid.append(node)

    if not resolved:
        fallback = next(iter(G.nodes()), None)
        if fallback is None:
            return [], invalid
        return [fallback], invalid

    return resolved, invalid


def simulate_flood(G, seed_nodes, steps=5, spread_probability=0.7, random_seed=42):
    """Simulate simple flood propagation over the road network."""
    rng = random.Random(random_seed)
    resolved_nodes, invalid_nodes = resolve_seed_nodes(G, seed_nodes)
    flooded_nodes = set(resolved_nodes)
    flooded_edges = set()
    node_levels = {node: 0 for node in flooded_nodes}

    for step in range(steps):
        frontier = [node for node in flooded_nodes if node_levels[node] == step]
        if not frontier:
            break

        for node in frontier:
            for neighbor in G.neighbors(node):
                if neighbor in flooded_nodes:
                    continue
                if rng.random() <= spread_probability:
                    flooded_nodes.add(neighbor)
                    node_levels[neighbor] = step + 1
                    flooded_edges.add((min(node, neighbor), max(node, neighbor)))

    flooded_edges.update(
        {(min(u, v), max(u, v)) for u in flooded_nodes for v in G.neighbors(u) if v in flooded_nodes}
    )

    return {
        "flooded_nodes": sorted(flooded_nodes),
        "flooded_edges": sorted(flooded_edges),
        "steps": steps,
    }


def identify_choke_points(G, top_n=5):
    """Rank nodes by their structural importance for flood spread."""
    scores = []
    betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
    for node in G.nodes:
        degree = G.degree(node)
        if degree <= 1:
            continue
        score = (degree * 0.6) + (betweenness[node] * 2.0)
        scores.append((node, score))

    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[:top_n]


def build_flood_metrics(G, flooded_nodes, flooded_edges):
    """Summarize flood impact on the graph."""
    total_nodes = len(G.nodes)
    total_edges = len(G.edges)
    flooded_fraction_nodes = len(flooded_nodes) / max(total_nodes, 1)
    flooded_fraction_edges = len(flooded_edges) / max(total_edges, 1)
    return {
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "flooded_nodes": len(flooded_nodes),
        "flooded_edges": len(flooded_edges),
        "flooded_fraction_nodes": flooded_fraction_nodes,
        "flooded_fraction_edges": flooded_fraction_edges,
    }


def get_node_positions(G):
    """Return a deterministic position mapping for visualization."""
    pos = {}
    for node, data in G.nodes(data=True):
        x = data.get("x")
        y = data.get("y")
        if x is None or y is None:
            x, y = node, node
        pos[node] = (float(x), float(y))
    return pos
