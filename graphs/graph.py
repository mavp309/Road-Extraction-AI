# import osmnx as ox
# import networkx as nx
# import random
# import pickle

# def get_real_but_broken_bengaluru_graph():
#     print("Fetching real road network data from Bengaluru via OSMnx API...")
#     box_lat, box_lon = 12.9716, 77.6412  # Indiranagar, BLR

#     G = ox.graph_from_point((box_lat, box_lon), dist=1000, network_type='drive')

#     try:
#         G = ox.convert.to_undirected(G)
#     except AttributeError:
#         G = ox.utils_graph.get_undirected(G)

#     for node, data in G.nodes(data=True):
#         data['pos'] = (data['x'], data['y'])

#     print(f"Base Graph Loaded: {len(G.nodes)} nodes, {len(G.edges)} edges.")

#     print("Executing scattered multi-zone destruction...")
#     G_broken = G.copy()

#     # 1. Random edge removal — knocks out 40% of edges across the whole map
#     destruction_rate = 0.40
#     edges_to_remove = random.sample(list(G_broken.edges()), int(len(G_broken.edges()) * destruction_rate))
#     G_broken.remove_edges_from(edges_to_remove)

#     # 2. A few small localized node cluster removals to punch hard holes
#     all_nodes = list(G_broken.nodes())
#     for _ in range(3):  # 3 small blast zones
#         if not all_nodes:
#             break
#         epicenter = random.choice(all_nodes)
#         ep_x = G_broken.nodes[epicenter]['x']
#         ep_y = G_broken.nodes[epicenter]['y']
#         # Small radius — just punches a tight hole, not a giant wipeout
#         radius = 0.003  # ~300m in degrees lon/lat
#         to_remove = [
#             n for n, d in G_broken.nodes(data=True)
#             if ((d['x'] - ep_x)**2 + (d['y'] - ep_y)**2)**0.5 < radius
#         ]
#         G_broken.remove_nodes_from(to_remove)
#         all_nodes = [n for n in all_nodes if n not in to_remove]

#     # 3. Drop fully isolated nodes (degree 0) — keeps the graph clean
#     isolated = [n for n, deg in G_broken.degree() if deg == 0]
#     G_broken.remove_nodes_from(isolated)

#     print(f"Destruction complete: {len(G_broken.nodes)} nodes, {len(G_broken.edges)} edges remaining.")
#     print(f"Connected components created: {nx.number_connected_components(G_broken)}")
#     return G, G_broken

# if __name__ == "__main__":
#     G_ground_truth, G_broken_mock = get_real_but_broken_bengaluru_graph()

#     with open("mock_bengaluru_ground_truth.gpickle", "wb") as f:
#         pickle.dump(G_ground_truth, f)

#     with open("mock_bengaluru_broken.gpickle", "wb") as f:
#         pickle.dump(G_broken_mock, f)

#     print("Saved graph files!")

import osmnx as ox
import networkx as nx
import random
import pickle
import cv2
import os
import numpy as np
from pathlib import Path
import sys
sys.path.append(os.path.abspath(os.path.join('.', '..')))

REPO_ROOT = Path(__file__).resolve().parent
SAVED_MASK_PATH = REPO_ROOT / "frontend" / "saved_mask.png"


def load_saved_mask(mask_path=SAVED_MASK_PATH):
    mask_path = Path(mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(f"Saved mask not found at {mask_path}")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Unable to read saved mask from {mask_path}")

    return (mask > 128).astype(np.uint8)


def skeletonize_mask(mask):
    mask_bin = (mask > 0).astype(np.uint8) * 255
    skel = np.zeros(mask.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(mask_bin, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(mask_bin, temp)
        skel = cv2.bitwise_or(skel, temp)
        mask_bin = eroded.copy()

        if cv2.countNonZero(mask_bin) == 0:
            break

    return (skel > 0).astype(np.uint8)


def mask_to_graph(mask, connectivity=8):
    G = nx.Graph()
    rows, cols = mask.shape
    node_map = {}

    for y in range(rows):
        for x in range(cols):
            if mask[y, x] > 0:
                node_id = y * cols + x
                node_map[(y, x)] = node_id
                G.add_node(node_id, x=float(x), y=float(y))

    if connectivity == 8:
        neighbor_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    else:
        neighbor_offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    for (y, x), node_id in node_map.items():
        for dy, dx in neighbor_offsets:
            neighbor = (y + dy, x + dx)
            if neighbor in node_map:
                neighbor_id = node_map[neighbor]
                if not G.has_edge(node_id, neighbor_id):
                    distance = float(np.hypot(dx, dy))
                    G.add_edge(node_id, neighbor_id, weight=distance, length=distance)

    return G


def build_graph_from_saved_mask(mask_path=SAVED_MASK_PATH):
    mask = load_saved_mask(mask_path)
    skeleton = skeletonize_mask(mask)
    G = mask_to_graph(skeleton, connectivity=8)
    print(f"Loaded saved mask from {mask_path}: {len(G.nodes)} skeleton nodes, {len(G.edges)} edges")
    return G

def get_real_but_broken_bengaluru_graph():
    print("Fetching real road network data from Bengaluru via OSMnx API...")
    box_lat, box_lon = 12.9716, 77.6412  # Indiranagar, BLR

    G = ox.graph_from_point((box_lat, box_lon), dist=1000, network_type="drive")

    try:
        G = ox.convert.to_undirected(G)
    except AttributeError:
        G = ox.utils_graph.get_undirected(G)

    for node, data in G.nodes(data=True):
        data["pos"] = (data["x"], data["y"])

    print(f"Base Graph Loaded: {len(G.nodes)} nodes, {len(G.edges)} edges.")

    print("Executing random edge destruction...")
    G_broken = G.copy()

    # Randomly remove 40% of edges
    destruction_rate = 0.40
    edges = list(G_broken.edges())
    edges_to_remove = random.sample(edges, int(len(edges) * destruction_rate))
    G_broken.remove_edges_from(edges_to_remove)

    print(f"Destruction complete: {len(G_broken.nodes)} nodes, {len(G_broken.edges)} edges remaining.")
    print(f"Connected components created: {nx.number_connected_components(G_broken)}")

    return G, G_broken


if __name__ == "__main__":
    G_ground_truth, G_broken_mock = get_real_but_broken_bengaluru_graph()

    with open("mock_bengaluru_ground_truth.gpickle", "wb") as f:
        pickle.dump(G_ground_truth, f)

    with open("mock_bengaluru_broken.gpickle", "wb") as f:
        pickle.dump(G_broken_mock, f)

    print("Saved graph files!")