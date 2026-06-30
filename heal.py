import pickle
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import numpy as np


# ─────────────────────────────────────────────
# 1. GRAPH LOADING
# ─────────────────────────────────────────────

def load_and_project_graph(filepath):
    with open(filepath, "rb") as f:
        G = pickle.load(f)
    print(f"Projecting {filepath} to UTM (meters)...")
    G_proj = ox.project_graph(G)
    if G_proj.is_directed():
        G_proj = G_proj.to_undirected()
    for node, data in G_proj.nodes(data=True):
        data['pos'] = (data['x'], data['y'])
    return G_proj


# ─────────────────────────────────────────────
# 2. BEARING ESTIMATION
# ─────────────────────────────────────────────

def get_endpoint_bearing(G, node):
    """
    Estimates the outward-facing direction at a node — i.e. the direction
    the road is heading *away* from the graph, toward the gap.

    - Degree-1 (dead end): vector points away from the single neighbour.
    - Degree-2+           : uses the two nearest neighbours to estimate
                            local road axis, then averages outward vectors.

    Returns a unit vector (dx, dy), or None if it can't be computed.
    """
    neighbors = list(G.neighbors(node))
    if not neighbors:
        return None

    nx_ = G.nodes[node]['x']
    ny_ = G.nodes[node]['y']

    if len(neighbors) == 1:
        nbx = G.nodes[neighbors[0]]['x']
        nby = G.nodes[neighbors[0]]['y']
        # Outward = away from the single neighbour (extrapolate forward)
        vec = np.array([nx_ - nbx, ny_ - nby])
    else:
        # Sort neighbours by proximity, use two closest for local road axis
        dists = sorted(
            neighbors,
            key=lambda nb: np.hypot(G.nodes[nb]['x'] - nx_, G.nodes[nb]['y'] - ny_)
        )
        nb1, nb2 = dists[0], dists[1]
        v1 = np.array([nx_ - G.nodes[nb1]['x'], ny_ - G.nodes[nb1]['y']])
        v2 = np.array([nx_ - G.nodes[nb2]['x'], ny_ - G.nodes[nb2]['y']])
        for v in [v1, v2]:
            n = np.linalg.norm(v)
            if n > 0:
                v /= n
        vec = v1 + v2

    norm = np.linalg.norm(vec)
    if norm == 0:
        return None
    return vec / norm


# ─────────────────────────────────────────────
# 3. ALIGNMENT SCORING
# ─────────────────────────────────────────────

def angular_alignment_score(G, u, v):
    """
    Measures how naturally a proposed bridge (u ↔ v) continues the
    existing road geometry at both endpoints.

    Returns a score in [0, 1]:
      1.0 = bridge continues the road in a straight line
      0.5 = perpendicular (T-junction)
      0.0 = bridge doubles back against the road direction
    """
    bearing_u = get_endpoint_bearing(G, u)
    bearing_v = get_endpoint_bearing(G, v)

    ux, uy = G.nodes[u]['x'], G.nodes[u]['y']
    vx, vy = G.nodes[v]['x'], G.nodes[v]['y']

    bridge_vec = np.array([vx - ux, vy - uy])
    norm = np.linalg.norm(bridge_vec)
    if norm == 0:
        return 0.0
    bridge_unit = bridge_vec / norm

    scores = []
    if bearing_u is not None:
        scores.append(abs(np.dot(bearing_u, bridge_unit)))
    if bearing_v is not None:
        # Flip bridge vector for v's perspective (v looks toward u)
        scores.append(abs(np.dot(bearing_v, -bridge_unit)))

    if not scores:
        return 0.5

    return float(np.mean(scores))


# ─────────────────────────────────────────────
# 3b. INTERIOR EDGE RECOVERY (Phase 3)
# ─────────────────────────────────────────────

def recover_interior_edges(G_healed, max_dist=175, alignment_thresh=0.50):
    """
    Recovers removed edges where BOTH endpoints were high-degree nodes
    (intersections) in the broken graph — these leave no dead-ends behind,
    so Phase 2's dead-end sweep can never find them.

    Strategy: look at degree-2 (pass-through) nodes. A genuine pass-through
    node has two neighbours that continue the road in a straight line.
    If a degree-2 node has a *better*-aligned, similarly-close candidate
    it isn't yet connected to, that's a signal an edge was severed between
    two such points along the same road.

    This only connects nodes that are NOT already neighbours, so it can't
    duplicate existing edges, and the degree-2 restriction keeps it from
    touching real intersections.
    """
    candidates = []
    deg2_nodes = [n for n in G_healed.nodes() if G_healed.degree(n) == 2]

    if len(deg2_nodes) < 2:
        return G_healed, 0

    coords = np.array([
        [G_healed.nodes[n]['x'], G_healed.nodes[n]['y']]
        for n in deg2_nodes
    ])
    tree = cKDTree(coords)
    k = min(10, len(deg2_nodes))
    dists_arr, idx_arr = tree.query(coords, k=k)

    for i, node_u in enumerate(deg2_nodes):
        nb_u = set(G_healed.neighbors(node_u))

        for j_idx, dist in zip(idx_arr[i], dists_arr[i]):
            if dist == 0 or dist > max_dist:
                continue
            node_v = deg2_nodes[j_idx]
            if node_v == node_u or node_v in nb_u:
                continue
            if G_healed.has_edge(node_u, node_v):
                continue

            alignment = angular_alignment_score(G_healed, node_u, node_v)
            if alignment < alignment_thresh:
                continue

            key = (min(node_u, node_v), max(node_u, node_v))
            candidates.append((key[0], key[1], dist, alignment))

    # Deduplicate: keep best alignment per node pair
    best = {}
    for u, v, d, a in candidates:
        if (u, v) not in best or a > best[(u, v)][1]:
            best[(u, v)] = (d, a)
    candidates = [(u, v, d, a) for (u, v), (d, a) in best.items()]
    candidates.sort(key=lambda x: x[2] / (x[3] + 1e-6))

    # Each node gets at most one new interior connection per call —
    # prevents a single degree-2 node from sprouting multiple new edges
    # and turning into an artificial intersection
    used = set()
    added = 0
    for u, v, dist, alignment in candidates:
        if G_healed.has_edge(u, v):
            continue
        if u in used or v in used:
            continue
        G_healed.add_edge(u, v, weight=dist, length=dist,
                          healed=True, alignment=alignment)
        used.add(u)
        used.add(v)
        added += 1

    return G_healed, added


# ─────────────────────────────────────────────
# 4. HEALING ALGORITHM
# ─────────────────────────────────────────────

def heal_topological_gaps(G, max_dist_meters=400, min_alignment=0.30, max_passes=5):
    """
    Two-phase healing:

    Phase 1 — Structural connectivity
        Connects isolated components via geometrically plausible bridges.
        Only considers degree <= 2 nodes (dead-ends / pass-through) as
        bridge endpoints — real occlusion gaps always terminate there.

    Phase 2 — Dead-end resolution (intra-region)
        Sweep 1 : dead-end -> dead-end, strict alignment (0.55), tight dist cap.
                  Catches the easy straight gaps first.
        Sweep 2+: dead-end -> any node (degree < 4), relaxed alignment (0.40).
                  Catches orphaned dead-ends whose natural partner was already
                  consumed as a Phase 1 component bridge (now degree >= 2).
    """
    G_healed = G.copy()
    total_healed = 0

    # ── PHASE 1: Structural connectivity ─────────────────────────────
    for pass_num in range(1, max_passes + 1):
        components = list(nx.connected_components(G_healed))
        n_comp = len(components)

        if n_comp <= 1:
            print(f"Fully connected after pass {pass_num - 1}.")
            break

        print(f"\n── Pass {pass_num} ── {n_comp} components remaining")

        node_list = []
        comp_label = {}
        for idx, comp in enumerate(components):
            for node in comp:
                node_list.append(node)
                comp_label[node] = idx

        coords = np.array([
            [G_healed.nodes[n]['x'], G_healed.nodes[n]['y']]
            for n in node_list
        ])

        # Scale k with graph size — don't cap at 50 which misses cross-component candidates
        k = min(200, len(node_list))
        tree = cKDTree(coords)
        dists, indices = tree.query(coords, k=k)

        candidate_edges = []
        rejected_angle = 0
        rejected_dist  = 0

        for i, node_u in enumerate(node_list):
            comp_u = comp_label[node_u]

            # Only bridge FROM dead-ends / pass-through nodes
            # Real occlusion gaps always terminate at degree <= 2
            if G_healed.degree(node_u) > 2:
                continue

            for j_idx, dist in zip(indices[i], dists[i]):
                if dist == 0:
                    continue
                if dist > max_dist_meters:
                    rejected_dist += 1
                    continue

                node_v = node_list[j_idx]
                if comp_label[node_v] == comp_u:
                    continue

                # Target must also be a dead-end / pass-through
                if G_healed.degree(node_v) > 2:
                    continue

                alignment = angular_alignment_score(G_healed, node_u, node_v)
                if alignment < min_alignment:
                    rejected_angle += 1
                    continue

                key = (min(node_u, node_v), max(node_u, node_v))
                candidate_edges.append((key[0], key[1], dist, alignment))

        # Deduplicate: keep best alignment per node pair
        best = {}
        for u, v, d, a in candidate_edges:
            if (u, v) not in best or a > best[(u, v)][1]:
                best[(u, v)] = (d, a)
        candidate_edges = [(u, v, d, a) for (u, v), (d, a) in best.items()]

        # Sort: short + well-aligned first
        candidate_edges.sort(key=lambda x: x[2] / (x[3] + 1e-6))

        print(f"   Candidates : {len(candidate_edges)}  |  "
              f"Rejected (dist) : {rejected_dist}  |  "
              f"Rejected (angle): {rejected_angle}")

        healed_this_pass = 0
        for u, v, dist, alignment in candidate_edges:
            if not nx.has_path(G_healed, u, v):
                G_healed.add_edge(u, v, weight=dist, length=dist,
                                  healed=True, alignment=alignment)
                healed_this_pass += 1

        total_healed += healed_this_pass
        print(f"   Bridged    : {healed_this_pass} gaps this pass")

        if healed_this_pass == 0:
            print("   No new bridges possible — stopping early.")
            break

    # ── PHASE 2: Dead-end resolution ─────────────────────────────────
    print("\n── Phase 2: Intra-region dead-end recovery ──")

    # Diagnostic: show dead-end distance distribution to validate cap choice
    dead_ends_diag = [n for n in G_healed.nodes() if G_healed.degree(n) == 1]
    print(f"   Dead-ends before Phase 2: {len(dead_ends_diag)}")
    if len(dead_ends_diag) >= 2:
        de_coords_diag = np.array([
            [G_healed.nodes[n]['x'], G_healed.nodes[n]['y']]
            for n in dead_ends_diag
        ])
        diag_tree = cKDTree(de_coords_diag)
        diag_dists, _ = diag_tree.query(de_coords_diag, k=2)
        nearest = diag_dists[:, 1]
        print(f"   Dead-end NN distances ->  "
              f"min={nearest.min():.1f}m  "
              f"median={np.median(nearest):.1f}m  "
              f"p90={np.percentile(nearest, 90):.1f}m  "
              f"max={nearest.max():.1f}m")

    # Cap based on observed max gap distance (164.5m in our data -> 175m)
    OCCLUSION_MAX_METERS = 175

    density_healed = 0
    for sweep in range(5):
        added_this_sweep = 0

        # Sweep 1: strict alignment — easy straight gaps
        # Sweep 2+: relaxed alignment — curves, T-junctions, harder geometry
        alignment_threshold = 0.55 if sweep == 0 else 0.40

        dead_ends = [n for n in G_healed.nodes() if G_healed.degree(n) == 1]
        if not dead_ends:
            print(f"   Sweep {sweep + 1}: no dead-ends remain — stopping.")
            break

        de_coords = np.array([
            [G_healed.nodes[n]['x'], G_healed.nodes[n]['y']]
            for n in dead_ends
        ])

        # Sweep 1: dead-end -> dead-end only (clean, no risk of mid-road attachment)
        # Sweep 2+: dead-end -> any node with degree < 4
        #           catches orphaned dead-ends whose partner was consumed by Phase 1
        if sweep == 0:
            target_nodes = dead_ends
            target_coords = de_coords
        else:
            target_nodes = list(G_healed.nodes())
            target_coords = np.array([
                [G_healed.nodes[n]['x'], G_healed.nodes[n]['y']]
                for n in target_nodes
            ])

        target_tree = cKDTree(target_coords)
        k = min(15, len(target_nodes))
        dists_arr, idx_arr = target_tree.query(de_coords, k=k)

        candidates = []
        for i, node_u in enumerate(dead_ends):
            for j_idx, dist in zip(idx_arr[i], dists_arr[i]):
                if dist == 0 or dist > OCCLUSION_MAX_METERS:
                    continue
                node_v = target_nodes[j_idx]
                if node_v == node_u or G_healed.has_edge(node_u, node_v):
                    continue

                # Sweep 2+: don't attach to busy intersections (degree >= 4)
                # Those are genuine X/T junctions that shouldn't gain a new arm
                if sweep > 0 and G_healed.degree(node_v) >= 4:
                    continue

                alignment = angular_alignment_score(G_healed, node_u, node_v)
                if alignment < alignment_threshold:
                    continue

                key = (min(node_u, node_v), max(node_u, node_v))
                candidates.append((key[0], key[1], dist, alignment))

        # Deduplicate: keep best alignment per node pair
        best = {}
        for u, v, d, a in candidates:
            if (u, v) not in best or a > best[(u, v)][1]:
                best[(u, v)] = (d, a)
        candidates = [(u, v, d, a) for (u, v), (d, a) in best.items()]
        candidates.sort(key=lambda x: x[2] / (x[3] + 1e-6))

        # Each dead-end gets at most one new connection per sweep
        # Prevents one node from sprouting multiple bridges in a single pass
        connected_this_sweep = set()
        for u, v, dist, alignment in candidates:
            if G_healed.has_edge(u, v):
                continue
            if u in connected_this_sweep or v in connected_this_sweep:
                continue
            G_healed.add_edge(u, v, weight=dist, length=dist,
                               healed=True, alignment=alignment)
            connected_this_sweep.add(u)
            connected_this_sweep.add(v)
            added_this_sweep += 1
            density_healed += 1

        remaining = len([n for n in G_healed.nodes() if G_healed.degree(n) == 1])
        target_desc = "dead-ends" if sweep == 0 else "all nodes"
        print(f"   Sweep {sweep + 1} (align>={alignment_threshold}, ->{target_desc}): "
              f"{added_this_sweep} pairs connected "
              f"({len(dead_ends)} dead-ends -> {remaining} remaining)")

        if added_this_sweep == 0:
            break

    total_healed += density_healed

    # ── PHASE 3: Interior edge recovery (degree-2 pass-through nodes) ──
    # Recovers removed edges whose BOTH endpoints were intersections in
    # the broken graph — invisible to the dead-end based Phase 1 & 2.
    print("\n── Phase 3: Interior edge recovery ──")
    G_healed, interior_added = recover_interior_edges(
        G_healed, max_dist=OCCLUSION_MAX_METERS, alignment_thresh=0.50
    )
    print(f"   Interior recovery: {interior_added} degree-2 pairs reconnected")
    total_healed += interior_added

    final_components = nx.number_connected_components(G_healed)
    print(f"\nHealing complete — {total_healed} total bridges added, "
          f"{final_components} component(s) remain.")
    return G_healed


# ─────────────────────────────────────────────
# 5. METRICS
# ─────────────────────────────────────────────

def calculate_hackathon_metrics(G_before, G_after):
    components_before  = nx.number_connected_components(G_before)
    components_after   = nx.number_connected_components(G_after)
    largest_cc_before  = len(max(nx.connected_components(G_before), key=len))
    largest_cc_after   = len(max(nx.connected_components(G_after),  key=len))
    connectivity_ratio = largest_cc_after / largest_cc_before

    print("\n" + "=" * 40)
    print("       HACKATHON EVALUATION METRICS       ")
    print("=" * 40)
    print(f"Isolated Sub-networks (Before) : {components_before}")
    print(f"Isolated Sub-networks (After)  : {components_after}")
    print(f"Largest Component Size (Before): {largest_cc_before} nodes")
    print(f"Largest Component Size (After) : {largest_cc_after} nodes")
    print(f"Connectivity Ratio             : {connectivity_ratio:.4f}x growth")
    print("=" * 40 + "\n")


# ─────────────────────────────────────────────
# 6. VISUALISATION
# ─────────────────────────────────────────────

def plot_three_panel(G_original, G_broken, G_healed):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8), sharex=True, sharey=True)

    # Panel 1 — Original
    ox.plot_graph(G_original, ax=ax1, node_size=10, node_color='#33a02c',
                  edge_color='#44aa44', edge_linewidth=1.0, show=False, close=False)
    ax1.set_title(
        f"ORIGINAL GRAPH\n({len(G_original.nodes)} nodes, {len(G_original.edges)} edges)",
        color='white', fontsize=13)
    ax1.set_facecolor('#111111')

    # Panel 2 — Broken
    ox.plot_graph(G_broken, ax=ax2, node_size=10, node_color='#33a02c',
                  edge_color='#44aa44', edge_linewidth=1.0, show=False, close=False)
    ax2.set_title(
        f"BROKEN GRAPH\n({len(G_broken.nodes)} nodes — "
        f"{nx.number_connected_components(G_broken)} components)",
        color='white', fontsize=13)
    ax2.set_facecolor('#111111')

    # Panel 3 — Healed (green base + red bridges)
    ox.plot_graph(G_healed, ax=ax3, node_size=10, node_color='#33a02c',
                  edge_color='#44aa44', edge_linewidth=1.0, show=False, close=False)

    healed_lines_x, healed_lines_y = [], []
    for u, v, data in G_healed.edges(data=True):
        if data.get('healed'):
            x1, y1 = G_healed.nodes[u]['x'], G_healed.nodes[u]['y']
            x2, y2 = G_healed.nodes[v]['x'], G_healed.nodes[v]['y']
            healed_lines_x.append([x1, x2])
            healed_lines_y.append([y1, y2])

    for hx, hy in zip(healed_lines_x, healed_lines_y):
        ax3.plot(hx, hy, color='red', linewidth=1.0, zorder=5, solid_capstyle='round')

    ax3.set_title(
        f"HEALED GRAPH\n({len(healed_lines_x)} bridges — "
        f"{nx.number_connected_components(G_healed)} components remain)",
        color='white', fontsize=13)
    ax3.set_facecolor('#111111')

    fig.patch.set_facecolor('#111111')
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# 7. GROUND TRUTH COMPARISON
# ─────────────────────────────────────────────

def compare_to_ground_truth(G_original, G_broken, G_healed, tolerance_meters=50):
    """
    Compares the healed graph against the ground truth on two axes:

    A) Edge-level precision / recall — with 1-to-1 matching so recall
       is correctly bounded in [0, 1]. Each original edge can only be
       claimed by one healed edge (greedy, shortest-distance first).

    B) Connectivity structure similarity — pairwise reachability sampled
       across the original's connected components.
    """

    # ── A) EDGE-LEVEL PRECISION / RECALL (1-to-1 matching) ───────────

    orig_edge_midpoints = []
    for u, v in G_original.edges():
        mx = (G_original.nodes[u]['x'] + G_original.nodes[v]['x']) / 2
        my = (G_original.nodes[u]['y'] + G_original.nodes[v]['y']) / 2
        orig_edge_midpoints.append([mx, my])

    orig_mid_arr = np.array(orig_edge_midpoints)
    orig_tree = cKDTree(orig_mid_arr)

    healed_edges = [(u, v, d) for u, v, d in G_healed.edges(data=True) if d.get('healed')]

    # 1-to-1 matching — each original edge matched at most once
    # Sort healed edges by distance to nearest original (best matches first)
    scored = []
    for u, v, data in healed_edges:
        mx = (G_healed.nodes[u]['x'] + G_healed.nodes[v]['x']) / 2
        my = (G_healed.nodes[u]['y'] + G_healed.nodes[v]['y']) / 2
        dist, idx = orig_tree.query([mx, my])
        scored.append((dist, idx, u, v))
    scored.sort(key=lambda x: x[0])

    matched_orig_indices = set()
    true_positives  = 0
    false_positives = 0

    for dist, idx, u, v in scored:
        if dist <= tolerance_meters and idx not in matched_orig_indices:
            true_positives += 1
            matched_orig_indices.add(idx)  # each original edge claimed at most once
        else:
            false_positives += 1

    broken_edge_set = set((min(u, v), max(u, v)) for u, v in G_broken.edges())
    orig_edge_set   = set((min(u, v), max(u, v)) for u, v in G_original.edges())
    removed_edges   = orig_edge_set - broken_edge_set
    n_removed       = len(removed_edges)

    precision = true_positives / len(healed_edges) if healed_edges else 0.0
    recall    = true_positives / n_removed if n_removed > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    # ── B) CONNECTIVITY STRUCTURE SIMILARITY ─────────────────────────

    orig_components = list(nx.connected_components(G_original))
    common_nodes    = set(G_original.nodes) & set(G_healed.nodes)

    same_comp_hits  = 0
    same_comp_total = 0
    diff_comp_hits  = 0
    diff_comp_total = 0

    rng = np.random.default_rng(42)
    sample_per_comp = 10

    for comp in orig_components:
        comp_nodes = list(comp & common_nodes)
        if len(comp_nodes) < 2:
            continue
        for _ in range(sample_per_comp):
            a, b = rng.choice(comp_nodes, size=2, replace=False)
            same_comp_total += 1
            if nx.has_path(G_healed, a, b):
                same_comp_hits += 1

    for i, comp_a in enumerate(orig_components[:10]):
        for comp_b in orig_components[i + 1:11]:
            nodes_a = list(comp_a & common_nodes)
            nodes_b = list(comp_b & common_nodes)
            if not nodes_a or not nodes_b:
                continue
            a = rng.choice(nodes_a)
            b = rng.choice(nodes_b)
            diff_comp_total += 1
            if nx.has_path(G_healed, a, b):
                diff_comp_hits += 1

    intra_connectivity = same_comp_hits / same_comp_total if same_comp_total else 0.0
    inter_connectivity = diff_comp_hits / diff_comp_total if diff_comp_total else 0.0

    # ── PRINT REPORT ──────────────────────────────────────────────────

    print("\n" + "=" * 45)
    print("        GROUND TRUTH COMPARISON REPORT        ")
    print("=" * 45)

    print("\n── A) Edge-Level Accuracy ──────────────────")
    print(f"  Bridges added           : {len(healed_edges)}")
    print(f"  Edges removed by break  : {n_removed}")
    print(f"  True Positives          : {true_positives}  (bridge matched a real edge)")
    print(f"  False Positives         : {false_positives} (bridge has no real counterpart)")
    print(f"  Precision               : {precision:.3f}  (of added bridges, how many were real)")
    print(f"  Recall                  : {recall:.3f}  (of removed edges, how many we recovered)")
    print(f"  F1 Score                : {f1:.3f}")

    print("\n── B) Connectivity Structure ───────────────")
    print(f"  Intra-component pairs tested : {same_comp_total}")
    print(f"  Still connected in healed   : {same_comp_hits} ({intra_connectivity:.1%})")
    print(f"  Cross-component pairs tested : {diff_comp_total}")
    print(f"  Correctly re-joined         : {diff_comp_hits} ({inter_connectivity:.1%})")

    print("\n── Summary ─────────────────────────────────")
    overall = (f1 + intra_connectivity + inter_connectivity) / 3
    print(f"  Overall Score (avg)     : {overall:.3f}")
    print("=" * 45 + "\n")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "intra_connectivity": intra_connectivity,
        "inter_connectivity": inter_connectivity,
        "overall": overall,
    }


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    G_original = load_and_project_graph("mock_bengaluru_ground_truth.gpickle")
    G_broken   = load_and_project_graph("mock_bengaluru_broken.gpickle")

    G_healed = heal_topological_gaps(
        G_broken,
        max_dist_meters=400,   # Phase 1 structural pass — can be wide
        min_alignment=0.30,    # Phase 1 alignment threshold
        max_passes=5,
    )

    calculate_hackathon_metrics(G_broken, G_healed)
    compare_to_ground_truth(G_original, G_broken, G_healed, tolerance_meters=50)
    plot_three_panel(G_original, G_broken, G_healed)