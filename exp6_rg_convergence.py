import numpy as np
import pandas as pd
import networkx as nx
import math

# Build a micro layered DAG: width=2, depth=3, with full bipartite connections between layers
def build_micro_dag(width=2, depth=3):
    G = nx.DiGraph()
    layer_nodes = {}
    nid = 0
    for layer in range(depth + 1):
        count = 1 if layer == 0 else width
        nodes = list(range(nid, nid + count))
        layer_nodes[layer] = nodes
        for node in nodes:
            G.add_node(node, layer=layer)
        nid += count
    # Connect each node in layer l to all nodes in layer l+1
    for layer in range(depth):
        for u in layer_nodes[layer]:
            for v in layer_nodes[layer + 1]:
                G.add_edge(u, v)
    return G, layer_nodes

# Compute log2(path-count) metric via DP for all pairs
def compute_path_metric(G):
    topo = list(nx.topological_sort(G))
    d = {}
    for u in topo:
        counts = {n: 0 for n in G.nodes}
        counts[u] = 1
        for x in topo[topo.index(u):]:
            for w in G.successors(x):
                counts[w] += counts[x]
        for v, cnt in counts.items():
            if cnt > 0:
                d[(u, v)] = math.log2(cnt)
    return d

# Build k-block DAG: edges for exact path length k
def build_block_dag(G, k):
    Gk = nx.DiGraph()
    Gk.add_nodes_from(G.nodes(data=True))
    for u in G.nodes:
        # BFS up to depth k
        lengths = nx.single_source_shortest_path_length(G, u, cutoff=k)
        for v, l in lengths.items():
            if l == k:
                Gk.add_edge(u, v)
    return Gk

# Simulation
width = 2
depth = 3
G, layer_nodes = build_micro_dag(width, depth)

# Original metric
d_orig = compute_path_metric(G)

results = []
for k in [1, 2, 3]:
    # Build block DAG for block size k
    Gk = build_block_dag(G, k)
    # Compute block metric (counts only length-k paths)
    # path counts from u to v of length k is (A^k)[u,v], where A is adjacency
    A = nx.to_numpy_array(G, dtype=int)
    Ak = np.linalg.matrix_power(A, k)
    kappas = []
    for u, v in Gk.edges:
        count_k = Ak[u, v]
        if count_k > 0:
            d_k = math.log2(count_k)
            # curvature density per edge: (d_k / k) - original metric d_orig
            kappa_uv = (d_k / k) - d_orig[(u, v)]
            kappas.append(kappa_uv)
    kappa_avg = np.mean(kappas)
    # Memory density: assume simple fusion yields log2(width) bits per layer => rho_mem = k*log2(width)
    rho_mem = k * math.log2(width)
    g = kappa_avg / rho_mem if rho_mem != 0 else None
    results.append({'k': k, 'kappa_avg': kappa_avg, 'rho_mem': rho_mem, 'g': g})

# Display result table to user
import ace_tools as tools; tools.display_dataframe_to_user("Micro-DAG RG Trend Demo", pd.DataFrame(results))

# Note: for this micro example, g = (1-k)/k, showing trend toward -1

