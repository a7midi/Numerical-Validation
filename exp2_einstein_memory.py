import math
from collections import defaultdict

def generate_complete_layered_dag(width, max_depth):
    # Build nodes and edges so every node at depth d points to every node at d+1.
    nodes, depth = [], {}
    for d in range(max_depth+1):
        for i in range(width):
            node = (d, i)
            nodes.append(node)
            depth[node] = d

    edges = defaultdict(list)
    for d in range(max_depth):
        for i in range(width):
            u = (d, i)
            for j in range(width):
                v = (d+1, j)
                edges[u].append(v)
    return nodes, depth, edges

def count_all_paths(edges, nodes):
    # P(x) = total number of directed paths (of any positive length) starting at x.
    # Since DAG, we can DP in reverse topological (by depth descending).
    by_depth = sorted(nodes, key=lambda n: -n[0])
    P = {n: 0 for n in nodes}
    for u in by_depth:
        total = 0
        for v in edges[u]:
            total += 1 + P[v]
        P[u] = total
    return P

def compute_kappa_rho(edges, nodes, P):
    # For each arrow u->v:
    #  κ = P[v] - P[u]
    #  ρ_mem = #Pred(j)^2 - #Pred(i)^2   per Lemma 5.4:
    #    = |Pred(v)|^2 - |Pred(u)|^2
    # (equivalently, ρ_mem = log₂|A_c| but reconstructed via Pred counts here).
    # Then g = κ / ρ_mem.
    # We compute Pred sets by inverting edges.
    preds = defaultdict(set)
    for u in nodes:
        for v in edges[u]:
            preds[v].add(u)

    g_values = []
    for u in nodes:
        for v in edges[u]:
            kapp = P[v] - P[u]
            rho = len(preds[v])**2 - len(preds[u])**2
            if rho != 0:
                g_values.append(kapp / rho)
    return g_values

if __name__ == "__main__":
    w, D = 3, 4  # width and max depth
    nodes, depth, edges = generate_complete_layered_dag(w, D)
    P = count_all_paths(edges, nodes)
    g_vals = compute_kappa_rho(edges, nodes, P)

    print(f"Unique g values: {sorted(set(g_vals))}")
