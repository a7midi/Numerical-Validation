import itertools
import random
import math
from collections import defaultdict

def generate_depth_graded_dag(widths, edge_prob=0.5):
    """
    Generate a depth-graded DAG with random edges between consecutive depths.
    widths: dict mapping depth -> number of nodes at that depth.
    edge_prob: probability of an edge from a parent at depth d to child at depth d+1.
    Returns: (nodes, depth_map, parent_map)
    """
    nodes = []
    depth_map = {}
    for d, w in widths.items():
        for i in range(w):
            node = f"{d}_{i}"
            nodes.append(node)
            depth_map[node] = d

    parent_map = defaultdict(list)
    for d, w in widths.items():
        if (d + 1) in widths:
            for i in range(w):
                parent = f"{d}_{i}"
                for j in range(widths[d + 1]):
                    child = f"{d+1}_{j}"
                    if random.random() < edge_prob:
                        parent_map[child].append(parent)
    return nodes, depth_map, parent_map

def nodes_by_depth(depth_map):
    nodes_depth = defaultdict(list)
    for node, d in depth_map.items():
        nodes_depth[d].append(node)
    return nodes_depth

def enumerate_fusion_microstates(nodes_depth, parent_map, hidden_alphabet_size):
    """
    Enumerate all possible fusion-state histories given DAG structure and a fusion rule:
    child_tag = (sum(parent_tags) + new_input_tag) mod hidden_alphabet_size.
    Returns list of entropies S_t at each tick.
    """
    states = [dict()]  # initial state, empty history
    entropies = [0.0]  # log2 of number of states
    for t in sorted(nodes_depth.keys()):
        new_nodes = nodes_depth[t]
        new_states = []
        for state in states:
            # all combinations of new input tags
            for input_tags in itertools.product(range(hidden_alphabet_size), repeat=len(new_nodes)):
                new_state = state.copy()
                for node, tag_input in zip(new_nodes, input_tags):
                    parents = parent_map.get(node, [])
                    parent_tags = [state[(p, t-1)] for p in parents] if parents else []
                    fused_parent_tag = sum(parent_tags) % hidden_alphabet_size if parent_tags else 0
                    child_tag = (fused_parent_tag + tag_input) % hidden_alphabet_size
                    new_state[(node, t)] = child_tag
                new_states.append(new_state)
        states = new_states
        entropies.append(math.log2(len(states)))
    return entropies

def test_random_fusion_dags(num_trials=3, max_depth=4, max_nodes_per_level=2, edge_prob=0.7, hidden_alphabet_size=2):
    random.seed(42)
    for i in range(1, num_trials + 1):
        widths = {d: random.randint(1, max_nodes_per_level) for d in range(max_depth + 1)}
        nodes, depth_map, parent_map = generate_depth_graded_dag(widths, edge_prob=edge_prob)
        nbd = nodes_by_depth(depth_map)
        entropies = enumerate_fusion_microstates(nbd, parent_map, hidden_alphabet_size)
        diffs = [entropies[t+1] - entropies[t] for t in range(len(entropies) - 1)]
        print(f"Trial {i}: widths = {widths}")
        print(f"  Parent map: {dict(parent_map)}")
        print(f"  Entropies: {[round(e, 2) for e in entropies]}")
        print(f"  Diffs:     {[round(d, 2) for d in diffs]}")
        print()

if __name__ == "__main__":
    test_random_fusion_dags()
