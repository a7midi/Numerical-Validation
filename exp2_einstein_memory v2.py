import random
from collections import defaultdict

# --- Step 1: Use a more generic graph generator ---
def generate_depth_graded_dag(widths, edge_prob=0.7):
    """
    Generates a more generic, randomized depth-graded DAG.
    """
    nodes = []
    depth_map = {}
    nodes_by_depth = defaultdict(list)
    edges = defaultdict(list)

    for d in sorted(widths.keys()):
        for i in range(widths[d]):
            node = (d, i)
            nodes.append(node)
            depth_map[node] = d
            nodes_by_depth[d].append(node)

    parent_map = defaultdict(list)
    sorted_depths = sorted(widths.keys())
    for idx, d in enumerate(sorted_depths[:-1]):
        next_d = sorted_depths[idx+1]
        parents_at_d = nodes_by_depth[d]
        children_at_next_d = nodes_by_depth[next_d]
        for parent in parents_at_d:
            for child in children_at_next_d:
                if random.random() < edge_prob:
                    edges[parent].append(child)
                    parent_map[child].append(parent)

    return nodes, depth_map, nodes_by_depth, parent_map, edges

# --- Step 2: Implement the correct definition of kappa ---
def calculate_kappa_from_predecessors(nodes, parent_map, edges):
    """
    Calculates kappa using the paper's primary definition: V₁(j) - V₁(i),
    where V₁(c) is the number of predecessors of c.
    """
    kappa_values = {}
    for u, children in edges.items():
        num_preds_u = len(parent_map.get(u, []))
        for v in children:
            num_preds_v = len(parent_map.get(v, []))
            kappa_values[(u, v)] = num_preds_v - num_preds_u
    return kappa_values

# --- Step 3: Implement the fundamental definition of rho_mem ---
def calculate_true_rho_mem(nodes, nodes_by_depth, parent_map, alphabet_size, num_ticks):
    """
    Calculates the true memory density by simulating tag evolution and counting
    the number of unique (tag_t, tag_t+1) transitions at each node.
    """
    current_tags = {node: random.randint(0, alphabet_size - 1) for node in nodes}
    transition_sets = {node: set() for node in nodes}

    for _ in range(num_ticks):
        next_tags = {}
        for d in sorted(nodes_by_depth.keys()):
            for node in nodes_by_depth[d]:
                parents = parent_map.get(node, [])
                parent_tags = [current_tags[p] for p in parents]
                fused_parent_tag = sum(parent_tags) % alphabet_size if parent_tags else 0
                
                # A simple, fixed deterministic update rule
                next_tags[node] = (fused_parent_tag + 1) % alphabet_size
        
        # Record transitions that occurred in this tick
        for node in nodes:
            pair = (current_tags[node], next_tags[node])
            if pair[0] != pair[1]:
                transition_sets[node].add(pair)
        
        current_tags = next_tags

    rho_mem_values = {node: len(transitions) for node, transitions in transition_sets.items()}
    return rho_mem_values

# --- Step 4: Main validation workflow ---
def run_validation_test(trial_num):
    """
    Orchestrates the full, corrected validation test.
    """
    print(f"--- Running Validation Trial #{trial_num} ---")
    
    # Parameters
    max_depth = 5
    avg_width = 4
    edge_prob = 0.6
    alphabet_size = 5  # Using a larger alphabet makes non-zero rho_mem more likely
    num_simulation_ticks = 100

    # 1. Generate a random, generic graph
    widths = {d: random.randint(avg_width - 1, avg_width + 1) for d in range(max_depth + 1)}
    nodes, depth_map, nbd, parent_map, edges = generate_depth_graded_dag(widths, edge_prob)
    print(f"Generated a random DAG with {len(nodes)} nodes and {sum(len(v) for v in edges.values())} edges.")

    # 2. Calculate kappa for every arrow using the correct definition
    kappa_values = calculate_kappa_from_predecessors(nodes, parent_map, edges)

    # 3. Calculate the true memory density by simulating the dynamics
    rho_mem_values = calculate_true_rho_mem(nodes, nbd, parent_map, alphabet_size, num_simulation_ticks)
    
    # 4. For every arrow where rho_mem is non-zero, calculate g
    g_values = []
    for u, children in edges.items():
        for v in children:
            arrow = (u, v)
            kappa = kappa_values.get(arrow)
            rho = rho_mem_values.get(v)
            
            if kappa is not None and rho is not None and rho != 0:
                g_values.append(kappa / rho)

    # 5. Analyze and report the results
    print(f"Found {len(g_values)} testable edges (where ρ_mem ≠ 0).")
    if not g_values:
        print("VALIDATION INCONCLUSIVE: No testable edges found in this random graph.")
        return

    unique_g = set(round(g, 8) for g in g_values)
    
    print(f"Unique 'g' values found: {unique_g}")
    if len(unique_g) == 1:
        print(f"--> VALIDATION SUCCESS: A single universal coupling constant g = {unique_g.pop()} was found.")
    else:
        print(f"--> VALIDATION FAILED: The coupling constant 'g' is not universal for this graph.")
    print("-" * 40 + "\n")


if __name__ == "__main__":
    for i in range(1, 4):
        run_validation_test(i)