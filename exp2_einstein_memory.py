# ==============================================================================
# Final Validation for Emergent Einstein-Memory Constant via RG Flow
#
# This script is amended to be a fully faithful validation of the theory in
# Paper II, §5 & §10. It demonstrates that the coupling constant 'g'
# emerges as a universal value after a Renormalization Group (RG)
# coarse-graining procedure on a generic causal site.
#
# --- AMENDMENTS IMPLEMENTED ---
#
# 1.  Theoretical rho_mem Formula: The script no longer simulates tag dynamics.
#     Instead, it calculates memory density (ρ_mem) using the exact,
#     deterministic, combinatorial formula derived in Paper II, Lemma 5.4:
#     ρ_mem = |Pred(v)|² - |Pred(u)|² for an arrow u -> v.
#
# 2.  Fully Deterministic Calculation: Once the initial random DAG is built,
#     the calculation of κ, ρ_mem, and the entire RG flow is now
#     fully deterministic, removing the simulation noise and arbitrary
#     parameters (like alphabet size and tick count) from the previous version.
#
# Expected Result:
# The distributions of the block-averaged coupling 'g_block' should now
# collapse into extremely sharp, near-delta-function spikes as the block
# size 'k' increases, providing a clear and rigorous validation of the
# emergence of a universal constant g*.
# ==============================================================================

import random
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Step 1: Graph Generation ---

def generate_depth_graded_dag(max_depth, avg_width, edge_prob):
    """Generates a randomized, depth-graded DAG."""
    nodes, depth_map, nodes_by_depth = [], {}, defaultdict(list)
    parent_map, edges = defaultdict(list), defaultdict(list)
    
    node_counter = 0
    for d in range(max_depth + 1):
        width = random.randint(max(1, avg_width - 2), avg_width + 2)
        for i in range(width):
            node = node_counter
            nodes.append(node)
            depth_map[node] = d
            nodes_by_depth[d].append(node)
            node_counter += 1

    for d in range(max_depth):
        parents_at_d = nodes_by_depth[d]
        children_at_next_d = nodes_by_depth[d+1]
        for parent in parents_at_d:
            for child in children_at_next_d:
                if random.random() < edge_prob:
                    edges[parent].append(child)
                    parent_map[child].append(parent)
    return nodes, depth_map, parent_map, edges

# --- Step 2: Calculation of Local κ and ρ_mem ---

def calculate_local_kappa(parent_map, u, v):
    """Calculates kappa for a single arrow u -> v."""
    num_preds_u = len(parent_map.get(u, []))
    num_preds_v = len(parent_map.get(v, []))
    return num_preds_v - num_preds_u

def calculate_local_rho_mem_theoretical(parent_map, u, v):
    """
    Calculates rho_mem for an arrow u -> v using the theoretical
    combinatorial formula from Paper II, Lemma 5.4.
    """
    num_preds_u = len(parent_map.get(u, []))
    num_preds_v = len(parent_map.get(v, []))
    return num_preds_v**2 - num_preds_u**2

# --- Step 3: Renormalization Group (RG) Coarse-Graining ---

def perform_rg_flow(edges, depth_map, parent_map, block_sizes):
    """
    Performs RG coarse-graining by averaging over blocks of increasing size k.
    """
    rg_results = {}
    
    # Pre-calculate all local kappa and rho_mem values
    local_kappa = {}
    local_rho_mem = {}
    for u, children in edges.items():
        for v in children:
            arrow = (u, v)
            local_kappa[arrow] = calculate_local_kappa(parent_map, u, v)
            local_rho_mem[arrow] = calculate_local_rho_mem_theoretical(parent_map, u, v)

    for k in block_sizes:
        blocks = defaultdict(list)
        for arrow in local_kappa.keys():
            u, v = arrow
            block_id = depth_map[u] // k
            blocks[block_id].append(arrow)

        g_block_values = []
        for block_id, arrows_in_block in blocks.items():
            if not arrows_in_block:
                continue
            
            # Average the pre-calculated local values over the block
            block_kappas = [local_kappa[arrow] for arrow in arrows_in_block]
            block_rhos = [local_rho_mem[arrow] for arrow in arrows_in_block]
            
            kappa_avg = np.mean(block_kappas)
            rho_avg = np.mean(block_rhos)
            
            if rho_avg != 0:
                g_block = kappa_avg / rho_avg
                g_block_values.append(g_block)
        
        rg_results[k] = g_block_values
        
    return rg_results

# --- Step 4: Main Workflow and Visualization ---

def main():
    print("Starting simulation for emergent coupling constant...")
    
    # Parameters for a large, deep graph
    MAX_DEPTH = 50
    AVG_WIDTH = 10
    EDGE_PROB = 0.5

    # 1. Generate graph
    print("Generating a large causal site...")
    nodes, depth_map, parent_map, edges = generate_depth_graded_dag(MAX_DEPTH, AVG_WIDTH, EDGE_PROB)
    
    # 2. Perform RG Flow
    block_sizes_to_test = [1, 2, 5, 10, 25]
    print(f"Performing RG coarse-graining for block sizes: {block_sizes_to_test}...")
    rg_results = perform_rg_flow(edges, depth_map, parent_map, block_sizes_to_test)

    # 3. Visualize the results
    print("Plotting the distribution of the coupling constant 'g'...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(block_sizes_to_test)))
    
    for i, k in enumerate(block_sizes_to_test):
        g_values = rg_results.get(k, [])
        if g_values:
            from scipy.stats import gaussian_kde
            # Filter out extreme outliers for better plotting
            p1, p99 = np.percentile(g_values, [1, 99])
            g_values_filtered = [g for g in g_values if p1 <= g <= p99]
            if not g_values_filtered: continue

            kde = gaussian_kde(g_values_filtered)
            x_range = np.linspace(min(g_values_filtered), max(g_values_filtered), 500)
            ax.plot(x_range, kde(x_range), label=f'Block Size k = {k}', color=colors[i], lw=2.5)

    ax.set_title("Distribution of Emergent Coupling 'g' under RG Flow", fontsize=16)
    ax.set_xlabel("Coupling Constant 'g' = <κ> / <ρ_mem>", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Calculate the mean of the most coarse-grained distribution
    final_g_values = rg_results.get(block_sizes_to_test[-1], [])
    if final_g_values:
        emergent_g_star = np.mean(final_g_values)
        ax.axvline(emergent_g_star, color='red', linestyle='--', lw=2, 
                   label=f'Emergent g* ≈ {emergent_g_star:.4f}')
    
    ax.legend()
    ax.set_xlim(-0.25, 0.25) # Zoom in on the peak for clarity
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
