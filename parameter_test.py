# ==============================================================================

# This version corrects a bug in the original simulation. The initial model
# failed to generate entropy because its first layer of nodes was static,
# leading to a highly predictable, uniform evolution.
#
# THE FIX:
# The first layer of the graph (nodes with no predecessors) is now treated
# as a source of new, unpredictable information, simulating the "hidden layer"
# (H_min) from the Alayar papers. At each tick, these nodes are assigned
# a new random tag. This continuous injection of "fresh" information drives
# the complex evolution and entropy growth of the entire system.
#
# This corrected model should now produce a meaningful result, identifying
# a non-trivial (q*, R*) pair that maximizes entropy generation.
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import hashlib
from tqdm import tqdm

def build_layered_dag(num_layers, nodes_per_layer, R_max):
    """
    Builds a layered Directed Acyclic Graph (DAG) to serve as the "universe".

    This represents the "finite acyclic causal site" from the papers.
    Edges only go from a layer to the next, representing causal flow.

    Args:
        num_layers (int): The number of layers (depth) in the graph.
        nodes_per_layer (int): The number of nodes (sites) in each layer.
        R_max (int): The maximum number of outgoing edges (fan-out) for any node.

    Returns:
        dict: A dictionary representing the predecessors for each node.
        list: A list of node IDs in the first layer.
    """
    nodes = {}
    for layer in range(num_layers):
        nodes[layer] = list(range(layer * nodes_per_layer, (layer + 1) * nodes_per_layer))

    predecessors = {node: [] for layer_nodes in nodes.values() for node in layer_nodes}
    first_layer_nodes = nodes[0]

    # Connect nodes from layer i to layer i+1
    for i in range(num_layers - 1):
        current_layer_nodes = nodes[i]
        next_layer_nodes = nodes[i+1]
        
        for node in current_layer_nodes:
            # Each node connects to R_max randomly chosen nodes in the next layer
            # This ensures the fan-out does not exceed R_max
            connections = np.random.choice(next_layer_nodes, size=R_max, replace=False)
            for conn in connections:
                predecessors[conn].append(node)
                
    return predecessors, first_layer_nodes

def shannon_entropy(tags):
    """
    Calculates the Shannon entropy of a list of tags.
    H(X) = - sum(p(x) * log2(p(x)))
    """
    if not tags:
        return 0
    
    n = len(tags)
    counts = Counter(tags)
    
    entropy = 0.0
    for count in counts.values():
        # Probability of a tag
        p_x = count / n
        if p_x > 0:
            entropy -= p_x * np.log2(p_x)
            
    return entropy

def run_simulation(q, R, num_layers, nodes_per_layer, ticks):
    """
    Runs a single simulation for a given (q, R) pair and calculates the
    average entropy increment.

    Args:
        q (int): The size of the tag alphabet.
        R (int): The maximum connectivity (fan-out).
        num_layers (int): Number of layers in the causal site.
        nodes_per_layer (int): Nodes per layer.
        ticks (int): Number of time steps to simulate.

    Returns:
        float: The average entropy increment per tick.
    """
    # 1. Build the universe with the specified connectivity R
    num_total_nodes = num_layers * nodes_per_layer
    predecessors, first_layer_nodes = build_layered_dag(num_layers, nodes_per_layer, R)
    
    # 2. Initialize the state of the universe (tags)
    tags = {node: 0 for node in range(num_total_nodes)}
    
    entropy_history = [shannon_entropy(list(tags.values()))]
    
    # 3. Run the simulation for the specified number of ticks
    for _ in range(ticks):
        new_tags = {}
        for node in range(num_total_nodes):
            # *** THE FIX IS HERE ***
            # The first layer acts as the "hidden layer" (H_min), injecting
            # new, unpredictable information at each tick.
            if node in first_layer_nodes:
                new_tags[node] = np.random.randint(0, q)
                continue

            # For all other nodes, their new tag is a deterministic function
            # of their predecessors' tags from the previous tick.
            pred_list = predecessors[node]
            pred_tags = tuple(sorted([tags[p] for p in pred_list]))
            
            # Apply the deterministic "Tag-Fusion" rule
            h = hashlib.sha256(str(pred_tags).encode()).hexdigest()
            new_tags[node] = int(h, 16) % q

        tags = new_tags
        entropy_history.append(shannon_entropy(list(tags.values())))

    # 5. Calculate the entropy increments and average
    entropy_increments = np.diff(entropy_history)
    stable_increments = entropy_increments[int(ticks * 0.2):] # Ignore initial transient phase
    
    if len(stable_increments) > 0:
        return np.mean(stable_increments)
    else:
        return 0

def main():
    """
    Main function to run the parameter search and visualize the results.
    """
    print("Starting simulation to test the Parameter Elimination Theorem...")
    
    # --- Simulation Parameters ---
    Q_RANGE = range(2, 11)
    R_RANGE = range(2, 11)
    
    NUM_LAYERS = 5
    NODES_PER_LAYER = 20
    TICKS = 100
    
    results = np.zeros((len(Q_RANGE), len(R_RANGE)))
    
    with tqdm(total=len(Q_RANGE) * len(R_RANGE), desc="Simulating (q, R) pairs") as pbar:
        for i, q in enumerate(Q_RANGE):
            for j, R in enumerate(R_RANGE):
                avg_entropy_increment = run_simulation(
                    q=q, R=R, num_layers=NUM_LAYERS,
                    nodes_per_layer=NODES_PER_LAYER, ticks=TICKS
                )
                results[i, j] = avg_entropy_increment
                pbar.update(1)

    # --- Find and Print the Maximizer ---
    max_val = np.max(results)
    if max_val <= 0:
        print("\n--- Simulation Warning ---")
        print("The simulation did not generate significant entropy.")
        print("This might be due to parameters (e.g., too few ticks or nodes).")
        print("Displaying results, but they may not be meaningful.")
        q_star, R_star = Q_RANGE[0], R_RANGE[0] # Default to first values
    else:
        max_indices = np.unravel_index(np.argmax(results, axis=None), results.shape)
        q_star = Q_RANGE[max_indices[0]]
        R_star = R_RANGE[max_indices[1]]
    
    print("\n--- Simulation Complete ---")
    print(f"The parameter pair that maximizes entropy generation is:")
    print(f"  q* = {q_star} (Alphabet Size)")
    print(f"  R* = {R_star} (Max Connectivity)")
    print(f"  Maximum Average Entropy Increment: {max_val:.4f} bits/tick")

    # --- Visualize the Results ---
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(results.T, interpolation='nearest', origin='lower',
                    cmap='viridis', aspect='auto')
    
    fig.colorbar(cax, label='Average Entropy Increment per Tick')
    
    ax.set_title(f'Entropy Generation vs. System Parameters (q, R)\nMaximizer at (q*, R*) = ({q_star}, {R_star})', fontsize=14)
    ax.set_xlabel('q (Alphabet Size)', fontsize=12)
    ax.set_ylabel('R (Max Connectivity)', fontsize=12)
    
    ax.set_xticks(np.arange(len(Q_RANGE)))
    ax.set_yticks(np.arange(len(R_RANGE)))
    ax.set_xticklabels(Q_RANGE)
    ax.set_yticklabels(R_RANGE)
    
    if max_val > 0:
        ax.scatter(max_indices[0], max_indices[1], color='red', s=150,
                   edgecolors='white', marker='*', label=f'Maximizer ({q_star}, {R_star})')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
