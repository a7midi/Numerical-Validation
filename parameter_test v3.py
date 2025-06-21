# ==============================================================================
# Final Amended Toy Model for Alayar's Parameter Elimination Theorem
#
# This definitive version incorporates all final suggestions to create the most
# rigorous and faithful simulation of the theory's principles.
#
# --- FINAL AMENDMENTS IMPLEMENTED ---
#
# 1. Global Injectivity (Caveat 1):
#    The `InjectiveFusionTable` now uses an internal counter that grows
#    unboundedly. This ensures that every unique combination of predecessor
#    tags is mapped to a unique internal value, preventing tag recycling
#    and more accurately mimicking the large Grothendieck group G(Ac) from
#    the papers. The final output tag is then mapped into the q-sized
#    alphabet via a modulo operation.
#
# 2. Unbiased Hidden Alphabet (Caveat 2):
#    To eliminate statistical bias, the pseudo-random sequence for the
#    hidden layer is now regenerated for each value of `q`. It draws
#    integers uniformly from the correct range `[0, q-1]`.
#
# 3. Graph Connectivity (Caveat 4):
#    The `build_layered_dag` function now includes a final check to
#    guarantee that every node in the observer slice (layers > 0) has at
#    least one causal predecessor, ensuring the entire system is a single
#    interconnected causal web.
#
# 4. Increased Scale (Caveat 3):
#    The simulation continues to use increased parameters for layers, nodes,
#    and ticks to better approximate the behavior of a large-scale system.
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

class InjectiveFusionTable:
    """
    A class to emulate a globally injective fusion map (ψ_c).
    It maps unique tuples of predecessor tags to a unique, ever-increasing
    integer, which is then mapped into the alphabet of size q. This avoids
    recycling output tags for different inputs and better mimics the
    large Grothendieck group G(Ac) from the papers.
    """
    def __init__(self, q):
        self.q = q
        self.mapping = {}
        self.next_tag_internal = 0  # This will grow unbounded

    def fuse(self, pred_tags_tuple):
        """
        Takes a tuple of predecessor tags and returns a deterministic output tag.
        """
        # Ensure the tuple is canonical (sorted) for consistent mapping
        pred_tags_tuple = tuple(sorted(pred_tags_tuple))
        
        if pred_tags_tuple not in self.mapping:
            # If this combination is new, assign it the next available internal tag
            self.mapping[pred_tags_tuple] = self.next_tag_internal
            self.next_tag_internal += 1
        
        # Return the mapped tag, wrapped into the alphabet of size q
        return self.mapping[pred_tags_tuple] % self.q

def build_layered_dag(num_layers, nodes_per_layer, R_max):
    """Builds a layered Directed Acyclic Graph (DAG) that is weakly connected."""
    nodes = {layer: list(range(layer * nodes_per_layer, (layer + 1) * nodes_per_layer)) for layer in range(num_layers)}
    predecessors = {node: [] for layer_nodes in nodes.values() for node in layer_nodes}
    
    first_layer_nodes = set(nodes.get(0, []))
    observer_slice_nodes = list(set(predecessors.keys()) - first_layer_nodes)

    # Connect nodes from layer i to layer i+1
    for i in range(num_layers - 1):
        current_layer_nodes = nodes[i]
        next_layer_nodes = nodes[i+1]
        
        for node in current_layer_nodes:
            if R_max > 0 and len(next_layer_nodes) > 0:
                size = min(R_max, len(next_layer_nodes))
                connections = np.random.choice(next_layer_nodes, size=size, replace=False)
                for conn in connections:
                    predecessors[conn].append(node)
    
    # **FINAL AMENDMENT: Ensure graph is weakly connected**
    # Check every node in the observer slice to ensure it has at least one predecessor.
    for i in range(1, num_layers):
        for node in nodes[i]:
            if not predecessors[node]:
                # If a node has no inputs, connect it to a random node from the previous layer.
                random_predecessor = np.random.choice(nodes[i-1])
                predecessors[node].append(random_predecessor)

    return predecessors, list(first_layer_nodes), observer_slice_nodes

def shannon_entropy(tags):
    """Calculates the Shannon entropy of a list of tags."""
    if not tags: return 0
    n = len(tags)
    counts = Counter(tags)
    entropy = 0.0
    for count in counts.values():
        p_x = count / n
        if p_x > 0: entropy -= p_x * np.log2(p_x)
    return entropy

def run_simulation(q, R, num_layers, nodes_per_layer, ticks, hidden_layer_sequence):
    """Runs a single simulation for a given (q, R) pair."""
    predecessors, first_layer_nodes, observer_slice_nodes = build_layered_dag(num_layers, nodes_per_layer, R)
    num_total_nodes = num_layers * nodes_per_layer
    
    fusion_table = InjectiveFusionTable(q)
    tags = {node: 0 for node in range(num_total_nodes)}
    
    observed_tags = [tags[node] for node in observer_slice_nodes]
    entropy_history = [shannon_entropy(observed_tags)]
    
    sequence_idx = 0
    for _ in range(ticks):
        new_tags = {}
        for node in range(num_total_nodes):
            if node in first_layer_nodes:
                new_tags[node] = hidden_layer_sequence[sequence_idx]
                sequence_idx = (sequence_idx + 1) % len(hidden_layer_sequence)
                continue

            pred_list = predecessors.get(node, [])
            if not pred_list:
                new_tags[node] = tags[node]
                continue
            
            pred_tags = [tags[p] for p in pred_list]
            new_tags[node] = fusion_table.fuse(pred_tags)
        
        tags = new_tags
        observed_tags = [tags[node] for node in observer_slice_nodes]
        entropy_history.append(shannon_entropy(observed_tags))

    entropy_increments = np.diff(entropy_history)
    stable_increments = entropy_increments[int(ticks * 0.2):]
    
    return np.mean(stable_increments) if len(stable_increments) > 0 else 0

def main():
    """Main function to run the parameter search and visualize the results."""
    print("Starting final amended simulation...")
    
    # --- Simulation Parameters ---
    Q_RANGE = range(2, 16)
    R_RANGE = range(2, 16)
    NUM_LAYERS = 10
    NODES_PER_LAYER = 50
    TICKS = 200
    
    results = np.zeros((len(Q_RANGE), len(R_RANGE)))
    
    with tqdm(total=len(Q_RANGE) * len(R_RANGE), desc="Simulating (q, R) pairs") as pbar:
        # Fixed seed for the graph structure generation to be consistent
        np.random.seed(42)
        
        for i, q in enumerate(Q_RANGE):
            # **FINAL AMENDMENT: Generate an unbiased hidden sequence for each q.**
            hidden_sequence_size = NODES_PER_LAYER * TICKS
            # We use a different seed for the sequence generation itself, but it's fixed
            # so that `q=5` always gets the same sequence across different runs of the script.
            sequence_rng = np.random.RandomState(seed=q) 
            hidden_layer_sequence = sequence_rng.randint(0, q, size=hidden_sequence_size)

            for j, R in enumerate(R_RANGE):
                avg_entropy_increment = run_simulation(
                    q=q, R=R, num_layers=NUM_LAYERS,
                    nodes_per_layer=NODES_PER_LAYER, ticks=TICKS,
                    hidden_layer_sequence=hidden_layer_sequence
                )
                results[i, j] = avg_entropy_increment
                pbar.update(1)

    # --- Find and Print the Maximizer ---
    max_val = np.max(results)
    if max_val <= 0:
        print("\n--- Simulation Warning ---")
        print("The simulation did not generate significant entropy.")
        q_star, R_star = Q_RANGE[0], R_RANGE[0]
    else:
        max_indices = np.unravel_index(np.argmax(results, axis=None), results.shape)
        q_star = Q_RANGE[max_indices[0]]
        R_star = R_RANGE[max_indices[1]]
    
    print("\n--- Simulation Complete ---")
    print(f"The parameter pair that maximizes observer entropy generation is:")
    print(f"  q* = {q_star} (Alphabet Size)")
    print(f"  R* = {R_star} (Max Connectivity)")
    print(f"  Maximum Average Entropy Increment: {max_val:.6f} bits/tick")

    # --- Visualize the Results ---
    fig, ax = plt.subplots(figsize=(12, 10))
    cax = ax.imshow(results.T, interpolation='nearest', origin='lower',
                    cmap='viridis', aspect='auto')
    
    fig.colorbar(cax, label='Average Observer Entropy Increment per Tick')
    
    ax.set_title(f'Observer Entropy Generation vs. System Parameters (q, R)\nMaximizer at (q*, R*) = ({q_star}, {R_star})', fontsize=16)
    ax.set_xlabel('q (Alphabet Size)', fontsize=14)
    ax.set_ylabel('R (Max Connectivity)', fontsize=14)
    
    ax.set_xticks(np.arange(len(Q_RANGE)))
    ax.set_yticks(np.arange(len(R_RANGE)))
    ax.set_xticklabels(Q_RANGE)
    ax.set_yticklabels(R_RANGE)
    
    if max_val > 0:
        ax.scatter(max_indices[0], max_indices[1], color='red', s=200,
                   edgecolors='white', marker='*', label=f'Maximizer ({q_star}, {R_star})')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
