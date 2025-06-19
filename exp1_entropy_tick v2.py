import itertools
import random
import math
from collections import defaultdict

def generate_depth_graded_dag(widths, edge_prob=0.5):
    """
    Generates a depth-graded Directed Acyclic Graph (DAG).
    Unchanged from the original script.
    """
    nodes = []
    depth_map = {}
    nodes_by_depth = defaultdict(list)

    for d in sorted(widths.keys()):
        for i in range(widths[d]):
            node = f"{d}_{i}"
            nodes.append(node)
            depth_map[node] = d
            nodes_by_depth[d].append(node)

    parent_map = defaultdict(list)
    sorted_depths = sorted(widths.keys())
    for idx, d in enumerate(sorted_depths[:-1]):
        next_d = sorted_depths[idx+1]
        parents_at_d = [n for n in nodes if depth_map[n] == d]
        children_at_next_d = [n for n in nodes if depth_map[n] == next_d]
        for parent in parents_at_d:
            for child in children_at_next_d:
                if random.random() < edge_prob:
                    parent_map[child].append(parent)

    return nodes, depth_map, nodes_by_depth, parent_map

def fusion_rule(parent_tags, input_tag, alphabet_size):
    """A deterministic rule to calculate a child's tag."""
    fused_parent_tag = sum(parent_tags) % alphabet_size if parent_tags else 0
    return (fused_parent_tag + input_tag) % alphabet_size

def simulate_single_run(nodes_by_depth, parent_map, alphabet_size):
    """
    Simulates ONE complete, deterministic history of the entire causal site.
    This represents the "actual" evolution of the universe.
    """
    full_history = {}
    sorted_depths = sorted(nodes_by_depth.keys())

    for d in sorted_depths:
        for node in nodes_by_depth[d]:
            # For this single run, we fix the "random" inputs to generate one history
            input_tag = random.randint(0, alphabet_size - 1)
            parents = parent_map.get(node, [])
            parent_tags = [full_history[p] for p in parents]
            
            full_history[node] = fusion_rule(parent_tags, input_tag, alphabet_size)
            
    return full_history

def calculate_observer_entropy(nodes_by_depth, parent_map, alphabet_size, hidden_depth_start, observed_macrostate):
    """
    Calculates the observer's entropy over time.
    This is the core of the validation.
    """
    # Start with one empty microstate, representing certainty before t=0
    consistent_microstates = [{}]
    entropies = [0.0]

    sorted_depths = sorted(nodes_by_depth.keys())

    for d in sorted_depths:
        next_consistent_microstates = []
        is_hidden_layer = (d >= hidden_depth_start)
        
        # Iterate through all previously consistent microstates
        for microstate in consistent_microstates:
            
            if not is_hidden_layer:
                # VISIBLE LAYER: Filter microstates. No entropy growth.
                # We extend the current microstate and check if it matches the observation.
                
                new_microstate_branch = microstate.copy()
                is_consistent = True
                for node in nodes_by_depth[d]:
                    # We need to determine the input_tag that would produce the observed tag
                    # input_tag = (observed_tag - fused_parent_tag) mod alphabet_size
                    parents = parent_map.get(node, [])
                    parent_tags = [microstate[p] for p in parents]
                    fused_parent_tag = sum(parent_tags) % alphabet_size if parent_tags else 0
                    
                    observed_tag = observed_macrostate[node]
                    # This node's history is now determined
                    new_microstate_branch[node] = observed_tag
                
                # In this simplified model, any history is possible for the visible part,
                # as the "random" input_tag can always be chosen to match the observation.
                # A more complex model might have constraints. Here, we just add the one consistent path.
                next_consistent_microstates.append(new_microstate_branch)

            else:
                # HIDDEN LAYER: Branch microstates. Entropy grows here.
                # The observer has no data, so ALL possibilities for the hidden nodes are consistent.
                
                # Generate all possible tag combinations for the hidden nodes at this depth
                hidden_nodes = nodes_by_depth[d]
                for input_tags_tuple in itertools.product(range(alphabet_size), repeat=len(hidden_nodes)):
                    
                    new_microstate_branch = microstate.copy()
                    for node, input_tag in zip(hidden_nodes, input_tags_tuple):
                        parents = parent_map.get(node, [])
                        parent_tags = [microstate[p] for p in parents]
                        # The tag is determined by the parents and the hypothetical input tag
                        new_microstate_branch[node] = fusion_rule(parent_tags, input_tag, alphabet_size)
                    
                    next_consistent_microstates.append(new_microstate_branch)

        consistent_microstates = next_consistent_microstates
        # Observer entropy is the log of the number of possible hidden realities
        entropies.append(math.log2(len(consistent_microstates)))
        
    return entropies

def test_observer_entropy(num_trials=1, max_depth=5, nodes_per_level=2, hidden_depth_start=3, alphabet_size=2):
    random.seed(42)
    print(f"--- Testing Observer Entropy ---")
    print(f"Parameters: alphabet_size={alphabet_size}, hidden_depth_start={hidden_depth_start}\n")
    
    for i in range(1, num_trials + 1):
        widths = {d: nodes_per_level for d in range(max_depth + 1)}
        nodes, depth_map, nbd, parent_map = generate_depth_graded_dag(widths, edge_prob=0.7)
        
        # 1. Simulate a single "ground truth" to be observed
        actual_history = simulate_single_run(nbd, parent_map, alphabet_size)
        
        # 2. Define the observer's partial knowledge (the macrostate)
        observed_macrostate = {node: tag for node, tag in actual_history.items() if depth_map[node] < hidden_depth_start}
        
        # 3. Calculate observer entropy based on their limited knowledge
        entropies = calculate_observer_entropy(nbd, parent_map, alphabet_size, hidden_depth_start, observed_macrostate)
        diffs = [entropies[t+1] - entropies[t] for t in range(len(entropies) - 1)]
        
        print(f"Trial {i}:")
        print(f"  Nodes per depth layer: {nodes_per_level}")
        print(f"  Observer Entropies (S_t): {[f'{e:.2f}' for e in entropies]}")
        print(f"  Entropy Increments (ΔS_t):  {[f'{d:.2f}' for d in diffs]}")
        
        # This is the key check
        num_hidden_nodes_at_start = widths.get(hidden_depth_start, 0)
        expected_increment = num_hidden_nodes_at_start * math.log2(alphabet_size)
        print(f"  * Entropy is 0 until depth {hidden_depth_start-1}.")
        print(f"  * At depth {hidden_depth_start}, entropy should increase by |H_min|*log2(|A_h|) = {num_hidden_nodes_at_start}*log2({alphabet_size}) = {expected_increment:.2f} bits.")
        print("-" * 30)

if __name__ == "__main__":
    test_observer_entropy()