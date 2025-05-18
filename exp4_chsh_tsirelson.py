# exp4_chsh_numerical_validation_final.py

import itertools
import cmath
import math

# Precompute the 8th roots of unity
roots = [cmath.exp(1j * 2 * math.pi * k / 8) for k in range(8)]

# Causal-site parent structure for CHSH
parents = {
    'H1': [], 'H2': [],        # Hidden seeds
    'M': ['H1', 'H2'],         # Mixing node
    'R_A0': [], 'R_A1': [],    # Rotation constants
    'R_B0': [], 'R_B1': [],
    'A0': ['M', 'R_A0'],       # Measurement nodes
    'A1': ['M', 'R_A1'],
    'B0': ['M', 'R_B0'],
    'B1': ['M', 'R_B1'],
}

# Topological evaluation order
order = ['H1', 'H2', 'M',
         'R_A0', 'R_A1', 'R_B0', 'R_B1',
         'A0', 'A1', 'B0', 'B1']

# Fixed rotation-phase tags
rotation_phases = {
    'R_A0': roots[0],  # 0°
    'R_A1': roots[2],  # 90°
    'R_B0': roots[1],  # 45°
    'R_B1': roots[7],  # -45°
}

# Fusion operation: complex multiplication
def fuse(phases):
    prod = 1+0j
    for p in phases:
        prod *= p
    return prod

# Accumulators for Real-part correlator sums
sum_E = {('A0','B0'): 0.0, ('A0','B1'): 0.0,
         ('A1','B0'): 0.0, ('A1','B1'): 0.0}
total = 0

# Enumerate over all hidden seed assignments
for h1_idx, h2_idx in itertools.product(range(8), repeat=2):
    # Initialize tags for hidden seeds and rotation constants
    tags = {
        'H1': roots[h1_idx],
        'H2': roots[h2_idx],
        **rotation_phases
    }
    # Propagate through the DAG
    for node in order:
        if node not in tags:
            tags[node] = fuse([tags[p] for p in parents[node]])
    # Accumulate Re[tag_A * conj(tag_B)] for each pair
    for (a, b) in sum_E:
        sum_E[(a, b)] += (tags[a] * tags[b].conjugate()).real
    total += 1

# Compute expectation values
E = { pair: sum_E[pair] / total for pair in sum_E }

# Compute CHSH S
S = E[('A0','B0')] + E[('A0','B1')] + E[('A1','B0')] - E[('A1','B1')]

# Output and verify
print("Correlators E(a,b):", {k: f"{v:.6f}" for k, v in E.items()})
print("CHSH S value:     ", f"{S:.6f}")
print("Tsirelson bound:   ", f"{2 * math.sqrt(2):.6f}")

assert math.isclose(S, 2 * math.sqrt(2), rel_tol=1e-9), "S does not match 2√2"

