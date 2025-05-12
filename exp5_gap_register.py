import numpy as np
import pandas as pd
from numpy.linalg import eigvals

# Function to build the deterministic shift-register Markov matrix
def build_shift_register_matrix(a, L):
    n_states = a**L
    P = np.zeros((n_states, n_states))
    for i in range(n_states):
        for symbol in range(a):
            j = ((i * a) % (a**L)) + symbol
            P[i, j] += 1 / a
    return P

# Parametrize over alphabet sizes, register lengths, and block sizes (RG steps)
results = []
for a in [2]:  # You can extend to other alphabet sizes
    for L in [3, 4, 5]:  # Various register lengths
        P = build_shift_register_matrix(a, L)
        for k in [2, 3, 4]:  # Block-dilation sizes
            Rk = np.linalg.matrix_power(P, k)  # RG operator via k-step composition
            eigs = eigvals(Rk)
            mags = np.abs(eigs)
            mags.sort()
            lambda2 = mags[-2]  # second-largest eigenvalue magnitude
            results.append({
                'Alphabet Size (a)': a,
                'Register Length (L)': L,
                'Block Size (k)': k,
                'Second-Largest |λ₂|': lambda2
            })

# Display the results
import ace_tools as tools; tools.display_dataframe_to_user("Spectral Gap under RG Blocks", pd.DataFrame(results))

# Verify the theoretical bound λ₂ ≤ 1/4
for row in results:
    assert row['Second-Largest |λ₂|'] <= 0.25 + 1e-8, \
        f"Spectral gap bound violated for a={row['Alphabet Size (a)']}, L={row['Register Length (L)']}, k={row['Block Size (k)']}"

