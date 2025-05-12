import numpy as np
import itertools

def build_shift_register_koopman(n_bits=4):
    """
    Build the Koopman operator P for the n_bits shift-register model
    with binary input averaging, as per Paper II, Appendix D.3.
    Returns:
      P: (2^n_bits x 2^n_bits) column-stochastic matrix
    """
    hidden_alphabet = [0, 1]
    states = list(itertools.product(hidden_alphabet, repeat=n_bits))
    S = len(states)
    state_index = {state: idx for idx, state in enumerate(states)}

    # Define update T: shift right and prepend new input bit b
    def T(state, b):
        return (b,) + state[:-1]

    # Build P by averaging over input bits
    P = np.zeros((S, S), dtype=float)
    for i, state in enumerate(states):
        for b in hidden_alphabet:
            new_state = T(state, b)
            j = state_index[new_state]
            P[j, i] += 1.0 / len(hidden_alphabet)
    return P

def spectral_analysis(P, tol=1e-6):
    """
    Compute all eigenvalues of P and return those with |λ| ≈ 1.
    """
    eigvals = np.linalg.eigvals(P)
    peripheral = [ev for ev in eigvals if abs(abs(ev) - 1) < tol]
    return eigvals, peripheral

def cesaro_average(P, v0, M=1000):
    """
    Compute the Cesàro average (1/M) ∑_{k=1}^M P^k v0.
    """
    v = v0.copy()
    avg = np.zeros_like(v, dtype=float)
    for _ in range(M):
        v = P @ v
        avg += v
    avg /= M
    return avg

def main():
    # Build Koopman operator
    n_bits = 4
    P = build_shift_register_koopman(n_bits)

    # Spectral analysis
    eigvals, peripheral = spectral_analysis(P)
    
    # Initial distribution: delta at state 0
    S = P.shape[0]
    v0 = np.zeros(S, dtype=float)
    v0[0] = 1.0

    # Cesàro average convergence
    M = 1000
    avg = cesaro_average(P, v0, M)
    uniform = np.ones(S, dtype=float) / S
    tv_distance = 0.5 * np.sum(np.abs(avg - uniform))

    # Output results
    print("Exp 3: 4-bit shift-register model validation")
    print(f"State-space size: {S}")
    print(f"Peripheral eigenvalues (|λ|≈1): {np.round(peripheral, 8)}")
    print(f"TV distance after M={M} steps: {tv_distance:.6e}")

if __name__ == "__main__":
    main()
