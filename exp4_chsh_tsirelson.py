import numpy as np

# Define the discrete finite phase group: 8th roots of unity
phase_set = np.exp(1j * np.pi * np.arange(8) / 4)

max_S = 0+0j
max_config = None

# Fix phi_a0 = 1 (phase_set[0])
phi_a0 = phase_set[0]

# Exhaustively enumerate phi_a1, phi_b0, phi_b1 from the discrete set
for phi_a1 in phase_set:
    for phi_b0 in phase_set:
        # Precompute correlations E(a0,b0) and E(a1,b0)
        E00 = np.real(phi_a0 * np.conj(phi_b0))
        E10 = np.real(phi_a1 * np.conj(phi_b0))
        for phi_b1 in phase_set:
            E01 = np.real(phi_a0 * np.conj(phi_b1))
            E11 = np.real(phi_a1 * np.conj(phi_b1))
            S = E00 + E01 + E10 - E11
            if S > max_S:
                max_S = S
                max_config = (phi_a1, phi_b0, phi_b1)

# Report results
print(f"Max CHSH from 8th roots enumeration: {max_S:.6f}")
print("Optimal discrete configuration (phi_a1, phi_b0, phi_b1):")
print(max_config)

# Verify against Tsirelson bound exactly
tsirelson = 2 * np.sqrt(2)
assert np.isclose(max_S, tsirelson, atol=1e-6), "Discrete enumeration fails to reach Tsirelson bound!"
