import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def create_random_couplings(N, J):
    """create random couplings"""
    num_couplings = N * (N-1) * (N-2) * (N-3) // 24
    print(f"create {num_couplings} couplings")
    return np.random.normal(0, J/np.sqrt(N**3), size=num_couplings)

def create_hamiltonian(N, couplings):
    """create hamiltonian"""
    dim = 2**(N//2)  # Majorana fermion number is twice of normal fermion
    H = np.zeros((dim, dim), dtype=complex)
    
    coupling_idx = 0
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                for l in range(k+1, N):
                    J_ijkl = couplings[coupling_idx]
                    coupling_idx += 1
                    
                    # use Majorana fermion's Jordan-Wigner transform
                    for state in range(dim):
                        new_state = state
                        sign = 1
                        for m in [i//2, j//2, k//2, l//2]:
                            if m < N//2:
                                if state & (1 << m):
                                    sign *= -1
                                else:
                                    new_state ^= (1 << m)
                        
                        if i%2 == j%2 == k%2 == l%2:
                            sign *= 1j
                        
                        H[state, new_state] += J_ijkl * sign

    return H + H.conj().T  # ensure hamiltonian is hermitian

def solve_syk(N, J, return_eigenvectors=False):
    """solve syk model"""
    couplings = create_random_couplings(N, J)
    H = create_hamiltonian(N, couplings)
    
    if return_eigenvectors:
        eigenvalues, eigenvectors = eigh(H)
        return eigenvalues, eigenvectors
    else:
        eigenvalues = eigh(H, eigvals_only=True)
        return eigenvalues

# example
N = 20  # fermion number
J = 1.0  # coupling strength

try:
    eigenvalues = solve_syk(N, J)
    print("energy eigenvalues:")
    print(eigenvalues)

    # calculate energy spectrum statistics
    mean_level_spacing = np.mean(np.diff(sorted(eigenvalues)))
    print(f"mean level spacing: {mean_level_spacing}")

    # calculate energy spectrum standard deviation
    energy_std = np.std(eigenvalues)
    print(f"energy spectrum standard deviation: {energy_std}")
except Exception as e:
    print(f"error: {e}")

# 計算無量綱化的能級間距
spacings = np.diff(sorted(eigenvalues))
normalized_spacings = spacings / np.mean(spacings)

# 計算r統計量
r_values = np.minimum(normalized_spacings[:-1], normalized_spacings[1:]) / np.maximum(normalized_spacings[:-1], normalized_spacings[1:])
r_mean = np.mean(r_values)
print(f"mean r value: {r_mean}")

# 繪製能級間距分佈
plt.figure(figsize=(10, 6))
kde = gaussian_kde(normalized_spacings)
x = np.linspace(0, 3, 100)
plt.plot(x, kde(x), label='numerical result')
plt.plot(x, np.pi/2 * x * np.exp(-np.pi/4 * x**2), label='Wigner-Dyson (GOE)')
plt.xlabel('normalized energy level spacing')
plt.ylabel('probability density')
plt.legend()
plt.title('SYK model energy level spacing distribution')
plt.show()