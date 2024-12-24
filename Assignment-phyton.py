#hashila assignment
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import root_scalar # type: ignore

# Constants and Parameters
h_bar = 1   # Reduced Planck constant
mass = 1    # Particle mass
alpha_param = 1
lambda_param = 4

# Define the range and step for x
x_min, x_max, step = -10, 10, 0.05
x_values = np.arange(x_min, x_max + step, step)

# Define the potential function
def potential(x):
    scaling_factor = (h_bar**2 / (2 * mass)) * alpha_param**2 * lambda_param * (lambda_param - 1)
    return scaling_factor * (0.5 - 1 / (np.cosh(alpha_param * x)**2))

# Plot the potential function with a new color
plt.plot(x_values, potential(x_values), color="purple", label="Potential V(x)")
plt.xlabel("x")
plt.ylabel("Energy")
plt.xlim(-5, 5)
plt.title("Potential Function for 1D Schrödinger Equation")
plt.grid()
plt.legend()
plt.show()

# Numerov method for solving the Schrödinger equation
def numerov_method(psi_start, psi_next, energy, x_vals, step_size):
    num_points = len(x_vals)
    psi_vals = np.zeros(num_points)
    psi_vals[0], psi_vals[1] = psi_start, psi_next

    # Define the effective potential function
    effective_potential = lambda x: 2 * mass / h_bar**2 * (energy - potential(x))

    for i in range(1, num_points - 1):
        k_0 = effective_potential(x_vals[i - 1])
        k_1 = effective_potential(x_vals[i])
        k_2 = effective_potential(x_vals[i + 1])

        psi_vals[i + 1] = (
            2 * (1 - 5 * step_size**2 * k_1 / 12) * psi_vals[i]
            - (1 + step_size**2 * k_0 / 12) * psi_vals[i - 1]
        ) / (1 + step_size**2 * k_2 / 12)

    return psi_vals

# Matching condition at turning points
def compute_matching_condition(energy, x_vals, step_size):
    psi_left = numerov_method(0.0, 1e-5, energy, x_vals, step_size)
    psi_right = numerov_method(0.0, 1e-5, energy, x_vals[::-1], step_size)[::-1]
    mid_index = len(x_vals) // 2
    left_ratio = (psi_left[mid_index + 1] - psi_left[mid_index - 1]) / (2 * step_size * psi_left[mid_index])
    right_ratio = (psi_right[mid_index + 1] - psi_right[mid_index - 1]) / (2 * step_size * psi_right[mid_index])
    return left_ratio - right_ratio

# Finding eigenvalues numerically
def determine_eigenvalues(x_vals, step_size, levels=3):
    eigenvalues = []
    energy_start, energy_end, energy_increment = -2, 0, 2

    for level in range(levels):
        energy_range = np.linspace(energy_start, energy_end, 100)
        for e1, e2 in zip(energy_range[:-1], energy_range[1:]):
            if compute_matching_condition(e1, x_vals, step_size) * compute_matching_condition(e2, x_vals, step_size) < 0:
                result = root_scalar(compute_matching_condition, args=(x_vals, step_size), bracket=[e1, e2], method='brentq')
                eigenvalues.append(result.root)
                break
        energy_start += energy_increment
        energy_end += energy_increment

    return eigenvalues

# Calculate and display eigenvalues
num_levels = 3
eigen_vals = determine_eigenvalues(x_values, step, levels=num_levels)
print("Computed Eigenvalues:")
for idx, eigen in enumerate(eigen_vals):
    print(f"Level {idx}: E = {eigen:.6f}")

# Plot the potential and corresponding wavefunctions with updated colors
plt.plot(x_values, potential(x_values), label="Potential V(x)", color='purple')
for idx, eigen in enumerate(eigen_vals):
    psi_wave = numerov_method(0.0, 1e-5, eigen, x_values, step)
    normalized_wave = psi_wave / np.max(np.abs(psi_wave)) + eigen
    plt.plot(x_values, normalized_wave, label=f"Eigenvalue {idx} (E={eigen:.3f})", color=f"C{idx + 1}")  # Unique color for each wavefunction
plt.xlabel("x")
plt.ylabel("Energy / Wavefunction")
plt.title("Wavefunctions for 1D Schrödinger Equation")
plt.legend()
plt.grid()
plt.show()
