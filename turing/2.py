import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import convolve2d
import time # To add timestamp or variation

# --- Simulation Parameters ---
N = 200
Du = 0.16
Dv = 0.08
# Parameters chosen for dynamic patterns ('U-Skate' like)
# Interpreted as defining the 'rules of the model', akin to choosing
# the specific equations or field in an algebraic geometry context.
f = 0.055
k = 0.062
dt = 1.0
simulation_steps = 5000
display_interval = 20

# --- Initialization ---
# Seeding the instability: Inspired by structured randomness,
# metaphorically linking to initial conditions in different 'models'.
U = np.ones((N, N))
V = np.zeros((N, N))

# --- More Structured Initial Perturbation ---
# Instead of a simple blob, let's use a pattern mixing locality and structure.
# This is still heuristic, not a direct AG implementation.
seed_size = 20 # Size of the central noisy area
center_x, center_y = N // 2, N // 2
x_start, x_end = center_x - seed_size // 2, center_x + seed_size // 2
y_start, y_end = center_y - seed_size // 2, center_y + seed_size // 2

# Add noise in a central square
noise_patch = np.random.rand(seed_size, seed_size)
U[y_start:y_end, x_start:x_end] = 0.50 + noise_patch * 0.1
V[y_start:y_end, x_start:x_end] = 0.25 + (1.0-noise_patch) * 0.1 # Complementary noise

# Add a few smaller, sparse 'seeds' - "désir léger, sans racine"
np.random.seed(int(time.time())) # Use time for some variation
for _ in range(10):
    px, py = np.random.randint(0, N, 2)
    pr = 2 # Small radius
    py_s, py_e = max(0, py-pr), min(N, py+pr)
    px_s, px_e = max(0, px-pr), min(N, px+pr)
    if py_e > py_s and px_e > px_s:
        U[py_s:py_e, px_s:px_e] = 0.7 + np.random.rand(py_e-py_s, px_e-px_s)*0.1
        V[py_s:py_e, px_s:px_e] = 0.1 + np.random.rand(py_e-py_s, px_e-px_s)*0.1

# Slight global noise remains important for allowing instability anywhere
U += np.random.rand(N, N) * 0.005
V += np.random.rand(N, N) * 0.005
U = np.clip(U, 0, 1) # Ensure bounds
V = np.clip(V, 0, 1)

# --- Laplacian Kernel ---
laplacian_kernel = np.array([[0.05, 0.2, 0.05],
                             [0.2, -1.0, 0.2],
                             [0.05, 0.2, 0.05]])

def discrete_laplacian(grid):
    return convolve2d(grid, laplacian_kernel, mode='same', boundary='wrap')

# --- Simulation Step ---
def update(frameNum, img, U, V):
    global f, k, Du, Dv, dt
    # The core update loop: Represents the 'morphism' or evolution rule
    # applied repeatedly. Numerical instability (handled carefully) allows
    # the underlying diffusion-driven (Turing) instability to manifest.
    for _ in range(display_interval):
        Lu = discrete_laplacian(U)
        Lv = discrete_laplacian(V)
        reaction = U * V**2
        U_new = U + (Du * Lu - reaction + f * (1 - U)) * dt
        V_new = V + (Dv * Lv + reaction - (f + k) * V) * dt
        U = np.maximum(0, U_new) # Using maximum instead of clip here
        V = np.maximum(0, V_new)
        # Clipping can sometimes suppress instabilities unnaturally
        # U = np.clip(U_new, 0, 1)
        # V = np.clip(V_new, 0, 1)


    # Update visualization - V often shows intricate patterns
    img.set_data(V)
    # Using 'magma' or 'plasma' for potential 'éclat d'un moment fuyant'
    img.set_cmap('magma')
    # Dynamic range adjustment helps visualize structures
    vmin, vmax = np.percentile(V, [1, 99]) # Avoid extreme outliers dominating scale
    if vmax <= vmin: vmax = vmin + 1e-4 # Handle flat case
    img.set_clim(vmin=vmin, vmax=vmax)

    return img,

# --- Visualization Setup ---
fig, ax = plt.subplots(figsize=(8, 8))
img = ax.imshow(V, cmap='magma', interpolation='bilinear', animated=True)
ax.set_xticks([])
ax.set_yticks([])

poem_text = """
Le printemps réveille en moi
un désir léger, sans racine ni chemin,
comme une âme bohémienne vagabonde,
cherchant l’instant, sans destin.

Les mots se dissolvent au matin,
les projets se perdent en l’air du temps,
et je reste, incertain,
face à l’éclat d’un moment fuyant.

Tel un cheval libre apprivoisé,
vendu aux ombres d’un conte désenchanté,
la raison finit par l’enchaîner
à une tristesse de plus, à jamais.
"""
fig.text(0.02, 0.02, poem_text, ha='left', va='bottom', fontsize=8, style='italic', color='grey')
fig.suptitle("Turing Diffusion: Instability & Structure\n(Inspired by AG/Field Concepts & Poem)", fontsize=12)
plt.tight_layout(rect=[0, 0.2, 1, 0.95])

# --- Run Animation ---
num_frames = simulation_steps // display_interval
ani = animation.FuncAnimation(fig, update, frames=num_frames,
                              fargs=(img, U, V), interval=20, blit=True)
plt.show()

print("Simulation complete.")
print(f"Parameters (defining the 'model'): f={f}, k={k}, Du={Du}, Dv={Dv}")
print("Interpretive Notes:")
print("- Initial structured noise acts as 'seeds' - 'désir léger'.")
print("- Reaction-diffusion dynamics = 'morphism' transforming the state.")
print("- Turing instability amplifies small variations ('irrational' deviations from uniformity).")
print("- This instability, analogous to complexity seen when relating different mathematical structures (e.g., via AG morphisms over varying fields), drives pattern formation ('moment fuyant').")
print("- The fixed PDEs are the 'reason' constraining the evolution ('enchaîner') into observable patterns.")
