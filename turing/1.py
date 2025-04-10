import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import convolve2d

# --- Simulation Parameters Inspired by Poem ---

# Grid size (Larger grid allows for more complex structures)
N = 200

# Diffusion rates (Dv > Du is typical for Turing patterns)
Du = 0.16
Dv = 0.08

# Reaction rates (f, k) - Chosen for dynamic patterns ("U-Skate World" like)
# These parameters often lead to evolving, non-static patterns ("moment fuyant")
# representing the instability studied by Zeldovich, emerging from randomness.
f = 0.055 # Feed rate - "léger désir"
k = 0.062 # Kill rate - balance allowing structure but also transience

# Time step and simulation duration
dt = 1.0 # Time step size
# simulation_steps = 10000 # Total steps for a longer evolution
# display_interval = 50     # Update display every N steps
simulation_steps = 5000
display_interval = 20

# --- Initialization ---
# Start with a near-uniform state ("sans racine") perturbed randomly ("âme bohémienne")
# This reflects the emergence from simplicity/randomness (Zeldovich instability concept)
U = np.ones((N, N))
V = np.zeros((N, N))

# Add localized perturbations - the "seed" of change
r = 10 # Radius of perturbation
center_x, center_y = N // 2, N // 2
y, x = np.ogrid[-center_y:N-center_y, -center_x:N-center_x]
mask = x*x + y*y <= r*r

# Initial perturbation: Small area with high V, low U
U[mask] = 0.50 + np.random.rand(N,N)[mask] * 0.1 # Add some noise within the seed
V[mask] = 0.25 + np.random.rand(N,N)[mask] * 0.1

# Add very slight noise everywhere else to represent the "incertain" background
U += np.random.rand(N, N) * 0.01
V += np.random.rand(N, N) * 0.01

# --- Laplacian Kernel for Diffusion ---
# Using convolution for periodic boundary conditions
# The Laplacian represents interaction/diffusion across the 'field'
laplacian_kernel = np.array([[0.05, 0.2, 0.05],
                             [0.2, -1.0, 0.2],
                             [0.05, 0.2, 0.05]])

def discrete_laplacian(grid):
    # Apply convolution with periodic boundary conditions
    return convolve2d(grid, laplacian_kernel, mode='same', boundary='wrap')

# --- Simulation Step ---
def update(frameNum, img, U, V):
    global f, k, Du, Dv, dt # Allow access to global parameters

    # Run multiple simulation steps per display frame for smoother animation
    for _ in range(display_interval):
        # Calculate Laplacians
        Lu = discrete_laplacian(U)
        Lv = discrete_laplacian(V)

        # Reaction term
        reaction = U * V**2

        # Update concentrations using Euler method
        U_new = U + (Du * Lu - reaction + f * (1 - U)) * dt
        V_new = V + (Dv * Lv + reaction - (f + k) * V) * dt

        # Ensure concentrations don't go negative (optional, but good practice)
        U = np.maximum(0, U_new)
        V = np.maximum(0, V_new)

    # Update the image display (visualizing V concentration)
    img.set_data(V)
    # Choose a colormap reflecting the poem's mood - 'viridis', 'plasma', 'cividis', 'Blues' or 'Greys'
    # 'cividis' is designed to be perceptually uniform and print-friendly, might suit the subtle mood.
    # 'Blues' or 'Greys' for a more melancholic feel ("ombres", "tristesse")
    img.set_cmap('cividis')
    img.set_clim(vmin=V.min(), vmax=V.max()) # Adjust color limits dynamically

    # Optional: Add title showing time step
    # plt.title(f"Turing Pattern (Gray-Scott) - Step {frameNum * display_interval}")

    return img,

# --- Visualization Setup ---
fig, ax = plt.subplots(figsize=(8, 8))

# Displaying V - often shows more intricate patterns
# Initial plot - using 'cividis' colormap for a slightly subdued but detailed look
img = ax.imshow(V, cmap='cividis', interpolation='bilinear', animated=True)

# Hide axes for cleaner look
ax.set_xticks([])
ax.set_yticks([])

# Add the poem text to the figure
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
# Position the text - adjust coordinates as needed
fig.text(0.02, 0.02, poem_text, ha='left', va='bottom', fontsize=8, style='italic', color='grey')
fig.suptitle("Turing Diffusion Inspired by 'Le printemps réveille en moi'\n(Gray-Scott Model)", fontsize=12)
plt.tight_layout(rect=[0, 0.2, 1, 0.95]) # Adjust layout to make space for text/title

# --- Run Animation ---
# Calculate number of frames needed
num_frames = simulation_steps // display_interval

# Create animation
ani = animation.FuncAnimation(fig, update, frames=num_frames,
                              fargs=(img, U, V), interval=20, blit=True)

# To save the animation (requires ffmpeg or imagemagick):
# ani.save('turing_poem_animation.mp4', writer='ffmpeg', fps=30)
# ani.save('turing_poem_animation.gif', writer='imagemagick', fps=20)

plt.show()

print("Simulation complete.")
print(f"Final parameters used: f={f}, k={k}, Du={Du}, Dv={Dv}")
print("Interpretation Notes:")
print("- Initial random seed mimics 'désir léger, sans racine'.")
print("- Dynamic patterns represent 'moment fuyant', 'âme vagabonde'.")
print("- Emergence from near-uniformity relates to Zeldovich instability concepts.")
print("- Complex geometric forms are a metaphorical nod to the complexity studied in algebraic geometry.")
print("- The chosen colormap ('cividis', or try 'Blues'/'Greys') hints at the poem's mood.")
