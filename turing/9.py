import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.ndimage import convolve

# Grid and simulation parameters
W, H = 256, 256
Du, Dv = 0.14, 0.06
phi = (1 + np.sqrt(5)) / 2  # golden ratio
irr = np.sqrt(2)            # irrational number for non-commutative effects

# Define elliptic curve-based algebraic variety V: y^2 = x^3 - x + phi
def generate_field_morphism(grid_x, grid_y):
    X, Y = np.meshgrid(grid_x, grid_y)
    E = Y**2 - (X**3 - X + phi)
    # Explicit field morphism: pushforward the error into instability geometry
    morphism = np.sin(irr * E) * np.cos(phi * E)
    # Normalize to be a positive modulation field
    return (morphism - morphism.min()) / (morphism.max() - morphism.min()) + 0.2

# Anisotropic diffusion kernels (Laplacian variants)
laplace_kernel = np.array([[0.05, 0.2 , 0.05],
                           [0.2 , -1.0, 0.2 ],
                           [0.05, 0.2 , 0.05]])

# Initialize U and V with small random perturbation
U = np.ones((W, H))
V = np.zeros((W, H))
U += 0.05 * np.random.randn(W, H)
V += 0.05 * np.random.randn(W, H)

# Seed perturbation
r = 10
U[W//2-r:W//2+r, H//2-r:H//2+r] = 0.50
V[W//2-r:W//2+r, H//2-r:H//2+r] = 0.25

# Generate morphism field from algebraic variety
grid_x = np.linspace(-2, 2, W)
grid_y = np.linspace(-2, 2, H)
field_morphism = generate_field_morphism(grid_x, grid_y)

# Reaction–diffusion update using morphism-influenced anisotropy
def update_reaction_diffusion(U, V, morphism):
    Lu = convolve(U, laplace_kernel, mode='reflect')
    Lv = convolve(V, laplace_kernel, mode='reflect')

    # Reaction terms (classic Gray-Scott)
    reaction = U * V**2
    U += (Du * Lu - reaction + 0.045 * (1 - U)) * morphism
    V += (Dv * Lv + reaction - (0.065 + 0.005 * np.sin(phi)) * V) * morphism
    return U, V

# Setup visualization
fig, ax = plt.subplots()
img = ax.imshow(U, cmap='inferno', interpolation='bilinear')
plt.axis('off')

# Animation loop
def animate(frame):
    global U, V
    modulation = 1.0 + 0.05 * np.sin(frame * 0.05)
    morph = field_morphism * modulation
    U, V = update_reaction_diffusion(U, V, morph)
    img.set_data(U)
    return [img]

ani = animation.FuncAnimation(fig, animate, frames=600, interval=30, blit=True)
plt.show()
