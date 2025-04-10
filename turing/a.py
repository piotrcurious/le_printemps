import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import convolve
import sympy as sp

# Global simulation parameters
W, H = 256, 256           # Grid dimensions
Du, Dv = 0.14, 0.06        # Diffusion coefficients for U and V
phi_val = (1 + np.sqrt(5)) / 2  # Golden ratio: an irrational constant
irr_val = np.sqrt(2)           # Another irrational number for deformation

# -------------------------------
# Symbolic definition using Sympy
# -------------------------------

# Define symbolic variables for the plane.
x, y = sp.symbols('x y')

# Define the elliptic curve (algebraic variety) as:
#    y^2 = x^3 - x + phi_val
# Rearranged as: y^2 - (x^3 - x + phi_val) = 0.
elliptic_poly = y**2 - (x**3 - x + phi_val)

# Define a field morphism from the algebraic variety:
# We use sinusoidal functions, deformed by the irrational numbers,
# to pushforward the error (the left-hand side of the polynomial).
morphism_expr = sp.sin(irr_val * elliptic_poly) * sp.cos(phi_val * elliptic_poly)

# Convert the symbolic morphism to a numerical function.
# This function will be evaluated over a grid.
morphism_func = sp.lambdify((x, y), morphism_expr, "numpy")

# -------------------------------
# Generate the modulation field
# -------------------------------

# Create a coordinate grid for evaluation.
grid_x = np.linspace(-2, 2, W)
grid_y = np.linspace(-2, 2, H)
X, Y = np.meshgrid(grid_x, grid_y)

# Evaluate the symbolic morphism function over the grid.
field_morphism = morphism_func(X, Y)

# Normalize the field_morphism so that it modulates diffusion meaningfully.
field_morphism = (field_morphism - field_morphism.min()) / (field_morphism.max() - field_morphism.min())
field_morphism = field_morphism * 0.8 + 0.2  # Scale to range [0.2, 1.0]

# -------------------------------
# Setup the Reaction–Diffusion System
# -------------------------------

# Initialize U and V fields with a small random perturbation.
U = np.ones((W, H)) + 0.05 * np.random.randn(W, H)
V = np.zeros((W, H)) + 0.05 * np.random.randn(W, H)

# Seed a localized perturbation in the middle.
r = 10
U[W//2 - r:W//2 + r, H//2 - r:H//2 + r] = 0.50
V[W//2 - r:W//2 + r, H//2 - r:H//2 + r] = 0.25

# Anisotropic Laplacian kernel (approximates the diffusion term)
laplace_kernel = np.array([[0.05, 0.2, 0.05],
                           [0.2, -1.0, 0.2],
                           [0.05, 0.2, 0.05]])

def update_reaction_diffusion(U, V, morphism):
    """Update the fields U and V using a reaction-diffusion equation modulated
       by the algebraic geometry inspired field morphism.
    """
    Lu = convolve(U, laplace_kernel, mode='reflect')
    Lv = convolve(V, laplace_kernel, mode='reflect')
    reaction = U * V**2
    U += (Du * Lu - reaction + 0.045 * (1 - U)) * morphism
    V += (Dv * Lv + reaction - (0.065 + 0.005 * np.sin(phi_val)) * V) * morphism
    return U, V

# -------------------------------
# Visualization using Matplotlib
# -------------------------------

fig, ax = plt.subplots()
img = ax.imshow(U, cmap='inferno', interpolation='bilinear')
plt.axis('off')

def animate(frame):
    global U, V
    # Introduce an additional poetic modulation to evoke the transient emotional arc.
    modulation = 1.0 + 0.05 * np.sin(frame * 0.05)
    morph = field_morphism * modulation
    U, V = update_reaction_diffusion(U, V, morph)
    img.set_data(U)
    return [img]

ani = animation.FuncAnimation(fig, animate, frames=600, interval=30, blit=True)
plt.show()
