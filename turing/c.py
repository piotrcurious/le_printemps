import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sympy import symbols, lambdify, simplify
from scipy.ndimage import convolve

# ---------------------------
# Define symbolic variables
x, y = symbols('x y')

# ---------------------------
# Define multiple symbolic morphisms (algebraic curves)
# Morphism 1: A classic Weierstrass form with golden ratio modulation
phi_val = (1 + np.sqrt(5)) / 2
f1 = y**2 - (x**3 - x + phi_val)

# Morphism 2: A heart-like curve
f2 = (x**2 + y**2 - 1)**3 - x**2 * y**3

# Morphism 3: A Möbius-style field echo
f3 = (x**2 + y**2)**2 - 4 * x**2 * y**2

# Simplify (optional) and lambdify each symbolic expression for fast evaluation
f1 = simplify(f1)
f2 = simplify(f2)
f3 = simplify(f3)

f1_func = lambdify((x, y), f1, 'numpy')
f2_func = lambdify((x, y), f2, 'numpy')
f3_func = lambdify((x, y), f3, 'numpy')

# ---------------------------
# Define function to build a combined morphism field
def build_combined_morphism(W, H, scale=3.0, weights=(1.0, 1.0, 1.0)):
    """
    Create a combined morphism field using three symbolic morphisms.
    Each morphism is evaluated on a 2D grid and transformed nonlinearly.
    Their weighted sum is then normalized to a modulation field.
    """
    grid_x = np.linspace(-scale, scale, W)
    grid_y = np.linspace(-scale, scale, H)
    X, Y = np.meshgrid(grid_x, grid_y)
    
    # Evaluate each symbolic function on the grid
    E1 = f1_func(X, Y)
    E2 = f2_func(X, Y)
    E3 = f3_func(X, Y)
    
    # Non-linear transformation to introduce oscillatory instability effects.
    morph1 = np.cos(np.sqrt(2) * E1) * np.sin(phi_val * E1)
    morph2 = np.sin(np.sqrt(3) * E2) * np.cos(phi_val * E2)
    morph3 = np.cos(np.sqrt(5) * E3) * np.sin(np.sqrt(3) * E3)
    
    # Weighted combination of the morphisms
    combined = (weights[0] * morph1 + weights[1] * morph2 + weights[2] * morph3)
    
    # Normalize the field to lie roughly in [0.2, 1.2]
    combined_norm = (combined - combined.min()) / (combined.max() - combined.min())
    return combined_norm + 0.2

# ---------------------------
# Reaction–Diffusion parameters
W, H = 256, 256   # Grid dimensions
Du, Dv = 0.14, 0.06 # Diffusion coefficients

# Anisotropic Laplace kernel to approximate spatial derivatives
laplace_kernel = np.array([[0.05, 0.2 , 0.05],
                           [0.2 , -1.0, 0.2 ],
                           [0.05, 0.2 , 0.05]])

# ---------------------------
# Initialize concentrations with slight random perturbations
U = np.ones((W, H)) + 0.01 * np.random.randn(W, H)
V = np.zeros((W, H)) + 0.01 * np.random.randn(W, H)

# Seed a central perturbation to trigger pattern formation
r = 10
U[W//2 - r:W//2 + r, H//2 - r:H//2 + r] = 0.50
V[W//2 - r:W//2 + r, H//2 - r:H//2 + r] = 0.25

# Build the combined morphism field from multiple symbolic morphisms.
# The weights can be adjusted to favor one geometric influence over another.
combined_morphism = build_combined_morphism(W, H, scale=3.0, weights=(1.0, 0.8, 1.2))

# ---------------------------
# Reaction–Diffusion update using the combined morphism field
def update(U, V, morphism):
    # Diffusion approximation using convolution with an anisotropic Laplacian kernel
    Lu = convolve(U, laplace_kernel, mode='reflect')
    Lv = convolve(V, laplace_kernel, mode='reflect')
    
    # Reaction terms (inspired by Gray-Scott type kinetics)
    reaction = U * V**2
    U += (Du * Lu - reaction + 0.045 * (1 - U)) * morphism
    V += (Dv * Lv + reaction - 0.065 * V) * morphism
    return U, V

# ---------------------------
# Setup visualization using matplotlib
fig, ax = plt.subplots()
img = ax.imshow(U, cmap='inferno', interpolation='bilinear')
plt.axis('off')

# Animation update function
def animate(frame):
    global U, V
    # Modulate the influence of the morphism field slowly over time.
    modulation = 1.0 + 0.05 * np.sin(frame * 0.1)
    U, V = update(U, V, combined_morphism * modulation)
    img.set_data(U)
    return [img]

ani = animation.FuncAnimation(fig, animate, frames=600, interval=30, blit=True)
plt.show()
