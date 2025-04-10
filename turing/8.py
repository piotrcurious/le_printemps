import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter

# Global parameters
phi = (1 + np.sqrt(5)) / 2  # Golden ratio, an irrational constant for modulation
width, height = 200, 200    # Spatial grid dimensions
Du, Dv = 0.16, 0.08         # Diffusion coefficients

# 1. Algebraic Morphism Field
def algebraic_morphism(grid_x, grid_y):
    X, Y = np.meshgrid(grid_x, grid_y)
    # Two elliptic curves in Weierstrass form with different parameters:
    f1 = Y**2 - (X**3 - 3 * X + 1)
    f2 = Y**2 - (X**3 - phi * X + phi**2)
    # Combined deviation from these algebraic curves:
    E = f1**2 + f2**2
    # The potential field: regions nearer to the curves have greater influence.
    return 1.0 / (1.0 + E)

# 2. Explicit Pattern Field using field morphisms for clear spatial pattern definitions.
def explicit_field(grid_x, grid_y):
    X, Y = np.meshgrid(grid_x, grid_y)
    # Explicit patterns: stripes and localized spots.
    # Sinusoidal stripe pattern along the x-axis.
    stripe = np.abs(np.sin(2.5 * np.pi * X))
    # Spot-like concentration by a Gaussian centered at (0, 0).
    spots = np.exp(-((2 * X)**2 + (2 * Y)**2))
    # An algebraic (polynomial) modulation gives additional structure.
    poly = (X**2 - Y**2)**2 - (X * Y) + 1
    # Combining these gives an explicit pattern field that is then normalized.
    explicit = (stripe * spots) / (1.0 + np.abs(poly))
    
    explicit -= explicit.min()
    explicit /= explicit.max()
    return explicit

# Combined modulation field from both algebraic invariants and explicit pattern definitions.
def combined_field(grid_x, grid_y):
    alg_field = algebraic_morphism(grid_x, grid_y)
    expl_field = explicit_field(grid_x, grid_y)
    # The fields multiply to yield a complex modulation in space.
    combined = alg_field * expl_field
    # Normalize if necessary:
    combined -= combined.min()
    combined /= combined.max()
    return combined

# Reaction–Diffusion dynamics using the modulated field.
def reaction_diffusion(U, V, mod_field):
    # Approximate the Laplacian using a Gaussian filter.
    Lu = gaussian_filter(U, sigma=1)
    Lv = gaussian_filter(V, sigma=1)
    # Reaction term coupling the two chemicals.
    reaction = U * V**2
    # Update equations, modulated locally by our combined field.
    U += (Du * Lu - reaction + 0.055 * (1 - U)) * mod_field
    V += (Dv * Lv + reaction - (0.062 + 0.005 * np.sin(phi)) * V) * mod_field
    return U, V

# Initial conditions: Uniform fields with a central perturbation.
U = np.ones((width, height))
V = np.zeros((width, height))
r = 20
U[width//2 - r:width//2 + r, height//2 - r:height//2 + r] = 0.50
V[width//2 - r:width//2 + r, height//2 - r:height//2 + r] = 0.25

# Set up the grid for our spatial domain.
grid_x = np.linspace(-2, 2, width)
grid_y = np.linspace(-2, 2, height)
modulation_field = combined_field(grid_x, grid_y)

# Poetic modulation: This factor, inspired by the emotional arc of the poem,
# slowly modulates the combined field over time.
poem_arc = np.linspace(0.8, 1.2, 500)

# Set up the matplotlib figure for animation.
fig, ax = plt.subplots()
img = ax.imshow(U, cmap='plasma', interpolation='bilinear')
plt.axis('off')

def update(frame):
    global U, V
    # Introduce a temporal modulation derived from the poem.
    current_modulation = modulation_field * poem_arc[frame % len(poem_arc)]
    U, V = reaction_diffusion(U, V, current_modulation)
    img.set_data(U)
    return [img]

ani = animation.FuncAnimation(fig, update, frames=600, interval=50, blit=True)
plt.show()
