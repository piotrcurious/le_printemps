import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter

# Global parameters
phi = (1 + np.sqrt(5)) / 2  # Golden ratio; an irrational number for modulation
width, height = 200, 200    # Grid dimensions
Du, Dv = 0.16, 0.08         # Diffusion coefficients

# Improved Algebraic Geometry Module
#
# Here we define two elliptic curves with parameters chosen to introduce rich algebraic
# invariants into the diffusion field. The idea is to compute two implicit functions:
#   f1(x, y) = y^2 - (x^3 - 3*x + 1)
#   f2(x, y) = y^2 - (x^3 - phi*x + phi**2)
#
# Their squared sums form an "error" field whose reciprocal (after proper normalization)
# creates a potential that modulates the reaction–diffusion process. This introduces 
# an interplay between algebraically defined structures and the emergence of complex patterns.
def algebraic_morphism(grid_x, grid_y):
    X, Y = np.meshgrid(grid_x, grid_y)
    # Define two elliptic curves (Weierstrass forms with distinct parameters)
    f1 = Y**2 - (X**3 - 3*X + 1)
    f2 = Y**2 - (X**3 - phi*X + phi**2)
    # Combined "error" reflecting deviation from these algebraic curves
    E = f1**2 + f2**2
    # The potential function inversely modulates diffusion intensity.
    # Higher values of E (further from the curves) result in smaller influence,
    # while regions near the curves have stronger modulation.
    return 1.0 / (1.0 + E)

# Reaction–Diffusion dynamics with field-modulated instability
def reaction_diffusion(U, V, morphism):
    # Simple Laplacian via Gaussian filter approximates diffusion;
    # this is then locally weighted by the algebraic morphism field.
    Lu = gaussian_filter(U, sigma=1)
    Lv = gaussian_filter(V, sigma=1)
    # The reaction term couples U and V in a nonlinear manner.
    reaction = U * V**2
    # Update equations incorporate algebraic geometry modulation:
    U += (Du * Lu - reaction + 0.055 * (1 - U)) * morphism
    V += (Dv * Lv + reaction - (0.062 + 0.005 * np.sin(phi)) * V) * morphism
    return U, V

# Initial conditions: a mostly uniform state with a slight perturbation in the center.
U = np.ones((width, height))
V = np.zeros((width, height))
r = 20
U[width//2 - r:width//2 + r, height//2 - r:height//2 + r] = 0.50
V[width//2 - r:width//2 + r, height//2 - r:height//2 + r] = 0.25

# Create a coordinate grid to be used in the algebraic morphism.
grid_x = np.linspace(-2, 2, width)
grid_y = np.linspace(-2, 2, height)
morphism_field = algebraic_morphism(grid_x, grid_y)

# Poetic modulation: A slowly changing modulation factor inspired by the emotional arc of the poem.
poem_arc = np.linspace(0.8, 1.2, 500)

# Set up the matplotlib figure
fig, ax = plt.subplots()
img = ax.imshow(U, cmap='plasma', interpolation='bilinear')
plt.axis('off')

def update(frame):
    global U, V
    # Modulate the effect of the algebraic field with the poetic arc to guide instability.
    current_modulation = morphism_field * poem_arc[frame % len(poem_arc)]
    U, V = reaction_diffusion(U, V, current_modulation)
    img.set_data(U)
    return [img]

ani = animation.FuncAnimation(fig, update, frames=600, interval=50, blit=True)
plt.show()
