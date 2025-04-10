import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sympy import symbols, solve, Eq, sqrt
from scipy.ndimage import gaussian_filter

# Constants & Geometry
phi = (1 + np.sqrt(5)) / 2  # golden ratio: irrational
width, height = 200, 200
Du, Dv = 0.16, 0.08  # diffusion coefficients

# Advanced algebraic geometry morphism: elliptic curve modulation
x, y = symbols('x y')
elliptic_curve = Eq(y**2, x**3 - 3*x + 1)

def elliptic_potential(grid_x, grid_y):
    X, Y = np.meshgrid(grid_x, grid_y)
    E = (Y**2 - X**3 + 3*X - 1)**2
    return 1 / (1 + E)

# Zeldovich-style instability: thin front diffusion influenced by morphism field
def reaction_diffusion(U, V, morphism):
    Lu = gaussian_filter(U, sigma=1)
    Lv = gaussian_filter(V, sigma=1)
    
    reaction = U * V**2
    U += (Du * Lu - reaction + 0.055 * (1 - U)) * morphism
    V += (Dv * Lv + reaction - (0.062 + 0.005 * np.sin(phi)) * V) * morphism
    return U, V

# Initial Conditions
U = np.ones((width, height))
V = np.zeros((width, height))
r = 20
U[width//2-r:width//2+r, height//2-r:height//2+r] = 0.50
V[width//2-r:width//2+r, height//2-r:height//2+r] = 0.25

grid_x = np.linspace(-2, 2, width)
grid_y = np.linspace(-2, 2, height)
morphism_field = elliptic_potential(grid_x, grid_y)

# Poetic narrative modulation (mapped emotionally)
poem_arc = np.linspace(0.8, 1.2, 500)

fig, ax = plt.subplots()
img = ax.imshow(U, cmap='plasma', interpolation='bilinear')
plt.axis('off')

def update(frame):
    global U, V
    U, V = reaction_diffusion(U, V, morphism_field * poem_arc[frame % len(poem_arc)])
    img.set_data(U)
    return [img]

ani = animation.FuncAnimation(fig, update, frames=600, interval=50, blit=True)
plt.show()
