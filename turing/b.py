import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sympy import symbols, lambdify, simplify
from scipy.ndimage import convolve

# Step 1: Define symbolic variety using SymPy
x, y = symbols('x y')

# Poetry-inspired symbolic curve (editable!)
# You can change this to define your own variety
f = y**2 - (x**3 - x + (1 + np.sqrt(5))/2)  # classic Weierstrass form with golden ratio

# Simplify and lambdify the expression
f_simplified = simplify(f)
f_func = lambdify((x, y), f_simplified, 'numpy')

# Step 2: Build the morphism field from the symbolic curve
def build_morphism_field(W, H, scale=3.0):
    grid_x = np.linspace(-scale, scale, W)
    grid_y = np.linspace(-scale, scale, H)
    X, Y = np.meshgrid(grid_x, grid_y)
    error_field = f_func(X, Y)
    morph = np.cos(np.sqrt(2) * error_field) * np.sin((1 + np.sqrt(5))/2 * error_field)
    morph = (morph - morph.min()) / (morph.max() - morph.min()) + 0.2
    return morph

# Step 3: Reaction–diffusion system with anisotropic instability
W, H = 256, 256
Du, Dv = 0.14, 0.06
laplace_kernel = np.array([[0.05, 0.2 , 0.05],
                           [0.2 , -1.0, 0.2 ],
                           [0.05, 0.2 , 0.05]])

U = np.ones((W, H)) + 0.01 * np.random.randn(W, H)
V = np.zeros((W, H)) + 0.01 * np.random.randn(W, H)
r = 10
U[W//2 - r:W//2 + r, H//2 - r:H//2 + r] = 0.50
V[W//2 - r:W//2 + r, H//2 - r:H//2 + r] = 0.25

morphism = build_morphism_field(W, H)

def update(U, V, morphism):
    Lu = convolve(U, laplace_kernel, mode='reflect')
    Lv = convolve(V, laplace_kernel, mode='reflect')
    reaction = U * V**2
    U += (Du * Lu - reaction + 0.045 * (1 - U)) * morphism
    V += (Dv * Lv + reaction - 0.065 * V) * morphism
    return U, V

# Visualization
fig, ax = plt.subplots()
img = ax.imshow(U, cmap='inferno', interpolation='bilinear')
plt.axis('off')

def animate(frame):
    global U, V
    modulation = 1.0 + 0.05 * np.sin(frame * 0.1)
    U, V = update(U, V, morphism * modulation)
    img.set_data(U)
    return [img]

ani = animation.FuncAnimation(fig, animate, frames=600, interval=30, blit=True)
plt.show()
