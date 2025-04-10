import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sympy import symbols, lambdify, simplify
from scipy.ndimage import convolve

# Step 1: Symbolic definitions
x, y = symbols('x y')

# Define multiple poetic algebraic varieties
varieties = [
    (y**2 - (x**3 - x + (1 + np.sqrt(5))/2)),            # elliptic golden curve
    ((x**2 + y**2 - 1)**3 - x**2 * y**3),                 # heart shape
    ((x**2 + y**2)**2 - 4 * x**2 * y**2),                 # Möbius-style symmetry
    (y - x**3 + 3 * x),                                   # chaotic broken-wing saddle
]

# Step 2: Build composite morphism field
def build_multi_morphism_field(W, H, varieties, scale=3.0):
    grid_x = np.linspace(-scale, scale, W)
    grid_y = np.linspace(-scale, scale, H)
    X, Y = np.meshgrid(grid_x, grid_y)

    morph_total = np.zeros_like(X)
    for i, expr in enumerate(varieties):
        f_simplified = simplify(expr)
        f_func = lambdify((x, y), f_simplified, 'numpy')
        E = f_func(X, Y)

        # Irrational deformation and modulation
        irr = np.sqrt(2 + 0.1 * i)
        phi = (1 + np.sqrt(5)) / 2
        morph = np.cos(irr * E) * np.sin(phi * E)

        # Combine morphisms (here, additive superposition)
        morph_total += morph

    # Normalize composite morphism
    morph_total = (morph_total - morph_total.min()) / (morph_total.max() - morph_total.min()) + 0.2
    return morph_total

# Step 3: Reaction–diffusion setup
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

morphism = build_multi_morphism_field(W, H, varieties)

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
    mod = 1.0 + 0.05 * np.sin(frame * 0.1)
    U, V = update(U, V, morphism * mod)
    img.set_data(U)
    return [img]

ani = animation.FuncAnimation(fig, animate, frames=600, interval=30, blit=True)
plt.show()
