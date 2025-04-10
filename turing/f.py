import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, lambdify, simplify, sin, cos, sqrt
from scipy.ndimage import convolve
from skimage.metrics import structural_similarity as ssim
import random

# Symbolic setup
x, y = symbols('x y')
poetic_atoms = [
    x**2 + y**2 - 1,
    y**2 - x**3 + x,
    (x**2 + y**2)**2 - 4*x**2*y**2,
    sin(x*y), cos(x**2 - y**2),
    x**3 - 3*x*y**2,
    sqrt(2)*x + (1 + sqrt(5))/2 * y
]

# Diffusion setup
W, H = 128, 128
Du, Dv = 0.14, 0.06
laplace_kernel = np.array([[0.05, 0.2 , 0.05],
                           [0.2 , -1.0, 0.2 ],
                           [0.05, 0.2 , 0.05]])

def random_variety():
    terms = random.sample(poetic_atoms, k=random.randint(2, 4))
    expr = sum(random.uniform(-3, 3) * t for t in terms)
    return simplify(expr)

def morphism_field(expr, W, H, scale=3.0):
    grid_x = np.linspace(-scale, scale, W)
    grid_y = np.linspace(-scale, scale, H)
    X, Y = np.meshgrid(grid_x, grid_y)
    f_func = lambdify((x, y), expr, 'numpy')
    try:
        E = f_func(X, Y)
    except:
        return np.ones((W, H))
    morph = np.sin(np.sqrt(2) * E) * np.cos((1 + np.sqrt(5))/2 * E)
    morph = np.nan_to_num(morph)
    morph = (morph - morph.min()) / (morph.max() - morph.min()) + 0.1
    return morph

def update(U, V, morphism):
    Lu = convolve(U, laplace_kernel, mode='reflect')
    Lv = convolve(V, laplace_kernel, mode='reflect')
    reaction = U * V**2
    U += (Du * Lu - reaction + 0.045 * (1 - U)) * morphism
    V += (Dv * Lv + reaction - 0.065 * V) * morphism
    return U, V

# Initialization
def init_state():
    U = np.ones((W, H)) + 0.01 * np.random.randn(W, H)
    V = np.zeros((W, H)) + 0.01 * np.random.randn(W, H)
    r = 10
    U[W//2 - r:W//2 + r, H//2 - r:H//2 + r] = 0.50
    V[W//2 - r:W//2 + r, H//2 - r:H//2 + r] = 0.25
    return U, V

# Similarity metric
def similarity_metric(prev, curr):
    return ssim(prev, curr)

# Main search engine loop
fig, ax = plt.subplots()
plt.axis('off')
img = ax.imshow(np.zeros((W, H)), cmap='inferno', interpolation='bilinear')

U, V = init_state()
prev_frame = np.zeros((W, H))
expr = random_variety()
morph = morphism_field(expr, W, H)

def step(frame_num):
    global U, V, morph, expr, prev_frame
    for _ in range(5):  # multiple steps per frame for visible change
        U, V = update(U, V, morph)

    pattern = U.copy()
    sim = similarity_metric(prev_frame, pattern)

    if 1.4 < sim < 1.6:  # 1.5:1 similarity window
        expr = random_variety()
        morph = morphism_field(expr, W, H)
        print(f"Found fractal-like match. New expr: {expr}")

    prev_frame = pattern.copy()
    img.set_data(U)
    return [img]

import matplotlib.animation as animation
ani = animation.FuncAnimation(fig, step, frames=5000, interval=50, blit=True)
plt.show()
