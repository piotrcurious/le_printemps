import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from sympy import symbols, lambdify, simplify
from scipy.ndimage import convolve, zoom
from scipy.signal import correlate2d

############################
# Symbolic and Algebraic Setup
############################

# Define symbolic variables and multiple poetic varieties
x, y = symbols('x y')
varieties = [
    (y**2 - (x**3 - x + (1 + np.sqrt(5)) / 2)),   # elliptic golden curve
    ((x**2 + y**2 - 1)**3 - x**2 * y**3),           # heart shape
    ((x**2 + y**2)**2 - 4 * x**2 * y**2),           # Möbius-style symmetry
    (y - x**3 + 3 * x),                             # broken-wing saddle
]

def build_multi_morphism_field(W, H, varieties, scale=3.0):
    grid_x = np.linspace(-scale, scale, W)
    grid_y = np.linspace(-scale, scale, H)
    X, Y = np.meshgrid(grid_x, grid_y)
    morph_total = np.zeros_like(X)
    for i, expr in enumerate(varieties):
        f_simplified = simplify(expr)
        f_func = lambdify((x, y), f_simplified, 'numpy')
        E = f_func(X, Y)
        # Incorporate irrational deformations to produce non–commutative modulation
        irr = np.sqrt(2 + 0.1 * i)
        phi_val = (1 + np.sqrt(5)) / 2
        morph = np.cos(irr * E) * np.sin(phi_val * E)
        morph_total += morph
    # Normalize composite morphism (ensuring strictly positive modulation)
    morph_total = (morph_total - morph_total.min()) / (morph_total.max() - morph_total.min()) + 0.2
    return morph_total

############################
# Reaction-Diffusion Setup
############################

W, H = 256, 256
Du, Dv = 0.14, 0.06
# Define an anisotropic Laplacian kernel
laplace_kernel = np.array([[0.05, 0.2, 0.05],
                           [0.2, -1.0, 0.2],
                           [0.05, 0.2, 0.05]])

# Initialize fields with small random perturbations
U = np.ones((W, H)) + 0.01 * np.random.randn(W, H)
V = np.zeros((W, H)) + 0.01 * np.random.randn(W, H)
r = 10
U[W // 2 - r:W // 2 + r, H // 2 - r:H // 2 + r] = 0.50
V[W // 2 - r:W // 2 + r, H // 2 - r:H // 2 + r] = 0.25

# Build the composite morphism field from the symbolic varieties
morphism = build_multi_morphism_field(W, H, varieties)

def update_reaction_diffusion(U, V, morphism):
    Lu = convolve(U, laplace_kernel, mode='reflect')
    Lv = convolve(V, laplace_kernel, mode='reflect')
    reaction = U * V**2
    U += (Du * Lu - reaction + 0.045 * (1 - U)) * morphism
    V += (Dv * Lv + reaction - 0.065 * V) * morphism
    return U, V

############################
# Fractal Pattern Search Engine
############################

def normalized_cross_correlation(patch, image):
    # Subtract mean
    patch_mean = patch - np.mean(patch)
    image_mean = image - np.mean(image)
    # Compute numerator and denominator for normalized cross-correlation
    numerator = correlate2d(image_mean, patch_mean, mode='valid')
    denominator = np.sqrt(
        correlate2d(image_mean**2, np.ones(patch.shape), mode='valid') *
        np.sum(patch_mean**2)
    )
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        ncc = np.where(denominator == 0, 0, numerator / denominator)
    return ncc

def search_fractal_patterns(U, patch_size=32, scale_factor=1.5, threshold=0.8, num_candidates=5):
    detected = []
    H_img, W_img = U.shape
    for _ in range(num_candidates):
        # Pick a random patch location in the field (ensuring full patch is contained)
        x0 = np.random.randint(0, W_img - patch_size)
        y0 = np.random.randint(0, H_img - patch_size)
        patch = U[y0:y0 + patch_size, x0:x0 + patch_size]

        # Scale the patch by the factor 1.5 using zoom; order=1 for bilinear interpolation
        scaled_patch = zoom(patch, scale_factor, order=1)
        sp_H, sp_W = scaled_patch.shape

        # Enforce a search only if scaled patch fits in the larger image
        if sp_H >= H_img or sp_W >= W_img:
            continue
        # Compute normalized cross-correlation on U (treated as template matching)
        ncc = normalized_cross_correlation(scaled_patch, U)
        max_corr = np.max(ncc)
        if max_corr > threshold:
            # Get the coordinates where max correlation occurs
            pos = np.unravel_index(np.argmax(ncc), ncc.shape)
            detected.append((pos[1], pos[0], sp_W, sp_H))
    return detected

############################
# Visualization and Animation
############################

fig, ax = plt.subplots()
img = ax.imshow(U, cmap='inferno', interpolation='bilinear')
plt.axis('off')

# Prepare a container for rectangle overlays
rectangles = []

# Search frequency (in frames)
search_interval = 50
# How long a detected region will be highlighted (in frames)
highlight_duration = 20
# List to store detected pattern overlays: each as (frame_detected, (x, y, w, h))
detected_overlays = []

def animate(frame):
    global U, V, detected_overlays, rectangles
    # Update reaction-diffusion simulation with an extra periodic modulation
    modulation = 1.0 + 0.05 * np.sin(frame * 0.1)
    U, V = update_reaction_diffusion(U, V, morphism * modulation)
    img.set_data(U)

    # Every `search_interval` frames perform the fractal search
    if frame % search_interval == 0:
        new_detections = search_fractal_patterns(U, patch_size=32,
                                                 scale_factor=1.5,
                                                 threshold=0.8,
                                                 num_candidates=5)
        # Record current frame with detected overlays
        for det in new_detections:
            detected_overlays.append((frame, det))

    # Remove old rectangle patches from axes
    for rect in rectangles:
        rect.remove()
    rectangles = []

    # Overlay detected regions that are still fresh (within highlight_duration)
    active_overlays = []
    for det_frame, (x, y, w, h) in detected_overlays:
        if frame - det_frame < highlight_duration:
            rect = Rectangle((x, y), w, h,
                             linewidth=2, edgecolor='cyan', facecolor='none')
            ax.add_patch(rect)
            rectangles.append(rect)
            active_overlays.append((det_frame, (x, y, w, h)))
    detected_overlays = active_overlays

    return [img] + rectangles

ani = animation.FuncAnimation(fig, animate, frames=600, interval=30, blit=True)
plt.show()
