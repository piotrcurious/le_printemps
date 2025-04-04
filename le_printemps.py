import pygame
import numpy as np
import time
from scipy.signal import convolve2d
from scipy.ndimage import distance_transform_edt, gaussian_filter
import math
import cmath

# --- Configuration ---
WIDTH, HEIGHT = 800, 600
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Mathematical Opus: Le Printemps (Deepened)")
pygame.font.init()
FONT = pygame.font.SysFont("Arial", 18)
CLOCK = pygame.time.Clock()
FPS = 30

# --- Colors (Expand Palette) ---
BLACK = (0, 0, 0, 255) # Add alpha
WHITE = (255, 255, 255, 255)
SPRING_GREEN = (127, 255, 150, 255)
SKY_BLUE = (135, 206, 250, 255)
GOLD_YELLOW = (255, 215, 0, 255)
ROSE_PINK = (255, 150, 192, 255)
AMETHYST_PURPLE = (150, 100, 215, 255)
SHADOW_GREY = (50, 50, 60, 255)
DEEP_INDIGO = (40, 0, 75, 255)
TRANSPARENT = (0, 0, 0, 0)

# --- Utility Functions ---
def lerp(a, b, t):
    """Linear interpolation"""
    return a + (b - a) * t

def lerp_color(c1, c2, t):
    """Linear interpolation for RGBA colors"""
    return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))

def map_value(value, in_min, in_max, out_min, out_max):
    """Map value from one range to another"""
    if in_max == in_min: return out_min # Avoid division by zero
    return out_min + (out_max - out_min) * ((value - in_min) / (in_max - in_min))

# --- State Management ---
WANDERING, DISSOLVING, CONSTRAINED = 0, 1, 2
current_state = WANDERING
state_start_time = time.time()
transition_start_time = 0
transition_duration = 2.0 # Seconds for smooth transition
is_transitioning = False
state_names = {WANDERING: "1. Wandering Spring / Emergence",
               DISSOLVING: "2. Dissolving Morning / Transience",
               CONSTRAINED: "3. Chained Sadness / Structure"}

# Target parameters for each state (will be interpolated)
state_params = {
    WANDERING: {
        "gol_color": SPRING_GREEN, "gol_update_interval": 150, "gol_base_alpha": 200,
        "fractal_alpha": 80, "fractal_max_iter": 15, "fractal_zoom": 1.0, "fractal_offset_x": -0.5, "fractal_offset_y": 0.0, "fractal_color_mode": "spring",
        "curve_type": "oscillating_circle", "curve_color": SKY_BLUE, "curve_alpha": 180, "curve_influence": 0.3, "curve_param": 1.0,
        "bg_color": lerp_color(DEEP_INDIGO, BLACK, 0.8)
    },
    DISSOLVING: {
        "gol_color": ROSE_PINK, "gol_update_interval": 300, "gol_base_alpha": 120,
        "fractal_alpha": 50, "fractal_max_iter": 25, "fractal_zoom": 1.2, "fractal_offset_x": -0.6, "fractal_offset_y": 0.2, "fractal_color_mode": "transient",
        "curve_type": "pulsing_heart", "curve_color": GOLD_YELLOW, "curve_alpha": 240, "curve_influence": 0.6, "curve_param": 1.1,
        "bg_color": lerp_color(DEEP_INDIGO, BLACK, 0.5)
    },
    CONSTRAINED: {
        "gol_color": SHADOW_GREY, "gol_update_interval": 500, "gol_base_alpha": 150,
        "fractal_alpha": 220, "fractal_max_iter": 60, "fractal_zoom": 2.5, "fractal_offset_x": -0.745, "fractal_offset_y": 0.113, "fractal_color_mode": "shadow",
        "curve_type": "none", "curve_color": TRANSPARENT, "curve_alpha": 0, "curve_influence": 0.0, "curve_param": 1.0, # Curve becomes constraint via fractal
        "bg_color": BLACK
    }
}
# Current interpolated parameters
params = state_params[WANDERING].copy()
prev_params = params.copy() # Store previous state's params for interpolation

# --- Game of Life ---
CELL_SIZE = 6 # Slightly larger cells
GRID_WIDTH, GRID_HEIGHT = WIDTH // CELL_SIZE, HEIGHT // CELL_SIZE
GOL_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
gol_grid = np.random.randint(0, 2, size=(GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)
gol_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
last_gol_update = 0
# For interaction: Store recent activity (e.g., births/deaths)
gol_activity = np.zeros_like(gol_grid, dtype=float)

# --- Fractals (Mandelbrot) ---
fractal_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
fractal_iterations = np.zeros((HEIGHT, WIDTH), dtype=int) # Store iteration data
fractal_needs_update = True
# Define color palettes
COLOR_PALETTES = {
    "spring": [lerp_color(SPRING_GREEN, SKY_BLUE, i/15) for i in range(16)],
    "transient": [lerp_color(GOLD_YELLOW, ROSE_PINK, i/25) for i in range(26)],
    "shadow": [lerp_color(SHADOW_GREY, DEEP_INDIGO, i/60) for i in range(61)]
}
COLOR_PALETTES["spring"][-1] = BLACK # Inside set
COLOR_PALETTES["transient"][-1] = BLACK
COLOR_PALETTES["shadow"][-1] = BLACK

# --- Algebraic Geometry & Interaction Field ---
curve_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
curve_mask = np.zeros((HEIGHT, GRID_HEIGHT, GRID_WIDTH), dtype=bool) # For drawing on GoL grid scale
distance_field = np.zeros_like(gol_grid, dtype=float) # Distance from curve
curve_needs_update = True
curve_line_thickness = 3

# == Core Functions ==

def update_params(t):
    """Interpolate parameters during transition"""
    global params
    for key in params:
        if isinstance(params[key], (int, float)):
            params[key] = lerp(prev_params[key], state_params[current_state][key], t)
        elif isinstance(params[key], tuple) and len(params[key]) == 4: # Color
             params[key] = lerp_color(prev_params[key], state_params[current_state][key], t)
        elif isinstance(params[key], str): # Non-interpolatable params like type/mode
             if t > 0.5: # Switch halfway through transition
                  params[key] = state_params[current_state][key]
             else:
                  params[key] = prev_params[key] # Keep previous during first half


def update_gol_interactive(grid, activity, dist_field, fractal_iter):
    """GoL update incorporating distance field and fractal structure"""
    global gol_activity
    neighbors = convolve2d(grid, GOL_KERNEL, mode='same', boundary='wrap')

    # --- Calculate Survival/Birth Probabilities ---
    # Base probability (can be adjusted)
    prob_survival = np.full(grid.shape, 0.95, dtype=float) # Base high survival if neighbors are right
    prob_birth = np.full(grid.shape, 0.15, dtype=float)   # Base lower birth rate

    # 1. Influence from Algebraic Curve (Distance Field)
    #    Closer to curve might encourage/discourage life depending on state
    #    Map distance (0 to max_dist) to modulation factor (-1 to 1)
    max_dist = max(GRID_WIDTH, GRID_HEIGHT) / 2 # Approximate max distance
    dist_modulation = map_value(dist_field, 0, max_dist, params['curve_influence'], -params['curve_influence'])
    prob_birth *= (1.0 + dist_modulation)
    prob_survival *= (1.0 - dist_modulation * 0.5) # Less impact on survival

    # 2. Influence from Fractal Structure
    #    Cells inside the set (max iterations) or in low-iter regions might die
    #    Map fractal iteration count at cell location
    fractal_at_gol = fractal_iter[::CELL_SIZE, ::CELL_SIZE][:GRID_HEIGHT, :GRID_WIDTH] # Sample fractal at grid points
    inside_set = fractal_at_gol >= params['fractal_max_iter'] -1 # Identify cells inside or very close
    # Reduce survival drastically inside the set (Constraint phase)
    constraint_factor = 1.0 if current_state != CONSTRAINED else 0.01
    prob_survival[inside_set] *= constraint_factor

    # Clip probabilities
    prob_survival = np.clip(prob_survival, 0.01, 0.99)
    prob_birth = np.clip(prob_birth, 0.01, 0.99)

    # --- Apply Conway's Rules with Probabilities ---
    new_grid = grid.copy()
    activity_update = np.zeros_like(grid, dtype=float)

    # Potential births (dead cells with 3 neighbors)
    potential_births = (neighbors == 3) & (grid == 0)
    birth_roll = np.random.rand(*grid.shape)
    born = potential_births & (birth_roll < prob_birth)
    new_grid[born] = 1
    activity_update[born] = 1.0 # High activity for birth

    # Potential deaths (live cells with <2 or >3 neighbors)
    underpopulated = (neighbors < 2) & (grid == 1)
    overpopulated = (neighbors > 3) & (grid == 1)
    potential_deaths = underpopulated | overpopulated
    death_roll = np.random.rand(*grid.shape)
    # Survival = NOT potential death OR (potential death AND survived the roll)
    survived_death_roll = potential_deaths & (death_roll >= (1.0 - prob_survival)) # If roll fails survival prob, you die
    died = potential_deaths & ~survived_death_roll
    new_grid[died] = 0
    activity_update[died] = 0.8 # High activity for death

    # Update activity buffer (fade over time)
    gol_activity = gaussian_filter(activity_update + gol_activity * 0.85, sigma=0.6)

    return new_grid


def draw_gol_interactive(surface, grid, activity):
    """Draw GoL, coloring based on state and potentially activity"""
    surface.fill(TRANSPARENT)
    alpha = int(params['gol_base_alpha'])
    base_color = params['gol_color']

    # Draw live cells
    for r in range(GRID_HEIGHT):
        for c in range(GRID_WIDTH):
            if grid[r, c] == 1:
                # Optional: Modulate color/alpha by activity level
                act = np.clip(activity[r,c], 0, 1)
                cell_color = lerp_color(base_color, WHITE, act * 0.5) # Whiter if active
                cell_color = (*cell_color[:3], alpha) # Apply base alpha

                pygame.draw.rect(surface, cell_color, (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    surface.set_alpha(alpha) # Overall surface alpha


def calculate_fractal_interactive(w, h, max_iter, zoom, offset_x, offset_y, gol_activity_map):
    """Calculate Mandelbrot, potentially influenced slightly by GoL activity"""
    x = np.linspace(-2.0 / zoom + offset_x, 1.0 / zoom + offset_x, w)
    y = np.linspace(-1.2 / zoom + offset_y, 1.2 / zoom + offset_y, h)
    c_base = x[:, np.newaxis] + 1j * y[np.newaxis, :]

    # --- GoL Influence ---
    # Scale up GoL activity map to fractal dimensions and apply subtle perturbation
    # Use basic resize for speed (or could use interpolation)
    influence_map = np.kron(gol_activity_map, np.ones((CELL_SIZE, CELL_SIZE)))[:h, :w] # Upscale activity map
    influence_map = gaussian_filter(influence_map, sigma=5) # Smooth influence
    perturbation_strength = 0.005 # Very small perturbation
    c_perturbed = c_base + influence_map * perturbation_strength * (1 + 1j)

    z = np.zeros_like(c_perturbed, dtype=np.complex128)
    iterations = np.zeros(c_perturbed.shape, dtype=int)
    mask = np.ones(c_perturbed.shape, dtype=bool)

    for i in range(max_iter):
        if not mask.any(): break # All points escaped or finished
        z[mask] = z[mask]**2 + c_perturbed[mask]
        diverged = np.abs(z) > 4.0 # Increase escape radius slightly
        newly_diverged = mask & diverged
        iterations[newly_diverged] = i
        mask &= ~diverged

    iterations[mask] = max_iter # Points inside the set
    return iterations.T # Transpose


def draw_fractal_colored(surface, iterations):
    """Draw fractal using state-specific colormaps"""
    max_iter = int(params['fractal_max_iter'])
    alpha = int(params['fractal_alpha'])
    color_mode = params['fractal_color_mode']
    palette = COLOR_PALETTES.get(color_mode, COLOR_PALETTES["shadow"]) # Default to shadow
    max_palette_index = len(palette) - 1

    # Map iterations to palette indices
    norm_iter = np.clip(iterations, 0, max_iter)
    indices = np.clip((norm_iter / max(1,max_iter) * max_palette_index).astype(int), 0, max_palette_index)

    # Create color array from palette
    # Ensure palette colors have alpha=255 for surfarray, apply surface alpha later
    color_array = np.array([(*c[:3], 255) for c in palette])[indices]

    pygame.surfarray.blit_array(surface, color_array)
    surface.set_alpha(alpha)


def update_curve_and_field(w, h, grid_w, grid_h):
    """Calculate curve mask and distance field"""
    global curve_mask, distance_field, curve_needs_update
    curve_type = params['curve_type']
    curve_param = params['curve_param']
    current_time = time.time()

    if curve_type == 'none':
        curve_mask.fill(False)
        distance_field.fill(max(grid_w, grid_h)) # Max distance everywhere
        curve_needs_update = False # No need to recalc if none
        return

    # Define curve equation space slightly larger than screen ratio
    aspect = w / h
    x_range = (-1.5 * aspect, 1.5 * aspect)
    y_range = (-1.5, 1.5)
    x_vals = np.linspace(x_range[0], x_range[1], grid_w)
    y_vals = np.linspace(y_range[0], y_range[1], grid_h)
    x, y = np.meshgrid(x_vals, y_vals)

    mask = np.zeros((grid_h, grid_w), dtype=bool)
    threshold_scale = 0.1 # Adjust sensitivity

    try:
        if curve_type == 'oscillating_circle':
            radius = 0.8 + 0.2 * math.sin(current_time * 0.8) # Oscillating radius
            center_x = 0.1 * math.cos(current_time * 0.5)
            center_y = 0.1 * math.sin(current_time * 0.6)
            val = (x - center_x)**2 + (y - center_y)**2 - radius**2
            threshold = threshold_scale * radius
            mask = np.abs(val) < threshold
        elif curve_type == 'pulsing_heart':
            scale = 1.0 + 0.15 * math.sin(current_time * 1.5) # Pulsing scale
            xs, ys = (x / scale), (y / scale)
            # Shift center slightly for visual interest
            xs -= 0.1 * math.sin(current_time * 0.4)
            ys += 0.1 - 0.1 * math.cos(current_time * 0.6)
            val = (xs**2 + ys**2 - 1)**3 - (xs**2 * ys**3)
            threshold = threshold_scale * 0.05 # Heart needs smaller threshold
            mask = np.abs(val) < threshold

    except Exception as e:
        print(f"Error calculating curve {curve_type}: {e}")
        mask.fill(False)

    # Calculate distance field from the curve mask (invert mask for distance_transform)
    distance_field = distance_transform_edt(~mask)
    curve_mask = mask # Store the calculated mask
    curve_needs_update = False # Mark as updated for this frame


def draw_curve_from_mask(surface):
    """Draw the curve using the pre-calculated GoL-grid-scale mask"""
    surface.fill(TRANSPARENT)
    alpha = int(params['curve_alpha'])
    color = params['curve_color']
    thickness = curve_line_thickness

    if alpha == 0: return # Don't draw if invisible

    # Draw points from the mask, scaled up to screen coords
    curve_points_indices = np.argwhere(curve_mask) # Get indices (row, col)
    for r, c in curve_points_indices:
        screen_x = c * CELL_SIZE + CELL_SIZE // 2 # Center of cell
        screen_y = r * CELL_SIZE + CELL_SIZE // 2
        # Draw small circle for smoother look
        pygame.draw.circle(surface, color, (screen_x, screen_y), thickness)

    surface.set_alpha(alpha)


# == Main Loop Setup ==
running = True
force_redraw_all = True # Initial draw

# == Main Game Loop ==
while running:
    current_time_global = time.time()
    delta_time = CLOCK.tick(FPS) / 1000.0 # Time since last frame in seconds

    # --- Event Handling ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            # Manual state triggering for debug
            if event.key == pygame.K_1: current_state = WANDERING; is_transitioning = True; transition_start_time = current_time_global; prev_params = params.copy(); force_redraw_all = True
            if event.key == pygame.K_2: current_state = DISSOLVING; is_transitioning = True; transition_start_time = current_time_global; prev_params = params.copy(); force_redraw_all = True
            if event.key == pygame.K_3: current_state = CONSTRAINED; is_transitioning = True; transition_start_time = current_time_global; prev_params = params.copy(); force_redraw_all = True


    # --- State Transition & Parameter Update ---
    if not is_transitioning:
         # Check for automatic state change
         time_in_state = current_time_global - state_start_time
         # Determine next state duration based on CURRENT state's params
         # This requires accessing the state_params dict directly for duration
         state_duration_key = {WANDERING: 15.0, DISSOLVING: 12.0, CONSTRAINED: 25.0} # Define durations here
         if time_in_state > state_duration_key[current_state]:
              prev_params = params.copy() # Store current interpolated params
              current_state = (current_state + 1) % 3
              state_start_time = current_time_global # Reset state timer
              transition_start_time = current_time_global
              is_transitioning = True
              force_redraw_all = True # Need redraw after transition setup
              print(f"Auto-transitioning to: {state_names[current_state]}")

    if is_transitioning:
        transition_time = current_time_global - transition_start_time
        t = min(transition_time / transition_duration, 1.0) # Normalized transition progress (0 to 1)
        update_params(t)
        # Check if transition elements need update based on new params
        fractal_needs_update = True # Always update fractal during/after transition
        curve_needs_update = True   # Always update curve during/after transition
        if t >= 1.0:
            is_transitioning = False
            state_start_time = current_time_global # Mark state as fully entered NOW
            print(f"Transition complete: {state_names[current_state]}")


    # --- Updates ---
    # Update GoL periodically based on interpolated interval
    if current_time_global * 1000 - last_gol_update > params['gol_update_interval']:
        # Curve update influences GoL, so ensure it's calculated first if needed
        if curve_needs_update or params['curve_type'] != 'none': # Recalc curve if needed or active
            update_curve_and_field(WIDTH, HEIGHT, GRID_WIDTH, GRID_HEIGHT)

        # Fractal update provides constraint/background for GoL
        if fractal_needs_update or force_redraw_all:
             # Pass GoL activity map to potentially influence fractal calc
             fractal_iterations = calculate_fractal_interactive(WIDTH, HEIGHT, int(params['fractal_max_iter']), params['fractal_zoom'], params['fractal_offset_x'], params['fractal_offset_y'], gol_activity)
             fractal_needs_update = False # Reset flag

        # Now update GoL using the latest distance field and fractal data
        gol_grid = update_gol_interactive(gol_grid, gol_activity, distance_field, fractal_iterations)
        last_gol_update = current_time_global * 1000
        force_redraw_all = False # Allow incremental updates now

    # Force curve update if its type requires animation (e.g., oscillating)
    if 'oscillating' in params['curve_type'] or 'pulsing' in params['curve_type']:
         curve_needs_update = True # Recalculate animated curve each frame


    # --- Drawing ---
    SCREEN.fill(params['bg_color']) # Background color reflects state

    # 1. Draw Fractal (potentially influenced by GoL)
    draw_fractal_colored(fractal_surface, fractal_iterations)
    SCREEN.blit(fractal_surface, (0, 0))

    # 2. Draw Curve (which influences GoL)
    if params['curve_alpha'] > 0:
        draw_curve_from_mask(curve_surface)
        SCREEN.blit(curve_surface, (0, 0))

    # 3. Draw GoL (influenced by Curve and Fractal)
    draw_gol_interactive(gol_surface, gol_grid, gol_activity)
    SCREEN.blit(gol_surface, (0, 0))

    # 4. Display State Info Text
    state_label = state_names[current_state]
    if is_transitioning:
        state_label += " (Transitioning...)"
    state_text = FONT.render(state_label, True, WHITE)
    SCREEN.blit(state_text, (10, 10))

    # --- Update Display ---
    pygame.display.flip()

pygame.quit()
