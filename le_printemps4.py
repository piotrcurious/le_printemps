import pygame
import numpy as np
import math
import random

# -------- Configuration --------
WIDTH, HEIGHT = 800, 600
CELL_SIZE = 8  # size of Game of Life cells
GRID_WIDTH = WIDTH // CELL_SIZE
GRID_HEIGHT = HEIGHT // CELL_SIZE

FPS = 15

# Colors (spring tones)
BG_COLOR = (230, 245, 230)      # very light greenish
CELL_COLOR = (80, 160, 120)     # soft green
CURVE_COLOR = (100, 180, 200)   # light blue
FRACTAL_COLOR = (200, 150, 100) # gentle orange

# Poem text lines
POEM_LINES = [
    "Le printemps réveille en moi",
    "un désir léger, sans racine ni chemin,",
    "comme une âme bohémienne vagabonde,",
    "cherchant l’instant, sans destin.",
    "",
    "Les mots se dissolvent au matin,",
    "les projets se perdent en l’air du temps,",
    "et je reste, incertain,",
    "face à l’éclat d’un moment fuyant.",
    "",
    "Tel un cheval libre apprivoisé,",
    "vendu aux ombres d’un conte désenchanté,",
    "la raison finit par l’enchaîner",
    "à une tristesse de plus, à jamais."
]

# -------- Chaos/Logistic Map --------
# Logistic map parameter; modulates some parameters of the visualization.
chaos_param = 0.2
r_logistic = 3.8

def update_logistic(x):
    return r_logistic * x * (1 - x)

# -------- Game of Life --------
def init_grid():
    # Start with a random grid
    return np.random.choice([0, 1], size=(GRID_HEIGHT, GRID_WIDTH), p=[0.8, 0.2])

def update_grid(grid):
    # Use periodic boundary conditions
    new_grid = np.copy(grid)
    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            # Count neighbors (with wrapping)
            total = sum([
                grid[(i + di) % GRID_HEIGHT, (j + dj) % GRID_WIDTH]
                for di in (-1, 0, 1) for dj in (-1, 0, 1)
                if not (di == 0 and dj == 0)
            ])
            # Standard Game of Life rules
            if grid[i, j] == 1:
                if total < 2 or total > 3:
                    new_grid[i, j] = 0
            else:
                if total == 3:
                    new_grid[i, j] = 1
    return new_grid

# -------- Algebraic Geometry: Rose Curve --------
def rose_curve_points(k, num_points=200, scale=100, offset=(WIDTH//2, HEIGHT//2)):
    points = []
    for i in range(num_points):
        theta = 2 * math.pi * i / num_points
        r = math.sin(k * theta)
        x = offset[0] + scale * r * math.cos(theta)
        y = offset[1] + scale * r * math.sin(theta)
        points.append((x, y))
    return points

# -------- Fractal Bloom: Recursive Branches --------
def draw_fractal(surface, x, y, length, angle, depth):
    if depth == 0 or length < 2:
        return
    # End point calculation
    x_end = x + length * math.cos(angle)
    y_end = y + length * math.sin(angle)
    pygame.draw.line(surface, FRACTAL_COLOR, (x, y), (x_end, y_end), max(1, depth//2))
    # Recursive branches with slight variation
    new_length = length * 0.7
    draw_fractal(surface, x_end, y_end, new_length, angle + math.pi/6, depth-1)
    draw_fractal(surface, x_end, y_end, new_length, angle - math.pi/6, depth-1)

# -------- Poem Rendering --------
def render_poem(surface, font, alpha):
    # Render the poem with a given transparency (alpha 0-255)
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    y = 10
    for line in POEM_LINES:
        text_surf = font.render(line, True, (0, 0, 0))
        text_surf.set_alpha(alpha)
        overlay.blit(text_surf, (10, y))
        y += text_surf.get_height() + 2
    surface.blit(overlay, (0, 0))

# -------- Main Pygame Setup --------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Le printemps réveille en moi")
clock = pygame.time.Clock()
font = pygame.font.SysFont("serif", 18)

# Initialize grid and chaos parameter
grid = init_grid()

# For looping chaotic parameter modulation
frame_count = 0

running = True
while running:
    clock.tick(FPS)
    frame_count += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- Chaos Engine: update logistic map parameter ---
    chaos_param = update_logistic(chaos_param)

    # --- Update Game of Life grid ---
    grid = update_grid(grid)

    # --- Draw Background ---
    screen.fill(BG_COLOR)

    # --- Draw Game of Life Cells ---
    # Color intensity modulated by chaos_param (to give dynamic spring tone)
    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            if grid[i, j] == 1:
                # Slight color variation based on chaos_param
                color = (
                    min(255, CELL_COLOR[0] + int(50 * chaos_param)),
                    min(255, CELL_COLOR[1] + int(50 * (1 - chaos_param))),
                    CELL_COLOR[2]
                )
                rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, color, rect)

    # --- Draw Algebraic Geometry: Evolving Rose Curve ---
    # Use chaos_param to modulate the "k" parameter (petal count) and scale
    k_val = 3 + 2 * chaos_param  # oscillates between 3 and 5
    scale_val = 80 + 40 * chaos_param  # oscillates between 80 and 120
    points = rose_curve_points(k_val, num_points=300, scale=scale_val)
    if len(points) > 1:
        pygame.draw.lines(screen, CURVE_COLOR, True, points, 2)

    # --- Fractal Bloom: At selected cell clusters ---
    # Identify clusters by checking if a cell is alive and has few alive neighbors (edge)
    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            if grid[i, j] == 1:
                # Count alive neighbors
                total = sum([
                    grid[(i + di) % GRID_HEIGHT, (j + dj) % GRID_WIDTH]
                    for di in (-1, 0, 1) for dj in (-1, 0, 1)
                    if not (di == 0 and dj == 0)
                ])
                if total <= 1:
                    # Draw a small fractal bloom at this location (map cell center to pixel coordinates)
                    x = int(j * CELL_SIZE + CELL_SIZE // 2)
                    y = int(i * CELL_SIZE + CELL_SIZE // 2)
                    # Depth of fractal based on chaos_param
                    depth = int(3 + 3 * chaos_param)
                    draw_fractal(screen, x, y, CELL_SIZE * 2, -math.pi/2, depth)

    # --- Poetic Layer: Render poem text ---
    # Fade effect: use a sinusoidal function of frame count to modulate alpha
    alpha = int(128 + 127 * math.sin(frame_count / 20))
    render_poem(screen, font, alpha)

    pygame.display.flip()

pygame.quit()
