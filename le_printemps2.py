import pygame
import numpy as np
import random
import math

# --- Configuration ---
WIDTH, HEIGHT = 800, 800
FPS = 15

# Game of Life grid dimensions
GRID_COLS, GRID_ROWS = 100, 100
CELL_SIZE = WIDTH // GRID_COLS

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
FRAC_COLOR = (200, 100, 240)
LIFE_COLOR = (50, 200, 50)

# Poem text (in French) to overlay
POEM_TEXT = (
    "Le printemps réveille en moi\n"
    "un désir léger, sans racine ni chemin,\n"
    "comme une âme bohémienne vagabonde,\n"
    "cherchant l’instant, sans destin.\n\n"
    "Les mots se dissolvent au matin,\n"
    "les projets se perdent en l’air du temps,\n"
    "et je reste, incertain,\n"
    "face à l’éclat d’un moment fuyant.\n\n"
    "Tel un cheval libre apprivoisé,\n"
    "vendu aux ombres d’un conte désenchanté,\n"
    "la raison finit par l’enchaîner\n"
    "à une tristesse de plus, à jamais."
)

# --- Helper functions ---
def linmap(val, src_min, src_max, dst_min, dst_max):
    """Linearly map a value from one range to another."""
    return dst_min + ((val - src_min) / (src_max - src_min)) * (dst_max - dst_min)

# --- Game of Life ---
class GameOfLife:
    def __init__(self, cols, rows):
        self.cols = cols
        self.rows = rows
        # Start with a random grid
        self.grid = np.random.choice([0, 1], size=(rows, cols), p=[0.8, 0.2])
    
    def update(self):
        new_grid = np.zeros_like(self.grid)
        for y in range(self.rows):
            for x in range(self.cols):
                # Count neighbors with toroidal (wrap-around) conditions
                total = np.sum(self.grid[(y-1)%self.rows:(y+2)%self.rows, (x-1)%self.cols:(x+2)%self.cols]) - self.grid[y, x]
                if self.grid[y, x] == 1 and total in (2, 3):
                    new_grid[y, x] = 1
                elif self.grid[y, x] == 0 and total == 3:
                    new_grid[y, x] = 1
                else:
                    new_grid[y, x] = 0
        self.grid = new_grid

    def get_live_cells(self):
        # Return a list of (x, y) positions for live cells
        live_cells = np.argwhere(self.grid == 1)
        return [(x, y) for y, x in live_cells]

# --- Logistic Map (chaotic parameter generator) ---
class LogisticMap:
    def __init__(self, x0=0.5, r=3.9):
        self.x = x0
        self.r = r

    def step(self):
        self.x = self.r * self.x * (1 - self.x)
        return self.x

# --- Fractal Branch Generator ---
def generate_fractal_branch(seed, chaos_param, iterations=10):
    """
    Given a seed (derived from a live cell coordinate) and a chaotic parameter,
    generate a list of points (complex numbers) using a Mandelbrot-style iteration.
    The seed is mapped into the complex plane, and chaos_param modulates the constant.
    """
    # Map cell coordinate seed (tuple: (x, y)) into a small region of the complex plane.
    # We use the grid dimensions to produce a c value in a region that produces fractal behavior.
    grid_x, grid_y = seed
    # Normalize grid coordinates and center them:
    norm_x = linmap(grid_x, 0, GRID_COLS, -1.0, 1.0)
    norm_y = linmap(grid_y, 0, GRID_ROWS, -1.0, 1.0)
    # Use the logistic output to modulate the constant.
    c = complex(norm_x * (0.5 + chaos_param), norm_y * (0.5 + (1 - chaos_param)))
    
    # Start from z = 0, iterate to form the branch
    z = 0 + 0j
    points = []
    for i in range(iterations):
        z = z**2 + c
        # Save the point (scale to screen coordinates later)
        points.append(z)
    return points

def complex_to_screen(z, center, scale):
    """
    Convert a complex number z to screen coordinates.
    center: (x, y) tuple for the center of the screen
    scale: scaling factor
    """
    x = center[0] + int(z.real * scale)
    y = center[1] + int(z.imag * scale)
    return (x, y)

# --- Main Pygame Visualization ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Mathematical Opus: Nature, Chaos & Poetic Geometry")
    clock = pygame.time.Clock()

    # Set up fonts for the poem overlay
    font = pygame.font.SysFont("serif", 16)
    poem_lines = POEM_TEXT.split("\n")
    
    # Create instances
    life = GameOfLife(GRID_COLS, GRID_ROWS)
    logistic = LogisticMap(x0=random.random(), r=3.9)
    
    # Off-screen surface for drawing fractal branches with alpha blending
    fractal_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    fractal_surf.fill((0, 0, 0, 0))
    
    running = True
    frame_count = 0
    center = (WIDTH // 2, HEIGHT // 2)
    scale = 100  # scale factor for fractal branch drawing

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Clear main screen
        screen.fill(BLACK)

        # --- Update Game of Life every frame ---
        life.update()
        live_cells = life.get_live_cells()

        # --- Update chaotic parameter ---
        chaos_param = logistic.step()

        # --- For a subset of live cells, generate fractal branches ---
        # To avoid overloading, randomly sample a few cells each frame.
        sampled_cells = random.sample(live_cells, min(10, len(live_cells)))
        for cell in sampled_cells:
            branch_points = generate_fractal_branch(cell, chaos_param, iterations=12)
            # Convert branch points to screen coordinates and draw a line
            if len(branch_points) >= 2:
                pts = [complex_to_screen(z, center, scale) for z in branch_points]
                pygame.draw.lines(fractal_surf, FRAC_COLOR, False, pts, 1)

        # --- Fade the fractal surface slowly to merge past evolutions ---
        fade = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        fade.fill((0, 0, 0, 15))  # slight transparency fade-out
        fractal_surf.blit(fade, (0, 0))

        # --- Draw Game of Life grid as subtle dots (optional) ---
        for (x, y) in live_cells:
            px = int(x * CELL_SIZE)
            py = int(y * CELL_SIZE)
            pygame.draw.rect(screen, LIFE_COLOR, (px, py, CELL_SIZE, CELL_SIZE))

        # Blit the fractal layer onto the screen
        screen.blit(fractal_surf, (0, 0))

        # --- Overlay the poem text ---
        text_y = 10
        for line in poem_lines:
            text_surface = font.render(line, True, WHITE)
            screen.blit(text_surface, (10, text_y))
            text_y += text_surface.get_height() + 2

        pygame.display.flip()
        clock.tick(FPS)
        frame_count += 1

    pygame.quit()

if __name__ == "__main__":
    main()
