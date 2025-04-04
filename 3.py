import pygame
import numpy as np
import math
import random
from pygame import gfxdraw

# --- Configuration ---
WIDTH, HEIGHT = 800, 600
GRID_SIZE = 100         # Game of Life grid resolution
CELL_SIZE = WIDTH // GRID_SIZE
FPS = 15

# Colors (spring tones)
BACKGROUND_COLOR = (230, 250, 240)
LIVE_COLOR = (50, 150, 80)
FRACAL_COLOR = (200, 100, 150)
TEXT_COLOR = (30, 30, 30)

# Poem lines to display (the full poem)
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

# --- Logistic Map (Chaos Engine) ---
class LogisticMap:
    def __init__(self, r=3.9, x0=0.5):
        self.r = r
        self.x = x0

    def update(self):
        self.x = self.r * self.x * (1 - self.x)
        return self.x

# --- Game of Life ---
class GameOfLife:
    def __init__(self, grid_size):
        # Initialize grid randomly with 0 or 1
        self.grid_size = grid_size
        self.grid = np.random.choice([0, 1], (grid_size, grid_size))
    
    def step(self):
        # Compute next state using periodic boundary conditions
        new_grid = np.zeros_like(self.grid)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Count neighbors with wrap-around (toroidal grid)
                total = 0
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        total += self.grid[(i+di)%self.grid_size, (j+dj)%self.grid_size]
                # Apply Game of Life rules
                if self.grid[i, j] == 1 and (total == 2 or total == 3):
                    new_grid[i, j] = 1
                elif self.grid[i, j] == 0 and total == 3:
                    new_grid[i, j] = 1
                else:
                    new_grid[i, j] = 0
        self.grid = new_grid

    def density(self):
        return np.sum(self.grid) / (self.grid_size * self.grid_size)

# --- Fractal Drawer with Algebraic (Rose curve) modulation ---
def draw_fractal(surface, x, y, angle, length, depth, chaos_mod):
    if depth == 0 or length < 2:
        return
    # The branch angle is modulated by an algebraic rose curve:
    # rose angle: theta = base_angle + chaos_mod * sin(k * angle)
    k = 4  # number of petals factor
    mod_angle = angle + chaos_mod * math.sin(k * angle)
    # Compute end point
    end_x = x + length * math.cos(mod_angle)
    end_y = y + length * math.sin(mod_angle)
    # Draw line (use anti-aliased line for smoothness)
    pygame.gfxdraw.line(surface, int(x), int(y), int(end_x), int(end_y), FRACAL_COLOR)
    # Recursive branch: two new branches with slightly varied angles
    new_length = length * 0.7
    draw_fractal(surface, end_x, end_y, mod_angle - 0.3, new_length, depth - 1, chaos_mod)
    draw_fractal(surface, end_x, end_y, mod_angle + 0.3, new_length, depth - 1, chaos_mod)

# --- Poem Overlay ---
class PoemOverlay:
    def __init__(self, lines, font, surface_rect):
        self.lines = lines
        self.font = font
        self.surface_rect = surface_rect
        self.line_surfaces = [font.render(line, True, TEXT_COLOR) for line in lines]
        self.alpha = 255  # full opacity

    def draw(self, surface, time_phase):
        # Fade each line in/out slowly based on a periodic function (e.g., cosine)
        alpha = int(128 + 127 * math.cos(time_phase))
        for idx, line_surf in enumerate(self.line_surfaces):
            surf = line_surf.copy()
            surf.set_alpha(alpha)
            # Position lines centered horizontally
            x = self.surface_rect.centerx - surf.get_width() // 2
            # Stack lines vertically
            y = 20 + idx * (surf.get_height() + 5)
            surface.blit(surf, (x, y))

# --- Main Visualization ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Mathematical Opus: Le printemps réveille en moi")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("serif", 18)

    # Create instances of our modules
    life = GameOfLife(GRID_SIZE)
    logistic = LogisticMap()
    poem_overlay = PoemOverlay(POEM_LINES, font, screen.get_rect())

    running = True
    frame = 0

    while running:
        clock.tick(FPS)
        frame += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Update modules ---
        life.step()                # Update Game of Life
        chaos_val = logistic.update()  # Chaos parameter evolves in [0,1]
        density = life.density()   # Get density of life
        # Map density to a fractal intensity factor (if high density, more fractal branching)
        fractal_intensity = max(0.2, min(1.0, density * 2))
        
        # --- Render background ---
        screen.fill(BACKGROUND_COLOR)
        
        # --- Render Game of Life as a subtle backdrop ---
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if life.grid[i, j] == 1:
                    rect = pygame.Rect(j*CELL_SIZE, i*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(screen, LIVE_COLOR, rect)
        
        # --- Integrate Algebraic & Fractal Growth ---
        # Use high-density regions to seed fractal growth.
        # For simplicity, sample random "seed" points weighted by density.
        num_seeds = int(5 + fractal_intensity * 15)
        for _ in range(num_seeds):
            # Random seed point (weighted towards the center for organic growth)
            seed_x = WIDTH//2 + random.randint(-WIDTH//4, WIDTH//4)
            seed_y = HEIGHT//2 + random.randint(-HEIGHT//4, HEIGHT//4)
            # Use chaos_val to determine branch initial angle and length (scaled by density)
            angle = random.uniform(0, 2*math.pi)
            length = 20 + chaos_val * 30 * fractal_intensity
            depth = 4 + int(chaos_val * 4)
            draw_fractal(screen, seed_x, seed_y, angle, length, depth, chaos_val)
        
        # --- Poem Overlay ---
        # Use a periodic function (based on frame count) for alpha modulation
        time_phase = frame / 20.0
        poem_overlay.draw(screen, time_phase)

        # --- Update display ---
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
