import pygame
import numpy as np
import mido
import random
from pygame.locals import *
import time
from scipy.spatial import distance

# --- Configuration ---
WIDTH, HEIGHT = 800, 600
GRID_SIZE = 8  # Increased grid resolution for more detail
CELL_SIZE = WIDTH // GRID_SIZE
BACKGROUND_COLOR = (0, 0, 0)
FPS = 15

# --- Color Palettes ---
PALETTE_BRIGHT = [(66, 165, 245), (30, 136, 229), (5, 117, 186), (4, 78, 160), (3, 51, 113)] # Blues
PALETTE_ORGANIC = [(165, 214, 167), (129, 199, 132), (99, 163, 102), (67, 160, 71), (38, 139, 34)] # Greens
PALETTE_FIRE = [(255, 235, 59), (255, 193, 7), (255, 152, 0), (251, 140, 0), (245, 127, 23)] # Yellows/Oranges
CURRENT_PALETTE = PALETTE_ORGANIC

# --- Abstract Algebra Concepts (More Integrated) ---
# The Game of Life operates on a grid, which can be thought of as a module over a ring (e.g., Z/2Z).
# Algebraic geometry defines the shape within which the dynamics occur.
# Chaos theory emerges from the non-linear, iterative nature of the Game of Life rules within these constraints.

# --- Fractal Definition (Mandelbrot Set with Iteration Count) ---
def mandelbrot_iteration(c, max_iter=50): # Reduced max_iter for performance
    z = 0
    for i in range(max_iter):
        z = z * z + c
        if abs(z) > 2:
            return i
    return max_iter

def map_to_complex(x, y, width, height, zoom=1.0, offset_x=0.0, offset_y=0.0):
    real = (x - width / 2) * 4.0 / width / zoom + offset_x
    imag = (y - height / 2) * 4.0 / height / zoom + offset_y
    return complex(real, imag)

# --- Algebraic Geometry Constraint as a Probability Field ---
def algebraic_probability(x, y, center_x, center_y, radius_x, radius_y, sharpness=2.0):
    dist = distance.euclidean((x, y), (center_x, center_y))
    normalized_dist = dist / max(radius_x, radius_y)
    # Using a Gaussian-like falloff for probability
    probability = np.exp(-sharpness * normalized_dist**2)
    return probability

# --- Game of Life Class (Improved) ---
class GameOfLife:
    def __init__(self, width, height, grid_size, fractal_function, algebraic_prob_function):
        self.width = width // grid_size
        self.height = height // grid_size
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.age = np.zeros((self.height, self.width), dtype=int) # Track cell age
        self.fractal_function = fractal_function
        self.algebraic_prob_function = algebraic_prob_function
        self.algebraic_center = (width // 2, height // 2)
        self.algebraic_radius = (width // 3, height // 3)
        self.algebraic_sharpness = 2.0

    def seed(self, locations):
        for row, col in locations:
            if 0 <= row < self.height and 0 <= col < self.width:
                self.grid[row, col] = 1
                self.age[row, col] = 1

    def get_neighbors(self, row, col):
        neighbors = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                r, c = row + i, col + j
                if 0 <= r < self.height and 0 <= c < self.width:
                    neighbors.append(self.grid[r, c])
        return neighbors

    def update(self):
        new_grid = np.zeros_like(self.grid)
        new_age = np.zeros_like(self.age)
        for row in range(self.height):
            for col in range(self.width):
                x = col * CELL_SIZE + CELL_SIZE // 2
                y = row * CELL_SIZE + CELL_SIZE // 2

                # Algebraic Geometry Influence (Probabilistic)
                algebraic_prob = self.algebraic_prob_function(x, y, self.algebraic_center[0], self.algebraic_center[1], self.algebraic_radius[0], self.algebraic_radius[1], self.algebraic_sharpness)
                if random.random() > algebraic_prob:
                    continue # Cell has a probability of not existing based on algebraic geometry

                # Fractal Influence (Survival Probability)
                c = map_to_complex(x, y, WIDTH, HEIGHT, zoom=1.5, offset_x=-0.5, offset_y=0.0)
                mandelbrot_iter = self.fractal_function(c)
                survival_factor = 1.0 - (mandelbrot_iter / 50.0) # Cells in deeper Mandelbrot regions less likely to survive

                neighbors = self.get_neighbors(row, col)
                live_neighbors = sum(neighbors)

                if self.grid[row, col] == 1:
                    if 2 <= live_neighbors <= 3 and random.random() < survival_factor:
                        new_grid[row, col] = 1
                        new_age[row, col] = self.age[row, col] + 1
                    else:
                        new_grid[row, col] = 0
                        new_age[row, col] = 0
                else:
                    if live_neighbors == 3 and random.random() < algebraic_prob: # Birth influenced by algebraic geometry
                        new_grid[row, col] = 1
                        new_age[row, col] = 1

        self.grid = new_grid
        self.age = new_age

    def draw(self, screen):
        for row in range(self.height):
            for col in range(self.width):
                if self.grid[row, col] == 1:
                    x = col * CELL_SIZE
                    y = row * CELL_SIZE
                    age_index = min(self.age[row, col] - 1, len(CURRENT_PALETTE) - 1)
                    color = CURRENT_PALETTE[age_index]
                    pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))

    def update_algebraic_center(self, center):
        self.algebraic_center = center

    def update_algebraic_radius(self, radius):
        self.algebraic_radius = radius

    def update_algebraic_sharpness(self, sharpness):
        self.algebraic_sharpness = sharpness

# --- MIDI Input Handling (Enhanced) ---
def get_midi_input():
    try:
        input_names = mido.get_input_names()
        if not input_names:
            print("No MIDI input devices found.")
            return None
        print("Available MIDI input devices:")
        for i, name in enumerate(input_names):
            print(f"{i}: {name}")
        port_index = 0  # You might want to let the user choose
        return mido.open_input(input_names[port_index])
    except Exception as e:
        print(f"Error opening MIDI input: {e}")
        return None

def handle_midi_input(game, midi_port):
    if midi_port:
        for msg in midi_port.iter_pending():
            if msg.type == 'note_on' and msg.velocity > 0:
                # Map MIDI note number to influence algebraic center
                center_x = WIDTH // 4 + (msg.note - 21) * (WIDTH // 2) // 88 # Map across X
                center_y = HEIGHT // 4 + (108 - msg.note) * (HEIGHT // 2) // 88 # Map across Y (inverted)
                game.update_algebraic_center((center_x, center_y))

                # Map MIDI velocity to influence algebraic sharpness (probability falloff)
                sharpness = 1.0 + msg.velocity / 127.0 * 5.0 # Range from 1 to 6
                game.update_algebraic_sharpness(sharpness)

                # Seed a few cells around the algebraic center based on the note
                seed_row = int(center_y // CELL_SIZE) + random.randint(-2, 2)
                seed_col = int(center_x // CELL_SIZE) + random.randint(-2, 2)
                game.seed([(seed_row, seed_col)])

            elif msg.type == 'control_change':
                if msg.control == 1: # Mod wheel controls algebraic radius
                    radius_factor = 0.5 + msg.value / 127.0 * 1.5 # Range from 0.5 to 2
                    game.update_algebraic_radius((int(WIDTH // 3 * radius_factor), int(HEIGHT // 3 * radius_factor)))
                elif msg.control == 7: # Volume controls global speed (FPS) - can be implemented in main loop
                    global FPS
                    FPS = 10 + int(msg.value / 127.0 * 20) # Range from 10 to 30

# --- Main Function ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Emergent Mathematical Opus")

    midi_port = get_midi_input()

    # Initialize Game of Life with fractal and algebraic probability
    game = GameOfLife(WIDTH, HEIGHT, GRID_SIZE, fractal_function=mandelbrot_iteration, algebraic_prob_function=algebraic_probability)

    running = True
    paused = False
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            if event.type == KEYDOWN:
                if event.key == K_SPACE:
                    paused = not paused
                if event.key == K_r:
                    game.grid = np.zeros_like(game.grid)
                    game.age = np.zeros_like(game.age)
                if event.key == K_c:
                    global CURRENT_PALETTE
                    CURRENT_PALETTE = random.choice([PALETTE_BRIGHT, PALETTE_ORGANIC, PALETTE_FIRE])

        if not paused:
            handle_midi_input(game, midi_port)
            game.update()

        screen.fill(BACKGROUND_COLOR)
        game.draw(screen)
        pygame.display.flip()

        clock.tick(FPS)

    if midi_port:
        midi_port.close()
    pygame.quit()

if __name__ == "__main__":
    main()
