import pygame
import numpy as np
import mido
import random
from pygame.locals import *
from scipy.signal import convolve2d

try:
    import cupy as cp
    gpu_available = True
except ImportError:
    gpu_available = False
    import numba

# --- Configuration ---
GRID_SIZE = 40
CELL_SIZE = 15
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE
BACKGROUND_COLOR = (0, 0, 0)
ALIVE_COLOR_MAP = {(0,): (0, 0, 0), (1,): (255, 255, 255)} # For GF(2)
FRACTAL_COLOR = (50, 50, 50)
MIDI_DEVICE_NAME = None
PRIME_FIELD = 2

# --- Algebraic Geometry: Polynomial Constraint Functions ---
def constraint_no_adjacent_vertical(grid, r, c, p):
    rows = grid.shape[0]
    x = lambda r_idx, c_idx: grid[r_idx % rows, c_idx]
    return (x(r, c) * x(r + 1, c)) % p

def constraint_no_adjacent_horizontal(grid, r, c, p):
    cols = grid.shape[1]
    x = lambda r_idx, c_idx: grid[r_idx, c_idx % cols]
    return (x(r, c) * x(r, c + 1)) % p

def constraint_sum_three_vertical_zero(grid, r, c, p):
    rows = grid.shape[0]
    x = lambda r_idx, c_idx: grid[r_idx % rows, c_idx]
    return (x(r, c) + x(r + 1, c) + x(r + 2, c)) % p

def constraint_sum_three_horizontal_zero(grid, r, c, p):
    cols = grid.shape[1]
    x = lambda r_idx, c_idx: grid[r_idx, c_idx % cols]
    return (x(r, c) + x(r, c + 1) + x(r, c + 2)) % p

# --- Game of Life Guided by Polynomial Constraints (NumPy Edition) ---
class GameOfLifePolynomialConstraintsNumPy:
    def __init__(self, grid_size, p):
        self.grid_size = grid_size
        self.p = p
        self.use_gpu = gpu_available
        if self.use_gpu:
            self.state_gpu = cp.asarray(np.random.randint(0, p, (grid_size, grid_size)).astype(np.int8))
        else:
            self.state_cpu = np.random.randint(0, p, (grid_size, grid_size)).astype(np.int8)
        self.time = 0
        self.active_constraints = []
        self.constraint_functions = {
            "no_adjacent_vertical": constraint_no_adjacent_vertical,
            "no_adjacent_horizontal": constraint_no_adjacent_horizontal,
            "sum_three_vertical_zero": constraint_sum_three_vertical_zero,
            "sum_three_horizontal_zero": constraint_sum_three_horizontal_zero,
        }

    def get_state(self):
        return self.state_gpu.get() if self.use_gpu else self.state_cpu

    def update(self):
        state = self.get_state()
        next_state = np.copy(state)
        rows, cols = state.shape

        for r in range(rows):
            for c in range(cols):
                violation = 0
                for constraint_name in self.active_constraints:
                    constraint_func = self.constraint_functions.get(constraint_name)
                    if constraint_func:
                        violation = (violation + constraint_func(state, r, c, self.p)) % self.p

                current_cell = state[r, c]
                if violation > 0:
                    if random.random() < 0.4:
                        next_state[r, c] = 1 - current_cell
                elif random.random() < 0.1 and current_cell == 0:
                    next_state[r, c] = 1

        if self.use_gpu:
            self.state_gpu = cp.asarray(next_state.astype(np.int8))
        else:
            self.state_cpu = next_state.astype(np.int8)

        self.time += 1
        self.learn()

    def learn(self):
        global poem_stage
        current_line_index = poem_stage % len(poem)
        current_line = poem[current_line_index]
        self.active_constraints = []

        if "racine ni chemin" in current_line:
            self.active_constraints.append("no_adjacent_vertical")
        if "vagabonde" in current_line:
            self.active_constraints.append("no_adjacent_horizontal")
        if "dissolvent au matin" in current_line:
            self.active_constraints.append("sum_three_vertical_zero")
        if "l’air du temps" in current_line:
            self.active_constraints.append("sum_three_horizontal_zero")
        if "incertain" in current_line:
            self.active_constraints = ["no_adjacent_vertical", "no_adjacent_horizontal"]

    def get_display_state(self):
        return self.get_state()

# --- Poem for Guidance --- (Same as before)
poem = [
    "Le printemps réveille en moi un désir léger, sans racine ni chemin,",
    "comme une âme bohémienne vagabonde, cherchant l’instant, sans destin.",
    "Les mots se dissolvent au matin, les projets se perdent en l’air du temps,",
    "et je reste, incertain, face à l’éclat d’un moment fuyant.",
    "Tel un cheval libre apprivoisé, vendu aux ombres d’un conte désenchanté,",
    "la raison finit par l’enchaîner à une tristesse de plus, à jamais."
]
poem_stage = 0
stage_duration = 80

# --- Visualization ---
def draw_grid(screen, game_of_life):
    display_state = game_of_life.get_display_state()
    global poem_stage
    line = poem[poem_stage % len(poem)]
    pygame.display.set_caption(f"AI Spring: Stage {poem_stage}, Line: '{line}', Constraints: {game_of_life.active_constraints} (GPU: {game_of_life.use_gpu})")

    for row in range(game_of_life.grid_size):
        for col in range(game_of_life.grid_size):
            x = col * CELL_SIZE
            y = row * CELL_SIZE
            state_value = display_state[row, col]
            violation = 0
            for constraint_name in game_of_life.active_constraints:
                constraint_func = game_of_life.constraint_functions.get(constraint_name)
                if constraint_func:
                    violation = (violation + constraint_func(display_state, row, col, PRIME_FIELD)) % PRIME_FIELD

            color = ALIVE_COLOR_MAP.get((state_value,), BACKGROUND_COLOR)
            if violation > 0:
                color = (255, 0, 0) # Indicate violation
            pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))

# --- Main Loop ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    game = GameOfLifePolynomialConstraintsNumPy(GRID_SIZE, PRIME_FIELD)

    running = True
    clock = pygame.time.Clock()
    global poem_stage

    while running:
        screen.fill(BACKGROUND_COLOR)

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            if event.type == KEYDOWN and event.key == K_SPACE:
                game.update()
            if event.type == KEYDOWN and event.key == K_r:
                game = GameOfLifePolynomialConstraintsNumPy(GRID_SIZE, PRIME_FIELD)

        game.update()
        draw_grid(screen, game)
        pygame.display.flip()
        clock.tick(60)

        if game.time % stage_duration == 0:
            poem_stage += 1

    pygame.quit()

if __name__ == '__main__':
    main()
