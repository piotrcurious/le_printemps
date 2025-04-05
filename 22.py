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
GRID_SIZE = 50
CELL_SIZE = 12
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE
BACKGROUND_COLOR = (0, 0, 0)
ALIVE_COLOR_MAP = {(0,): (0, 0, 0), (1,): (255, 255, 255)} # For GF(2)
FRACTAL_COLOR = (50, 50, 50)
MIDI_DEVICE_NAME = None
PRIME_FIELD = 2

# --- Algebraic Geometry: Direct Polynomial Constraints (Ideal Analogy) ---
def check_constraints(grid, r, c, p, constraints):
    rows, cols = grid.shape
    x = lambda r_idx, c_idx: int(grid[r_idx % rows, c_idx % cols])
    satisfied_count = 0
    total_constraints = 0

    if "no_adjacent_vertical" in constraints:
        total_constraints += 1
        if not (x(r, c) == 1 and x(r + 1, c) == 1):
            satisfied_count += 1
    if "no_adjacent_horizontal" in constraints:
        total_constraints += 1
        if not (x(r, c) == 1 and x(r, c + 1) == 1):
            satisfied_count += 1
    if "sum_three_one_vertical" in constraints:
        total_constraints += 1
        if (x(r, c) + x(r + 1, c) + x(r + 2, c)) % p == 1:
            satisfied_count += 1
    if "sum_three_one_horizontal" in constraints:
        total_constraints += 1
        if (x(r, c) + x(r, c + 1) + x(r, c + 2)) % p == 1:
            satisfied_count += 1

    return satisfied_count, total_constraints

# --- Game of Life Guided by Polynomial Constraints (Ideal Analogy) ---
class GameOfLifeIdealConstraints:
    def __init__(self, grid_size, p):
        self.grid_size = grid_size
        self.p = p
        self.use_gpu = gpu_available
        if self.use_gpu:
            self.state_gpu = cp.asarray(np.random.randint(0, p, (grid_size, grid_size)).astype(np.int8))
        else:
            self.state_cpu = np.random.randint(0, p, (grid_size, grid_size)).astype(np.int8)
        self.time = 0
        self.active_constraints = ["no_adjacent_vertical", "no_adjacent_horizontal"]

    def get_state(self):
        return self.state_gpu.get() if self.use_gpu else self.state_cpu

    def update(self):
        if self.use_gpu:
            state = self.state_gpu.get()
        else:
            state = self.state_cpu
        next_state = np.copy(state)
        rows, cols = state.shape

        for r in range(rows):
            for c in range(cols):
                satisfied, total = check_constraints(state, r, c, self.p, self.active_constraints)
                current_cell = state[r, c]

                if total > 0:
                    satisfaction_ratio = satisfied / total
                    if satisfaction_ratio < 0.5 and random.random() < 0.3:
                        next_state[r, c] = 1 - current_cell # Try flipping to satisfy more
                    elif satisfaction_ratio > 0.8 and random.random() < 0.1:
                        next_state[r, c] = 1 - current_cell # Introduce some randomness to explore

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
        if "se dissolvent au matin" in current_line:
            self.active_constraints.append("sum_three_one_vertical")
        if "l’air du temps" in current_line:
            self.active_constraints.append("sum_three_one_horizontal")
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
            satisfied, total = check_constraints(display_state, row, col, PRIME_FIELD, game_of_life.active_constraints)
            color = ALIVE_COLOR_MAP.get((state_value,), BACKGROUND_COLOR)
            if total > 0:
                ratio = satisfied / total
                intensity = int(255 * ratio)
                color = (0, intensity, 0) # Green intensity based on satisfaction
            pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))

# --- Main Loop ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    game = GameOfLifeIdealConstraints(GRID_SIZE, PRIME_FIELD)

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
                game = GameOfLifeIdealConstraints(GRID_SIZE, PRIME_FIELD)

        game.update()
        draw_grid(screen, game)
        pygame.display.flip()
        clock.tick(60)

        if game.time % stage_duration == 0:
            poem_stage += 1

    pygame.quit()

if __name__ == '__main__':
    main()
