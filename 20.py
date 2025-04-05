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
GRID_SIZE = 60
CELL_SIZE = 10
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE
BACKGROUND_COLOR = (0, 0, 0)
ALIVE_COLOR_MAP = {(0,): (0, 0, 0), (1,): (255, 255, 255)} # For GF(2)
FRACTAL_COLOR = (50, 50, 50)
MIDI_DEVICE_NAME = None
PRIME_FIELD = 2

# --- Algebraic Geometry: Simple Ideal Evaluation ---
def evaluate_ideal(grid, r, c, p, polynomials):
    value = 0
    rows, cols = grid.shape
    x = lambda r_idx, c_idx: int(grid[r_idx % rows, c_idx % cols])

    for poly_type in polynomials:
        if poly_type == "down":
            value = (value + x(r, c) * (1 + x(r + 1, c))) % p
        elif poly_type == "up":
            value = (value + x(r, c) * (1 + x(r - 1, c))) % p
        elif poly_type == "right":
            value = (value + x(r, c) * (1 + x(r, c + 1))) % p
        elif poly_type == "left":
            value = (value + x(r, c) * (1 + x(r, c - 1))) % p
        elif poly_type == "diag1": # (i,j) and (i+1, j+1)
            value = (value + x(r, c) * (1 + x(r + 1, c + 1))) % p
        elif poly_type == "diag2": # (i,j) and (i+1, j-1)
            value = (value + x(r, c) * (1 + x(r + 1, c - 1))) % p
    return value

# --- Game of Life Guided by Ideal ---
class GameOfLifeIdealGuided:
    def __init__(self, grid_size, p):
        self.grid_size = grid_size
        self.p = p
        self.use_gpu = gpu_available
        if self.use_gpu:
            self.state_gpu = cp.asarray(np.random.randint(0, p, (grid_size, grid_size)).astype(np.int8))
        else:
            self.state_cpu = np.random.randint(0, p, (grid_size, grid_size)).astype(np.int8)
        self.time = 0
        self.polynomials = ["down", "right"] # Initial ideal

    def get_state(self):
        return self.state_gpu.get() if self.use_gpu else self.state_cpu

    def update(self):
        state = self.get_state()
        next_state = np.copy(state)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                ideal_value = evaluate_ideal(state, r, c, self.p, self.polynomials)
                current_cell = state[r, c]

                # Probabilistic update to reduce ideal value
                if ideal_value > 0:
                    if random.random() < 0.3:
                        next_state[r, c] = 0
                    elif random.random() < 0.1 and current_cell == 0:
                        next_state[r, c] = 1
                elif random.random() < 0.05 and current_cell == 0:
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

        if "printemps" in current_line:
            self.polynomials = ["down", "right"]
        elif "bohémienne" in current_line:
            self.polynomials = ["up", "left"]
        elif "dissolvent" in current_line:
            self.polynomials = ["diag1"]
        elif "temps" in current_line:
            self.polynomials = ["diag2"]
        elif "incertain" in current_line:
            self.polynomials = ["down", "left", "right", "up"]

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
    pygame.display.set_caption(f"AI Spring: Stage {poem_stage}, Line: '{line}', Ideal: {game_of_life.polynomials} (GPU: {game_of_life.use_gpu})")

    for row in range(game_of_life.grid_size):
        for col in range(game_of_life.grid_size):
            x = col * CELL_SIZE
            y = row * CELL_SIZE
            state_value = display_state[row, col]
            ideal_value = evaluate_ideal(display_state, row, col, PRIME_FIELD, game_of_life.polynomials)
            color = ALIVE_COLOR_MAP.get((state_value,), BACKGROUND_COLOR)
            if ideal_value > 0:
                color = (255, 0, 0) # Indicate non-zero ideal value
            pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))

# --- Main Loop ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    game = GameOfLifeIdealGuided(GRID_SIZE, PRIME_FIELD)

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
                game = GameOfLifeIdealGuided(GRID_SIZE, PRIME_FIELD)

        game.update()
        draw_grid(screen, game)
        pygame.display.flip()
        clock.tick(60)

        if game.time % stage_duration == 0:
            poem_stage += 1

    pygame.quit()

if __name__ == '__main__':
    main()
