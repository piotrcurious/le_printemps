import pygame
import numpy as np
import mido
import random
from pygame.locals import *
from scipy.signal import convolve2d
from sympy import symbols, Poly, groebner
from sympy.domains import ZZ
import multiprocessing as mp

try:
    import cupy as cp
    gpu_available = True
except ImportError:
    gpu_available = False
    import numba

# --- Configuration ---
GRID_SIZE = 80
CELL_SIZE = 8
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE
BACKGROUND_COLOR = (0, 0, 0)
ALIVE_COLOR_MAP = {(0,): (0, 0, 0), (1,): (255, 255, 255)} # For GF(2)
FRACTAL_COLOR = (50, 50, 50)
MIDI_DEVICE_NAME = None
PRIME_FIELD = 2

# --- Algebraic Geometry: Simple Polynomial Update Rule (Algebraic Dynamical System) ---
def polynomial_update_rule(grid, r, c, p):
    rows, cols = grid.shape
    x = lambda r_idx, c_idx: int(grid[r_idx % rows, c_idx % cols])
    # A very simple polynomial update: XOR of neighbors
    return (x(r - 1, c) + x(r + 1, c) + x(r, c - 1) + x(r, c + 1)) % p

# --- Game of Life as Algebraic Dynamical System ---
class GameOfLifeAlgebraicSystem:
    def __init__(self, grid_size, p):
        self.grid_size = grid_size
        self.p = p
        self.use_gpu = gpu_available
        if self.use_gpu:
            self.state_gpu = cp.asarray(np.random.randint(0, p, (grid_size, grid_size)).astype(np.int8))
        else:
            self.state_cpu = np.random.randint(0, p, (grid_size, grid_size)).astype(np.int8)
        self.time = 0

    def get_state(self):
        return self.state_gpu.get() if self.use_gpu else self.state_cpu

    def update(self):
        state = self.get_state()
        next_state = np.zeros_like(state)
        rows, cols = state.shape
        for r in range(rows):
            for c in range(cols):
                next_state[r, c] = polynomial_update_rule(state, r, c, self.p)

        if self.use_gpu:
            self.state_gpu = cp.asarray(next_state.astype(np.int8))
        else:
            self.state_cpu = next_state.astype(np.int8)

        self.time += 1

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
    pygame.display.set_caption(f"AI Spring: Stage {poem_stage}, Line: '{line}', System: Algebraic (GPU: {game_of_life.use_gpu})")

    rows, cols = display_state.shape
    for r in range(rows):
        for c in range(cols):
            x = c * CELL_SIZE
            y = r * CELL_SIZE
            state_value = display_state[r, c]
            color = ALIVE_COLOR_MAP.get((state_value,), BACKGROUND_COLOR)
            pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))

# --- Main Loop ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    game = GameOfLifeAlgebraicSystem(GRID_SIZE, PRIME_FIELD)

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
                game = GameOfLifeAlgebraicSystem(GRID_SIZE, PRIME_FIELD)

        game.update()
        draw_grid(screen, game)
        pygame.display.flip()
        clock.tick(60)

        if game.time % stage_duration == 0:
            poem_stage += 1

    pygame.quit()

if __name__ == '__main__':
    main()
