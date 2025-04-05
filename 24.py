import pygame
import numpy as np
import mido
import random
from pygame.locals import *
from scipy.signal import convolve2d
from sympy import symbols, Poly, groebner
from sympy.domains import ZZ

try:
    import cupy as cp
    gpu_available = True
except ImportError:
    gpu_available = False
    import numba

# --- Configuration ---
GRID_SIZE = 30
CELL_SIZE = 20
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE
BACKGROUND_COLOR = (0, 0, 0)
ALIVE_COLOR_MAP = {(0,): (0, 0, 0), (1,): (255, 255, 255)} # For GF(2)
FRACTAL_COLOR = (50, 50, 50)
MIDI_DEVICE_NAME = None
PRIME_FIELD = 2

# --- Algebraic Geometry: Gröbner Basis with SymPy (Local) ---
def get_local_symbols():
    return symbols('x00 x01 x02 x10 x11 x12 x20 x21 x22')

def compute_groebner_basis(polynomials):
    x = get_local_symbols()
    polys = [Poly(p, x, domain=ZZ.mod(PRIME_FIELD)) for p in polynomials]
    return groebner(polys, x, domain=ZZ.mod(PRIME_FIELD))

# --- Game of Life Guided by Gröbner Basis (Local) ---
class GameOfLifeGroebner:
    def __init__(self, grid_size, p):
        self.grid_size = grid_size
        self.p = p
        self.use_gpu = gpu_available
        if self.use_gpu:
            self.state_gpu = cp.asarray(np.random.randint(0, p, (grid_size, grid_size)).astype(np.int8))
        else:
            self.state_cpu = np.random.randint(0, p, (grid_size, grid_size)).astype(np.int8)
        self.time = 0
        self.local_symbols = get_local_symbols()
        self.basis = None
        self.target_pattern = "line" # Guided by poem

    def get_local_state(self, grid, r, c):
        rows, cols = grid.shape
        return [
            int(grid[(r - 1) % rows, (c - 1) % cols]), int(grid[(r - 1) % rows, c % cols]), int(grid[(r - 1) % rows, (c + 1) % cols]),
            int(grid[r % rows, (c - 1) % cols]),     int(grid[r % rows, c % cols]),     int(grid[r % rows, (c + 1) % cols]),
            int(grid[(r + 1) % rows, (c - 1) % cols]), int(grid[(r + 1) % rows, c % cols]), int(grid[(r + 1) % rows, (c + 1) % cols])
        ]

    def update(self):
        if self.use_gpu:
            state = self.state_gpu.get()
        else:
            state = self.state_cpu
        next_state = np.copy(state)
        rows, cols = state.shape

        if self.basis:
            for r in range(rows):
                for c in range(cols):
                    local_state_values = self.get_local_state(state, r, c)
                    symbol_values = dict(zip(self.local_symbols, local_state_values))
                    basis_evaluations = [poly.eval(symbol_values) % self.p for poly in self.basis]

                    center_r, center_c = r, c
                    if basis_evaluations and any(eval_val != 0 for eval_val in basis_evaluations):
                        # Try to flip the center cell to move towards the variety
                        next_state[center_r % rows, center_c % cols] = 1 - state[center_r % rows, center_c % cols]
                    elif random.random() < 0.05:
                        next_state[center_r % rows, center_c % cols] = random.randint(0, 1)

        else: # Basic Game of Life if no basis
            neighbor_sum = convolve2d(state, np.ones((3, 3)), mode='same', boundary='wrap') - state
            for r in range(rows):
                for c in range(cols):
                    live_neighbors = neighbor_sum[r, c]
                    if state[r, c] == 1:
                        next_state[r, c] = 1 if 2 <= live_neighbors <= 3 else 0
                    else:
                        next_state[r, c] = 1 if live_neighbors == 3 else 0

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
        polynomials = []

        if "racine ni chemin" in current_line:
            self.target_pattern = "line"
            x = self.local_symbols
            # Ideal for a horizontal line in the center: x10=x11=x12=1, others 0
            polynomials = [x[0], x[2], x[3] - 1, x[4] - 1, x[5] - 1, x[6], x[7], x[8]]
        elif "vagabonde" in current_line:
            self.target_pattern = "block"
            x = self.local_symbols
            # Ideal for a block in the center: x11=x12=x21=x22=1, others 0
            polynomials = [x[0], x[1], x[3], x[4] - 1, x[5] - 1, x[6] - 1, x[7] - 1, x[8]]

        if polynomials:
            self.basis = compute_groebner_basis(polynomials)
        else:
            self.basis = None

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
stage_duration = 120

# --- Visualization ---
def draw_grid(screen, game_of_life):
    display_state = game_of_life.get_display_state()
    global poem_stage
    line = poem[poem_stage % len(poem)]
    pygame.display.set_caption(f"AI Spring: Stage {poem_stage}, Line: '{line}', Pattern: {game_of_life.target_pattern}, Basis: {len(game_of_life.basis) if game_of_life.basis else 0} (GPU: {game_of_life.use_gpu})")

    rows, cols = display_state.shape
    for r in range(rows):
        for c in range(cols):
            x = c * CELL_SIZE
            y = r * CELL_SIZE
            state_value = display_state[r, c]
            color = ALIVE_COLOR_MAP.get((state_value,), BACKGROUND_COLOR)
            if game_of_life.basis:
                local_state_values = game_of_life.get_local_state(display_state, r, c)
                symbol_values = dict(zip(game_of_life.local_symbols, local_state_values))
                basis_evaluations = [poly.eval(symbol_values) % PRIME_FIELD for poly in game_of_life.basis]
                if any(eval_val != 0 for eval_val in basis_evaluations):
                    color = (255, 0, 0) # Indicate not on the variety locally

            pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))

# --- Main Loop ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    game = GameOfLifeGroebner(GRID_SIZE, PRIME_FIELD)

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
                game = GameOfLifeGroebner(GRID_SIZE, PRIME_FIELD)

        game.update()
        draw_grid(screen, game)
        pygame.display.flip()
        clock.tick(30) # Slow down for computation

        if game.time % stage_duration == 0:
            poem_stage += 1

    pygame.quit()

if __name__ == '__main__':
    main()
