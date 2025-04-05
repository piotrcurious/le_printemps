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
GRID_SIZE = 60
CELL_SIZE = 10
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE
BACKGROUND_COLOR = (0, 0, 0)
ALIVE_COLOR_MAP = {(0,): (0, 0, 0), (1,): (255, 255, 255)} # For GF(2)
FRACTAL_COLOR = (50, 50, 50)
MIDI_DEVICE_NAME = None
PRIME_FIELD = 2
LOCAL_SIZE = 3 # Size of the local neighborhood for Gröbner basis

# --- Algebraic Geometry: Gröbner Basis with SymPy (Local) ---
def get_local_symbols(size):
    n = size * size
    return symbols(f'x0:{n}')

def compute_groebner_basis(polynomials, size):
    n = size * size
    x = get_local_symbols(size)
    polys = [Poly(p, x, domain=ZZ.mod(PRIME_FIELD)) for p in polynomials]
    return groebner(polys, x, domain=ZZ.mod(PRIME_FIELD))

def get_local_state_np(grid, r, c, size):
    rows, cols = grid.shape
    offset = size // 2
    state = []
    for i in range(-offset, offset + 1):
        for j in range(-offset, offset + 1):
            state.append(int(grid[(r + i) % rows, (c + j) % cols]))
    return np.array(state, dtype=np.int8)

# --- Game of Life Guided by Gröbner Basis (GPU Attempt) ---
class GameOfLifeGroebnerAdvancedGPU:
    def __init__(self, grid_size, p, local_size):
        self.grid_size = grid_size
        self.p = p
        self.local_size = local_size
        self.use_gpu = gpu_available
        if self.use_gpu:
            self.state_gpu = cp.asarray(np.random.randint(0, p, (grid_size, grid_size)).astype(np.int8))
        else:
            self.state_cpu = np.random.randint(0, p, (grid_size, grid_size)).astype(np.int8)
        self.time = 0
        self.local_symbols = get_local_symbols(local_size)
        self.basis = None
        self.target_pattern = "line" # Guided by poem

    def get_state(self):
        return self.state_gpu.get() if self.use_gpu else self.state_cpu

    def evaluate_basis_gpu(self, state_gpu, basis, local_symbols, p, size):
        rows, cols = state_gpu.shape
        offset = size // 2
        basis_violation_gpu = cp.zeros_like(state_gpu, dtype=cp.uint8)

        @cp.fuse()
        def eval_point(r, c, grid):
            local_state_np = get_local_state_np(grid.get(), r, c, size)
            symbol_values = dict(zip(local_symbols, local_state_np))
            basis_evaluations = [poly.eval(symbol_values) % p for poly in basis]
            return 1 if any(eval_val != 0 for eval_val in basis_evaluations) else 0

        basis_violation_gpu = cp.array([[eval_point(r, c, state_gpu) for c in range(cols)] for r in range(rows)], dtype=cp.uint8)
        return basis_violation_gpu

    def update(self):
        if self.use_gpu and self.basis:
            state_gpu = self.state_gpu
            basis_violation_gpu = self.evaluate_basis_gpu(state_gpu, self.basis, self.local_symbols, self.p, self.local_size)
            next_state_gpu = cp.where(basis_violation_gpu, 1 - state_gpu, state_gpu)
            random_mask = cp.random.rand(*state_gpu.shape) < 0.05
            random_values = cp.random.randint(0, 2, state_gpu.shape)
            next_state_gpu = cp.where(random_mask & (state_gpu == 0), random_values, next_state_gpu)
            self.state_gpu = next_state_gpu
        else:
            state = self.get_state()
            next_state = np.copy(state)
            rows, cols = state.shape
            offset = self.local_size // 2

            if self.basis:
                basis_violation = np.zeros_like(state, dtype=np.uint8)
                for r in range(rows):
                    for c in range(cols):
                        local_state_values = get_local_state(state, r, c, self.local_size)
                        symbol_values = dict(zip(self.local_symbols, local_state_values))
                        basis_evaluations = [poly.eval(symbol_values) % self.p for poly in self.basis]
                        if any(eval_val != 0 for eval_val in basis_evaluations):
                            basis_violation[r, c] = 1
                for r in range(rows):
                    for c in range(cols):
                        if basis_violation[r, c]:
                            next_state[r, c] = 1 - state[r, c]
                        elif random.random() < 0.05:
                            next_state[r, c] = random.randint(0, 1)
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
            offset = self.local_size // 2
            center_index = offset * self.local_size + offset
            line_indices = [center_index - 1, center_index, center_index + 1]
            for i in range(self.local_size * self.local_size):
                if i not in line_indices:
                    polynomials.append(x[i])
                else:
                    polynomials.append(x[i] - 1)
        elif "vagabonde" in current_line:
            self.target_pattern = "block"
            x = self.local_symbols
            offset = self.local_size // 2
            block_indices = [
                (offset - 1) * self.local_size + offset - 1, (offset - 1) * self.local_size + offset,
                offset * self.local_size + offset - 1,     offset * self.local_size + offset
            ]
            flat_indices = [i * self.local_size + j for i in range(self.local_size) for j in range(self.local_size)]
            for i in flat_indices:
                row = i // self.local_size - offset + 1
                col = i % self.local_size - offset + 1
                if row in [-1, 0] and col in [-1, 0]:
                    polynomials.append(x[i] - 1)
                else:
                    polynomials.append(x[i])

        if polynomials:
            self.basis = compute_groebner_basis(polynomials, self.local_size)
        else:
            self.basis = None

    def get_display_state(self):
        return self.state_gpu.get() if self.use_gpu else self.state_cpu

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
    pygame.display.set_caption(f"AI Spring: Stage {poem_stage}, Line: '{line}', Pattern: {game_of_life.target_pattern}, Basis Size: {len(game_of_life.basis) if game_of_life.basis else 0}, GPU: {game_of_life.use_gpu}")

    rows, cols = display_state.shape
    offset = game_of_life.local_size // 2
    for r in range(rows):
        for c in range(cols):
            x = c * CELL_SIZE
            y = r * CELL_SIZE
            state_value = display_state[r, c]
            color = ALIVE_COLOR_MAP.get((state_value,), BACKGROUND_COLOR)
            if game_of_life.basis and game_of_life.use_gpu:
                local_state_np = get_local_state_np(display_state, r, c, game_of_life.local_size)
                symbol_values = dict(zip(game_of_life.local_symbols, local_state_np))
                basis_evaluations = [poly.eval(symbol_values) % PRIME_FIELD for poly in game_of_life.basis]
                if any(eval_val != 0 for eval_val in basis_evaluations):
                    color = (255, 0, 0)
            elif game_of_life.basis and not game_of_life.use_gpu:
                local_state_values = get_local_state(display_state, r, c, game_of_life.local_size)
                symbol_values = dict(zip(game_of_life.local_symbols, local_state_values))
                basis_evaluations = [poly.eval(symbol_values) % PRIME_FIELD for poly in game_of_life.basis]
                if any(eval_val != 0 for eval_val in basis_evaluations):
                    color = (255, 0, 0)

            pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))

# --- Main Loop ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    game = GameOfLifeGroebnerAdvancedGPU(GRID_SIZE, PRIME_FIELD, LOCAL_SIZE)

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
                game = GameOfLifeGroebnerAdvancedGPU(GRID_SIZE, PRIME_FIELD, LOCAL_SIZE)

        game.update()
        draw_grid(screen, game)
        pygame.display.flip()
        clock.tick(30) # Slow down for computation

        if game.time % stage_duration == 0:
            poem_stage += 1

    pygame.quit()

if __name__ == '__main__':
    main()
