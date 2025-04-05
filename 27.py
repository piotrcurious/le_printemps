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
LOCAL_SIZE = 3 # Size of the local neighborhood for Gröbner basis
CHUNK_SIZE = 10 # Size of grid chunks for parallel processing

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

def evaluate_basis_local_chunk(chunk_info):
    grid_chunk, r_start, c_start, basis, local_symbols, p, size = chunk_info
    rows, cols = grid_chunk.shape
    basis_violation_chunk = np.zeros_like(grid_chunk, dtype=np.uint8)
    for r_local in range(rows):
        for c_local in range(cols):
            r_global = r_start + r_local
            c_global = c_start + c_local
            local_state_np = get_local_state_np(grid_chunk, r_local, c_local, size)
            symbol_values = dict(zip(local_symbols, local_state_np))
            basis_evaluations = [poly.eval(symbol_values) % p for poly in basis]
            if any(eval_val != 0 for eval_val in basis_evaluations):
                basis_violation_chunk[r_local, c_local] = 1
    return basis_violation_chunk, r_start, c_start

# --- Game of Life Guided by Gröbner Basis (Parallel Processing) ---
class GameOfLifeGroebnerParallel:
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
        self.pool = mp.Pool(processes=mp.cpu_count())

    def get_state(self):
        return self.state_gpu.get() if self.use_gpu else self.state_cpu

    def update(self):
        state = self.get_state()
        next_state = np.copy(state)
        rows, cols = state.shape

        if self.basis and self.pool:
            chunks = []
            for r in range(0, rows, CHUNK_SIZE):
                for c in range(0, cols, CHUNK_SIZE):
                    chunk = state[r:min(r + CHUNK_SIZE, rows), c:min(c + CHUNK_SIZE, cols)]
                    chunks.append((chunk, r, c, self.basis, self.local_symbols, self.p, self.local_size))

            results = self.pool.map(evaluate_basis_local_chunk, chunks)
            basis_violation = np.zeros_like(state, dtype=np.uint8)
            for violation_chunk, r_start, c_start in results:
                basis_violation[r_start:r_start + violation_chunk.shape[0], c_start:c_start + violation_chunk.shape[1]] = violation_chunk

            for r in range(rows):
                for c in range(cols):
                    if basis_violation[r, c]:
                        next_state[r, c] = 1 - state[r, c]
                    elif random.random() < 0.05:
                        next_state[r, c] = random.randint(0, 1)
        else: # Basic Game of Life if no basis or pool
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

    def close_pool(self):
        if self.pool:
            self.pool.close()
            self.pool.join()

    def __del__(self):
        self.close_pool()

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
    pygame.display.set_caption(f"AI Spring: Stage {poem_stage}, Line: '{line}', Pattern: {game_of_life.target_pattern}, Basis Size: {len(game_of_life.basis) if game_of_life.basis else 0}, Parallel Processes: {game_of_life.pool._processes if hasattr(game_of_life.pool, '_processes') else 0}, GPU: {game_of_life.use_gpu}")

    rows, cols = display_state.shape
    offset = game_of_life.local_size // 2
    for r in range(rows):
        for c in range(cols):
            x = c * CELL_SIZE
            y = r * CELL_SIZE
            state_value = display_state[r, c]
            color = ALIVE_COLOR_MAP.get((state_value,), BACKGROUND_COLOR)
            if game_of_life.basis:
                local_state_values = get_local_state_np(display_state, r, c, game_of_life.local_size)
                symbol_values = dict(zip(game_of_life.local_symbols, local_state_values))
                basis_evaluations = [poly.eval(symbol_values) % PRIME_FIELD for poly in game_of_life.basis]
                if any(eval_val != 0 for eval_val in basis_evaluations):
                    color = (255, 0, 0)

            pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))

# --- Main Loop ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    game = GameOfLifeGroebnerParallel(GRID_SIZE, PRIME_FIELD, LOCAL_SIZE)

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
                game = GameOfLifeGroebnerParallel(GRID_SIZE, PRIME_FIELD, LOCAL_SIZE)

        game.update()
        draw_grid(screen, game)
        pygame.display.flip()
        clock.tick(30) # Slow down for computation

        if game.time % stage_duration == 0:
            poem_stage += 1

    pygame.quit()

if __name__ == '__main__':
    main()
