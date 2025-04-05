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
GRID_SIZE = 60 # Adjusted for potential complexity
CELL_SIZE = 10
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE
BACKGROUND_COLOR = (0, 0, 0)
ALIVE_COLOR_MAP = {i: tuple(np.random.randint(0, 256, 3)) for i in range(16)}
FRACTAL_COLOR = (50, 50, 50)
MIDI_DEVICE_NAME = None
PRIME_FIELD = 2

# --- The Field Z_p --- (Same as before)
class Zp:
    def __init__(self, value, p):
        self.p = p
        self.value = value % p

    def __add__(self, other):
        return Zp((self.value + other.value) % self.p, self.p)

    def __mul__(self, other):
        return Zp((self.value * other.value) % self.p, self.p)

    def __int__(self):
        return self.value

    def __repr__(self):
        return str(self.value)

# --- Algebraic Geometry: Simple Polynomial Constraint ---
def satisfies_polynomial(local_grid, p):
    # Example: Check if sum of a 2x2 block is 0 mod p
    if local_grid.shape == (2, 2):
        s = sum(int(x) for row in local_grid for x in row) % p
        return s == 0
    return False

# --- Game of Life with Enhanced Math and Poem Guidance ---
class GameOfLifeAdvancedMathPlusPlus:
    def __init__(self, grid_size, p):
        self.grid_size = grid_size
        self.p = p
        self.use_gpu = gpu_available
        if self.use_gpu:
            self.state_gpu = cp.asarray(np.random.randint(0, p, (grid_size, grid_size)).astype(np.int8))
            self.kernel_gpu = cp.asarray(np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]), dtype=np.int8)
        else:
            self.state_cpu = np.random.randint(0, p, (grid_size, grid_size)).astype(np.int8)
            self.kernel_cpu = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]), dtype=np.int8)
        self.time = 0
        self.learning_rate = 0.1
        self.spawn_probability = 0.5 # Parameter to be learned
        self.pattern_counts = {} # Track simple patterns

    def get_local_grid(self, r, c, size):
        if self.use_gpu:
            state = self.state_gpu.get()
        else:
            state = self.state_cpu
        r_start = max(0, r - size // 2)
        r_end = min(self.grid_size, r + size // 2 + 1)
        c_start = max(0, c - size // 2)
        c_end = min(self.grid_size, c + size // 2 + 1)
        return state[r_start:r_end, c_start:c_end]

    def update(self):
        if self.use_gpu:
            neighbor_sum_gpu = cp.convolve2d(self.state_gpu, self.kernel_gpu, mode='same', boundary='wrap')
            next_state_gpu = cp.zeros_like(self.state_gpu)
            rand_gpu = cp.random.rand(self.grid_size, self.grid_size)
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    live_neighbors = neighbor_sum_gpu[r, c]
                    current_cell = self.state_gpu[r, c]
                    if current_cell == 1:
                        next_state_gpu[r, c] = 1 if 2 <= live_neighbors <= 3 else 0
                    else:
                        next_state_gpu[r, c] = 1 if live_neighbors == 3 and rand_gpu[r, c] > (1 - self.spawn_probability) else 0
            self.state_gpu = next_state_gpu
        else:
            self._update_cpu()
        self.time += 1
        self.learn()
        self.track_patterns()

    @numba.jit(nopython=True)
    def _update_cpu(self):
        neighbor_sum = convolve2d(self.state_cpu, self.kernel_cpu, mode='same', boundary='wrap')
        next_state = np.zeros_like(self.state_cpu)
        rand_cpu = np.random.rand(self.grid_size, self.grid_size)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                live_neighbors = neighbor_sum[r, c]
                current_cell = self.state_cpu[r, c]
                if current_cell == 1:
                    next_state[r, c] = 1 if 2 <= live_neighbors <= 3 else 0
                else:
                    next_state[r, c] = 1 if live_neighbors == 3 and rand_cpu[r, c] > (1 - self.spawn_probability) else 0
        self.state_cpu = next_state

    def track_patterns(self):
        pattern = tuple(self.state_cpu[0:2, 0:2].flatten().tolist())
        self.pattern_counts[pattern] = self.pattern_counts.get(pattern, 0) + 1

    def learn(self):
        global poem_stage
        current_line_index = poem_stage % len(poem)
        current_line = poem[current_line_index]

        avg_state = np.mean(self.get_display_state())

        if "désir léger" in current_line:
            if self.spawn_probability < 0.7:
                self.spawn_probability += self.learning_rate
        elif "sans destin" in current_line:
            if self.spawn_probability > 0.3:
                self.spawn_probability -= self.learning_rate
        elif "se dissolvent" in current_line:
            if avg_state > 0.5:
                self.spawn_probability -= self.learning_rate * 2
        elif "éclat d’un moment fuyant" in current_line:
            self.spawn_probability = 0.5 # Reset to a neutral value
        elif "conte désenchanté" in current_line:
            if avg_state > 0.7:
                self.spawn_probability -= self.learning_rate * 3
        elif "tristesse de plus" in current_line:
            if avg_state < 0.2:
                self.spawn_probability += self.learning_rate * 2

        self.spawn_probability = np.clip(self.spawn_probability, 0, 1)

    def get_display_state(self):
        if self.use_gpu:
            return self.state_gpu.get()
        else:
            return self.state_cpu

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
    pygame.display.set_caption(f"AI Spring: Stage {poem_stage}, Line: '{line}', Spawn Prob: {game_of_life.spawn_probability:.2f} (GPU: {game_of_life.use_gpu})")

    for row in range(game_of_life.grid_size):
        for col in range(game_of_life.grid_size):
            x = col * CELL_SIZE
            y = row * CELL_SIZE
            state_value = display_state[row, col]
            local_grid = game_of_life.get_local_grid(row, col, 2)
            color = ALIVE_COLOR_MAP.get(state_value, BACKGROUND_COLOR)
            if local_grid.shape == (2, 2) and satisfies_polynomial(local_grid, PRIME_FIELD):
                color = (255, 255, 255) # Highlight if polynomial constraint is met
            pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))

# --- Main Loop ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    game = GameOfLifeAdvancedMathPlusPlus(GRID_SIZE, PRIME_FIELD)

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
                game = GameOfLifeAdvancedMathPlusPlus(GRID_SIZE, PRIME_FIELD)

        game.update()
        draw_grid(screen, game)
        pygame.display.flip()
        clock.tick(60)

        if game.time % stage_duration == 0:
            poem_stage += 1

    pygame.quit()

if __name__ == '__main__':
    main()
