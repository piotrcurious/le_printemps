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
ALIVE_COLOR_MAP = {(0, 0): (0, 0, 0), (1, 0): (255, 0, 0), (0, 1): (0, 255, 0), (1, 1): (255, 255, 0)} # For GF(2)
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

# --- Algebraic Geometry: Simple Fractal Pattern (Checkerboard for GF(2)) ---
def is_checkerboard(r, c):
    return (r + c) % 2

# --- Game of Life Evolving Towards a Fractal (Analogy) ---
class GameOfLifeFractalEvolution:
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
        self.fractal_attraction = 0.5 # Parameter to be learned

    def update(self):
        if self.use_gpu:
            neighbor_sum_gpu = cp.convolve2d(self.state_gpu, self.kernel_gpu, mode='same', boundary='wrap')
            next_state_gpu = cp.zeros_like(self.state_gpu)
            rand_gpu = cp.random.rand(self.grid_size, self.grid_size)
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    live_neighbors = neighbor_sum_gpu[r, c]
                    current_cell = self.state_gpu[r, c]
                    target_state = is_checkerboard(r, c)

                    # Modified rules to favor the checkerboard pattern
                    if current_cell == 1:
                        next_state_gpu[r, c] = 1 if 2 <= live_neighbors <= 3 else 0
                        if target_state == 0 and live_neighbors == 3 and rand_gpu[r, c] < self.fractal_attraction:
                            next_state_gpu[r, c] = 0
                    else:
                        next_state_gpu[r, c] = 1 if live_neighbors == 3 and rand_gpu[r, c] > (1 - self.fractal_attraction) else 0
                        if target_state == 1 and live_neighbors == 3 and rand_gpu[r, c] < self.fractal_attraction:
                            next_state_gpu[r, c] = 1

            self.state_gpu = next_state_gpu
        else:
            self._update_cpu()
        self.time += 1
        self.learn()

    @numba.jit(nopython=True)
    def _update_cpu(self):
        neighbor_sum = convolve2d(self.state_cpu, self.kernel_cpu, mode='same', boundary='wrap')
        next_state = np.zeros_like(self.state_cpu)
        rand_cpu = np.random.rand(self.grid_size, self.grid_size)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                live_neighbors = neighbor_sum[r, c]
                current_cell = self.state_cpu[r, c]
                target_state = is_checkerboard(r, c)

                if current_cell == 1:
                    next_state[r, c] = 1 if 2 <= live_neighbors <= 3 else 0
                    if target_state == 0 and live_neighbors == 3 and rand_cpu[r, c] < self.fractal_attraction:
                        next_state[r, c] = 0
                else:
                    next_state[r, c] = 1 if live_neighbors == 3 and rand_cpu[r, c] > (1 - self.fractal_attraction) else 0
                    if target_state == 1 and live_neighbors == 3 and rand_cpu[r, c] < self.fractal_attraction:
                        next_state[r, c] = 1
        self.state_cpu = next_state

    def learn(self):
        global poem_stage
        current_line_index = poem_stage % len(poem)
        current_line = poem[current_line_index]

        similarity = 0
        state = self.get_display_state()
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if int(state[r, c]) == is_checkerboard(r, c):
                    similarity += 1
        similarity /= (self.grid_size * self.grid_size)

        if "printemps" in current_line:
            if self.fractal_attraction < 0.8:
                self.fractal_attraction += self.learning_rate
        elif "dissolvent" in current_line:
            if self.fractal_attraction > 0.2:
                self.fractal_attraction -= self.learning_rate
        elif "incertain" in current_line:
            self.fractal_attraction = 0.5

        self.fractal_attraction = np.clip(self.fractal_attraction, 0, 1)

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
    pygame.display.set_caption(f"AI Spring: Stage {poem_stage}, Line: '{line}', Attraction: {game_of_life.fractal_attraction:.2f} (GPU: {game_of_life.use_gpu})")

    for row in range(game_of_life.grid_size):
        for col in range(game_of_life.grid_size):
            x = col * CELL_SIZE
            y = row * CELL_SIZE
            state_value = display_state[row, col]
            target_state = is_checkerboard(row, col)
            color_key = (state_value, target_state)
            color = ALIVE_COLOR_MAP.get(color_key, BACKGROUND_COLOR)
            pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))

# --- Main Loop ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    game = GameOfLifeFractalEvolution(GRID_SIZE, PRIME_FIELD)

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
                game = GameOfLifeFractalEvolution(GRID_SIZE, PRIME_FIELD)

        game.update()
        draw_grid(screen, game)
        pygame.display.flip()
        clock.tick(60)

        if game.time % stage_duration == 0:
            poem_stage += 1

    pygame.quit()

if __name__ == '__main__':
    main()
