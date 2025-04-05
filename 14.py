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

# --- Configuration --- (Adjusted for simplicity)
GRID_SIZE = 100
CELL_SIZE = 8
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE
BACKGROUND_COLOR = (0, 0, 0)
ALIVE_COLOR_MAP = {i: tuple(np.random.randint(0, 256, 3)) for i in range(16)}
FRACTAL_COLOR = (50, 50, 50)
MIDI_DEVICE_NAME = None
PRIME_FIELD = 2  # Start with GF(2)

# --- The Field Z_p (Simplified for GF(2)) ---
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

# --- Algebraic Geometry: Fractal Parameter ---
def get_fractal_parameter(normalized_x, normalized_y, equation_type="circle", **kwargs):
    if equation_type == "circle":
        radius = kwargs.get("radius", 0.5)
        return normalized_x**2 + normalized_y**2 - radius**2
    elif equation_type == "julia":
        c = complex(*kwargs.get("c", (-0.7, 0.27015)))
        power = kwargs.get("power", 2)
        z = complex(normalized_x, normalized_y)
        for _ in range(kwargs.get("max_iter", 10)):
            z = z**power + c
        return abs(z) - 2
    elif equation_type == "polynomial":
        a = kwargs.get("a", 0.3)
        return normalized_x**3 + normalized_y**3 - a * normalized_x * normalized_y
    return 0

# --- Game of Life with Matrix Operations and Basic Learning ---
class GameOfLifeAdvancedMathPlus:
    def __init__(self, grid_size, p):
        self.grid_size = grid_size
        self.p = p
        self.use_gpu = gpu_available
        if self.use_gpu:
            self.state_gpu = cp.asarray(np.random.randint(0, p, (grid_size, grid_size)).astype(np.int8))
            self.kernel_gpu = cp.asarray(np.array([[1, 1, 1],
                                                   [1, 0, 1],
                                                   [1, 1, 1]]), dtype=np.int8)
        else:
            self.state_cpu = np.random.randint(0, p, (grid_size, grid_size)).astype(np.int8)
            self.kernel_cpu = np.array([[1, 1, 1],
                                       [1, 0, 1],
                                       [1, 1, 1]], dtype=np.int8)
        self.time = 0
        self.learning_rate = 0.05
        self.fractal_threshold = 0.5

    def update(self):
        if self.use_gpu:
            neighbor_sum_gpu = cp.convolve2d(self.state_gpu, self.kernel_gpu, mode='same', boundary='wrap')
            next_state_gpu = cp.zeros_like(self.state_gpu)
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    live_neighbors = neighbor_sum_gpu[r, c]
                    current_cell = self.state_gpu[r, c]
                    if current_cell == 1:
                        next_state_gpu[r, c] = 1 if 2 <= live_neighbors <= 3 else 0
                    else:
                        next_state_gpu[r, c] = 1 if live_neighbors == 3 and cp.random.rand() > (1 - self.fractal_threshold) else 0
            self.state_gpu = next_state_gpu
        else:
            self._update_cpu()
        self.time += 1
        self.learn()

    @numba.jit(nopython=True)
    def _update_cpu(self):
        neighbor_sum = convolve2d(self.state_cpu, self.kernel_cpu, mode='same', boundary='wrap')
        next_state = np.zeros_like(self.state_cpu)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                live_neighbors = neighbor_sum[r, c]
                current_cell = self.state_cpu[r, c]
                if current_cell == 1:
                    next_state[r, c] = 1 if 2 <= live_neighbors <= 3 else 0
                else:
                    next_state[r, c] = 1 if live_neighbors == 3 and np.random.rand() > (1 - self.fractal_threshold) else 0
        self.state_cpu = next_state

    def learn(self):
        if self.use_gpu:
            state_vector = self.state_gpu.get().flatten()
        else:
            state_vector = self.state_cpu.flatten()
        if np.mean(state_vector) < 0.4 and self.fractal_threshold > 0:
            self.fractal_threshold -= self.learning_rate
        elif np.mean(state_vector) > 0.6 and self.fractal_threshold < 1:
            self.fractal_threshold += self.learning_rate

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
stage_duration = 50

# --- Visualization ---
def draw_grid(screen, game_of_life):
    display_state = game_of_life.get_display_state()
    global poem_stage
    line = poem[poem_stage % len(poem)]
    pygame.display.set_caption(f"AI Spring: Stage {poem_stage}, Line: '{line}', Threshold: {game_of_life.fractal_threshold:.2f} (GPU: {game_of_life.use_gpu})")

    for row in range(game_of_life.grid_size):
        for col in range(game_of_life.grid_size):
            x = col * CELL_SIZE
            y = row * CELL_SIZE
            state_value = display_state[row, col]
            color = ALIVE_COLOR_MAP.get(state_value, BACKGROUND_COLOR)
            pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))

# --- Main Loop ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    game = GameOfLifeAdvancedMathPlus(GRID_SIZE, PRIME_FIELD)

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
                game = GameOfLifeAdvancedMathPlus(GRID_SIZE, PRIME_FIELD)

        game.update()
        draw_grid(screen, game)
        pygame.display.flip()
        clock.tick(60) # Increased frame rate for smoother experience

        if game.time % stage_duration == 0:
            poem_stage += 1

    pygame.quit()

if __name__ == '__main__':
    main()
