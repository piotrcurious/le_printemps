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

# --- Algebraic Geometry: Simple Fractal Variety (Cross for Analogy) ---
def on_fractal(r, c, center_r, center_c):
    return r == center_r or c == center_c

# --- Game of Life Evolving on a Conceptual Algebraic Variety ---
class GameOfLifeAlgebraicVariety:
    def __init__(self, grid_size, p):
        self.grid_size = grid_size
        self.p = p
        self.use_gpu = gpu_available
        if self.use_gpu:
            self.state_gpu = cp.asarray(np.random.randint(0, p, (grid_size, grid_size)).astype(np.int8))
        else:
            self.state_cpu = np.random.randint(0, p, (grid_size, grid_size)).astype(np.int8)
        self.time = 0
        self.learning_rate = 1
        self.center_r = grid_size // 2
        self.center_c = grid_size // 2

    def get_state(self):
        return self.state_gpu.get() if self.use_gpu else self.state_cpu

    def update(self):
        state = self.get_state()
        next_state = np.zeros_like(state)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                neighbors_sum = (
                    state[(r - 1) % self.grid_size, c] +
                    state[(r + 1) % self.grid_size, c] +
                    state[r, (c - 1) % self.grid_size] +
                    state[r, (c + 1) % self.grid_size]
                ) % self.p

                on_f = on_fractal(r, c, self.center_r, self.center_c)

                if state[r, c] == 1:
                    next_state[r, c] = 1 if on_f or neighbors_sum >= 1 else 0
                else:
                    next_state[r, c] = 1 if not on_f and neighbors_sum >= 2 else 0

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
            self.center_r = max(0, self.center_r - 1)
        elif "bohémienne" in current_line:
            self.center_c = (self.center_c + 1) % self.grid_size
        elif "dissolvent" in current_line:
            self.center_r = min(self.grid_size - 1, self.center_r + 1)
        elif "temps" in current_line:
            self.center_c = max(0, self.center_c - 1)

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
    pygame.display.set_caption(f"AI Spring: Stage {poem_stage}, Line: '{line}', Center: ({game_of_life.center_r}, {game_of_life.center_c}) (GPU: {game_of_life.use_gpu})")

    for row in range(game_of_life.grid_size):
        for col in range(game_of_life.grid_size):
            x = col * CELL_SIZE
            y = row * CELL_SIZE
            state_value = display_state[row, col]
            on_f = on_fractal(row, col, game_of_life.center_r, game_of_life.center_c)
            color = ALIVE_COLOR_MAP.get((state_value,), BACKGROUND_COLOR)
            if on_f:
                color = (0, 255, 0) # Highlight cells on the fractal
            pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))

# --- Main Loop ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    game = GameOfLifeAlgebraicVariety(GRID_SIZE, PRIME_FIELD)

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
                game = GameOfLifeAlgebraicVariety(GRID_SIZE, PRIME_FIELD)

        game.update()
        draw_grid(screen, game)
        pygame.display.flip()
        clock.tick(60)

        if game.time % stage_duration == 0:
            poem_stage += 1

    pygame.quit()

if __name__ == '__main__':
    main()
