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

# --- Algebraic Geometry: Simple Fractal Ideal (Cross Pattern Analogy for GF(2)) ---
def cross_pattern_violation(grid, r, c, p):
    center = int(grid[r % grid.shape[0], c % grid.shape[1]])
    up = int(grid[(r - 1) % grid.shape[0], c % grid.shape[1]])
    down = int(grid[(r + 1) % grid.shape[0], c % grid.shape[1]])
    left = int(grid[r % grid.shape[0], (c - 1) % grid.shape[1]])
    right = int(grid[r % grid.shape[0], (c + 1) % grid.shape[1]])
    # Polynomials that should be zero for the cross pattern (if center is 1, neighbors should be 0)
    violation = 0
    if center == 1:
        violation += up
        violation += down
        violation += left
        violation += right
    return violation % p

# --- Game of Life Evolving Towards a Fractal (More Advanced Analogy) ---
class GameOfLifeAlgebraicEvolution:
    def __init__(self, grid_size, p):
        self.grid_size = grid_size
        self.p = p
        self.use_gpu = gpu_available
        if self.use_gpu:
            self.state_gpu = cp.asarray(np.random.randint(0, p, (grid_size, grid_size)).astype(np.int8))
        else:
            self.state_cpu = np.random.randint(0, p, (grid_size, grid_size)).astype(np.int8)
        self.time = 0
        self.learning_rate = 0.1
        self.neighbor_influence = 0.5 # Parameter to be learned

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
                    state[r, (c + 1) % self.grid_size] +
                    state[(r - 1) % self.grid_size, (c - 1) % self.grid_size] +
                    state[(r - 1) % self.grid_size, (c + 1) % self.grid_size] +
                    state[(r + 1) % self.grid_size, (c - 1) % self.grid_size] +
                    state[(r + 1) % self.grid_size, (c + 1) % self.grid_size]
                ) % self.p

                # Simple polynomial update rule with a learnable parameter
                next_val = (state[r, c] * (1 - self.neighbor_influence) + neighbors_sum * self.neighbor_influence) % self.p
                next_state[r, c] = int(round(next_val)) # Assuming values are 0 or 1

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

        state = self.get_state()
        violation_sum = 0
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                violation_sum += cross_pattern_violation(state, r, c, self.p)

        avg_violation = violation_sum / (self.grid_size * self.grid_size)

        if "réveille" in current_line:
            if self.neighbor_influence < 0.7:
                self.neighbor_influence += self.learning_rate
        elif "dissolvent" in current_line:
            if self.neighbor_influence > 0.3:
                self.neighbor_influence -= self.learning_rate
        elif "incertain" in current_line:
            self.neighbor_influence = 0.5

        self.neighbor_influence = np.clip(self.neighbor_influence, 0, 1)

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
    pygame.display.set_caption(f"AI Spring: Stage {poem_stage}, Line: '{line}', Influence: {game_of_life.neighbor_influence:.2f} (GPU: {game_of_life.use_gpu})")

    for row in range(game_of_life.grid_size):
        for col in range(game_of_life.grid_size):
            x = col * CELL_SIZE
            y = row * CELL_SIZE
            state_value = display_state[row, col]
            violation = cross_pattern_violation(display_state, row, col, PRIME_FIELD)
            color = ALIVE_COLOR_MAP.get((state_value,), BACKGROUND_COLOR)
            if violation > 0:
                color = (255, 0, 0) # Indicate violation of the cross pattern
            pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))

# --- Main Loop ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    game = GameOfLifeAlgebraicEvolution(GRID_SIZE, PRIME_FIELD)

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
                game = GameOfLifeAlgebraicEvolution(GRID_SIZE, PRIME_FIELD)

        game.update()
        draw_grid(screen, game)
        pygame.display.flip()
        clock.tick(60)

        if game.time % stage_duration == 0:
            poem_stage += 1

    pygame.quit()

if __name__ == '__main__':
    main()
