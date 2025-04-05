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
GRID_SIZE = 50
CELL_SIZE = 12
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE
BACKGROUND_COLOR = (0, 0, 0)
ALIVE_COLOR_MAP = {(0,): (0, 0, 0), (1,): (255, 255, 255)} # For GF(2)
FRACTAL_COLOR = (50, 50, 50)
MIDI_DEVICE_NAME = None
PRIME_FIELD = 2

# --- Algebraic Geometry: Conceptual Ideal via Pattern Tracking ---
def check_pattern(grid, r, c, pattern):
    rows, cols = grid.shape
    pr, pc = len(pattern), len(pattern[0])
    for i in range(pr):
        for j in range(pc):
            if pattern[i][j] is not None and int(grid[(r + i) % rows, (c + j) % cols]) != pattern[i][j]:
                return False
    return True

# --- Game of Life Learning to Favor Patterns (Ideal Analogy) ---
class GameOfLifeLearningIdeal:
    def __init__(self, grid_size, p):
        self.grid_size = grid_size
        self.p = p
        self.use_gpu = gpu_available
        if self.use_gpu:
            self.state_gpu = cp.asarray(np.random.randint(0, p, (grid_size, grid_size)).astype(np.int8))
        else:
            self.state_cpu = np.random.randint(0, p, (grid_size, grid_size)).astype(np.int8)
        self.time = 0
        self.favored_patterns = []
        self.pattern_influence = 0.1

    def get_state(self):
        return self.state_gpu.get() if self.use_gpu else self.state_cpu

    def update(self):
        if self.use_gpu:
            state = self.state_gpu.get()
        else:
            state = self.state_cpu
        next_state = np.zeros_like(state)
        rows, cols = state.shape

        for r in range(rows):
            for c in range(cols):
                live_neighbors = self.count_live_neighbors(state, r, c)
                current_cell = state[r, c]
                pattern_bonus = 0
                for pattern in self.favored_patterns:
                    if check_pattern(state, r - len(pattern) // 2, c - len(pattern[0]) // 2, pattern):
                        pattern_bonus += self.pattern_influence

                if current_cell == 1:
                    next_state[r, c] = 1 if 2 + pattern_bonus <= live_neighbors <= 3 + pattern_bonus else 0
                else:
                    next_state[r, c] = 1 if live_neighbors >= 3 - pattern_bonus else 0

        if self.use_gpu:
            self.state_gpu = cp.asarray(next_state.astype(np.int8))
        else:
            self.state_cpu = next_state.astype(np.int8)

        self.time += 1
        self.learn()

    def count_live_neighbors(self, grid, r, c):
        count = 0
        rows, cols = grid.shape
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                if grid[(r + i) % rows, (c + j) % cols] == 1:
                    count += 1
        return count

    def learn(self):
        global poem_stage
        current_line_index = poem_stage % len(poem)
        current_line = poem[current_line_index]
        self.favored_patterns = []

        if "désir léger" in current_line:
            self.favored_patterns.append([[1, 0], [0, 1]])
        elif "sans destin" in current_line:
            self.favored_patterns.append([[1, 1], [0, 0]])
        elif "dissolvent" in current_line:
            self.favored_patterns.append([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        elif "éclat d’un moment" in current_line:
            self.favored_patterns.append([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

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
    pygame.display.set_caption(f"AI Spring: Stage {poem_stage}, Line: '{line}', Patterns: {[len(p) for p in game_of_life.favored_patterns]} (GPU: {game_of_life.use_gpu})")

    for row in range(game_of_life.grid_size):
        for col in range(game_of_life.grid_size):
            x = col * CELL_SIZE
            y = row * CELL_SIZE
            state_value = display_state[row, col]
            color = ALIVE_COLOR_MAP.get((state_value,), BACKGROUND_COLOR)
            for pattern in game_of_life.favored_patterns:
                if check_pattern(display_state, row - len(pattern) // 2, col - len(pattern[0]) // 2, pattern):
                    color = (0, 255, 0) # Highlight favored patterns
                    break
            pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))

# --- Main Loop ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    game = GameOfLifeLearningIdeal(GRID_SIZE, PRIME_FIELD)

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
                game = GameOfLifeLearningIdeal(GRID_SIZE, PRIME_FIELD)

        game.update()
        draw_grid(screen, game)
        pygame.display.flip()
        clock.tick(60)

        if game.time % stage_duration == 0:
            poem_stage += 1

    pygame.quit()

if __name__ == '__main__':
    main()
