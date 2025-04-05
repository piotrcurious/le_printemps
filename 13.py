import pygame
import numpy as np
import mido
import random
from pygame.locals import *
from scipy.signal import convolve2d

# --- Configuration --- (Adjusted for simplicity)
GRID_SIZE = 30
CELL_SIZE = 15
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE
BACKGROUND_COLOR = (0, 0, 0)
ALIVE_COLOR_MAP = {i: tuple(np.random.randint(0, 256, 3)) for i in range(16)}
FRACTAL_COLOR = (50, 50, 50)
MIDI_DEVICE_NAME = None
PRIME_FIELD = 2  # Start with GF(2) for simplicity of polynomials

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

# --- Algebraic Geometry: Fractal Constraint (Simplified) ---
def fractal_constraint(state_vector, threshold=0.3):
    # Example: Check if the average state is above a threshold
    return np.mean(state_vector) > threshold

# --- Game of Life with Matrix Operations and Basic Learning ---
class GameOfLifeAdvancedMathPlus:
    def __init__(self, grid_size, p):
        self.grid_size = grid_size
        self.p = p
        self.state = np.random.randint(0, p, (grid_size, grid_size)).astype(Zp)
        self.time = 0
        self.learning_rate = 0.05
        self.fractal_threshold = 0.5 # Parameter to be learned
        self.kernel = np.array([[1, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1]])

    def update(self):
        current_state_int = np.vectorize(int)(self.state)
        neighbor_sum = convolve2d(current_state_int, self.kernel, mode='same', boundary='wrap')

        next_state = np.zeros_like(self.state)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                live_neighbors = neighbor_sum[r, c]
                current_cell = int(self.state[r, c])

                # Polynomial-like update rule influenced by a global fractal parameter
                if current_cell == 1:
                    if live_neighbors < 2 or live_neighbors > 3:
                        next_state[r, c] = Zp(0, self.p)
                    else:
                        next_state[r, c] = Zp(1, self.p)
                else:
                    if live_neighbors == 3 and np.random.rand() > (1 - self.fractal_threshold): # Fractal influence
                        next_state[r, c] = Zp(1, self.p)
                    else:
                        next_state[r, c] = Zp(0, self.p)

        self.state = next_state
        self.time += 1
        self.learn()

    def learn(self):
        state_vector = np.vectorize(int)(self.state).flatten()
        if np.mean(state_vector) < 0.4 and self.fractal_threshold > 0:
            self.fractal_threshold -= self.learning_rate
        elif np.mean(state_vector) > 0.6 and self.fractal_threshold < 1:
            self.fractal_threshold += self.learning_rate

    def get_display_state(self):
        return np.vectorize(int)(self.state)

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
    pygame.display.set_caption(f"AI Spring: Stage {poem_stage}, Line: '{line}', Threshold: {game_of_life.fractal_threshold:.2f}")

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
    fractal_equation_type = "julia"
    fractal_parameters = {"c": (-0.7, 0.27015), "power": 2, "max_iter": 20}

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
        clock.tick(10)

        if game.time % stage_duration == 0:
            poem_stage += 1

    pygame.quit()

if __name__ == '__main__':
    main()
