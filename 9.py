import pygame
import numpy as np
import mido
import random
from pygame.locals import *

# --- Configuration ---
GRID_SIZE = 100
CELL_SIZE = 8
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE
BACKGROUND_COLOR = (0, 0, 0)
ALIVE_COLOR_MAP = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 255, 0)} # Extend as needed
FRACTAL_COLOR = (50, 50, 50)
MIDI_DEVICE_NAME = None
PRIME_FIELD = 3  # Choose a prime number for Z_p

# --- The Field Z_p ---
class Zp:
    def __init__(self, value, p):
        self.p = p
        self.value = value % p

    def __add__(self, other):
        if isinstance(other, Zp) and other.p == self.p:
            return Zp((self.value + other.value) % self.p, self.p)
        elif isinstance(other, int):
            return Zp((self.value + other) % self.p, self.p)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Zp) and other.p == self.p:
            return Zp((self.value - other.value) % self.p, self.p)
        elif isinstance(other, int):
            return Zp((self.value - other) % self.p, self.p)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Zp) and other.p == self.p:
            return Zp((self.value * other.value) % self.p, self.p)
        elif isinstance(other, int):
            return Zp((self.value * other) % self.p, self.p)
        return NotImplemented

    def __int__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, Zp) and other.p == self.p:
            return self.value == other.value
        elif isinstance(other, int):
            return self.value == other
        return False

    def __repr__(self):
        return str(self.value)

# --- Algebraic Geometry: Polynomial defining the Fractal ---
def get_fractal_value(normalized_x, normalized_y, equation_type="circle", **kwargs):
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

# --- Game of Life in Z_p with Fractal Influenced Polynomial Rule ---
class GameOfLifeZp:
    def __init__(self, grid_size, p):
        self.grid_size = grid_size
        self.p = p
        self.state = np.array([[Zp(0, p) for _ in range(grid_size)] for _ in range(grid_size)])

    def get_neighbors_sum(self, row, col):
        neighbor_sum = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                r, c = (row + i + self.grid_size) % self.grid_size, (col + j + self.grid_size) % self.grid_size
                neighbor_sum += int(self.state[r][c])
        return neighbor_sum

    def update(self, fractal_equation="circle", fractal_params=None):
        new_state = np.copy(self.state)
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                normalized_x = (col - self.grid_size / 2) / (self.grid_size / 2)
                normalized_y = (row - self.grid_size / 2) / (self.grid_size / 2)
                fractal_value = get_fractal_value(normalized_x, normalized_y, fractal_equation, **(fractal_params or {}))

                neighbors = self.get_neighbors_sum(row, col)
                current_state = self.state[row][col]

                # Define the coefficients of the polynomial rule based on the fractal value
                # Example linear rule: next_state = (a * current_state + b * neighbor_sum + c) mod p
                # where a, b, c are functions of fractal_value

                # Simple linear functions for coefficients (can be made more complex)
                a_coeff = int(abs(fractal_value) * 2) % self.p
                b_coeff = (int(fractal_value * 3) + 1) % self.p
                c_coeff = int(abs(fractal_value) * self.p) % self.p

                next_state_val = (a_coeff * current_state.value + b_coeff * neighbors + c_coeff) % self.p
                new_state[row][col] = Zp(next_state_val, self.p)

        self.state = new_state

    def seed(self, positions, value=1):
        for r, c in positions:
            if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                self.state[r][c] = Zp(value % self.p, self.p)

    def get_display_state(self):
        return np.array([[cell.value for cell in row] for row in self.state])

# --- MIDI Input Handling --- (Same as before)
def get_midi_input_names():
    return mido.get_input_names()

def open_midi_input(device_name=None):
    try:
        if device_name:
            return mido.open_input(device_name)
        else:
            input_names = get_midi_input_names()
            if input_names:
                print("Available MIDI input devices:")
                for i, name in enumerate(input_names):
                    print(f"{i}: {name}")
                selected_index = input("Enter the index of the MIDI input device to use (or leave blank for none): ")
                if selected_index.isdigit() and 0 <= int(selected_index) < len(input_names):
                    return mido.open_input(input_names[int(selected_index)])
                else:
                    print("No MIDI input selected.")
                    return None
            else:
                print("No MIDI input devices found.")
                return None
    except Exception as e:
        print(f"Error opening MIDI input: {e}")
        return None

# --- Visualization ---
def draw_grid(screen, game_of_life):
    display_state = game_of_life.get_display_state()
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
    pygame.display.set_caption(f"Chaos, Fractals, and Life in Z/{PRIME_FIELD}Z")

    game = GameOfLifeZp(GRID_SIZE, PRIME_FIELD)
    fractal_equation_type = "circle"
    fractal_parameters = {"radius": 0.4}

    # Poem Inspired Initial State: "Le printemps réveille en moi un désir léger, sans racine ni chemin,"
    # -> Initial random seeding with varying states
    initial_seed_count = 2000  # Increase seeding
    for _ in range(initial_seed_count):
        r = random.randrange(GRID_SIZE)
        c = random.randrange(GRID_SIZE)
        game.seed([(r, c)], random.randint(0, PRIME_FIELD - 1))

    midi_input = open_midi_input(MIDI_DEVICE_NAME)
    midi_notes_buffer = []

    running = True
    clock = pygame.time.Clock()

    while running:
        screen.fill(BACKGROUND_COLOR)

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            if event.type == KEYDOWN:
                if event.key == K_SPACE:
                    game.update(fractal_equation_type, fractal_parameters)
                if event.key == K_r:
                    game = GameOfLifeZp(GRID_SIZE, PRIME_FIELD)
                    for _ in range(initial_seed_count):
                        r = random.randrange(GRID_SIZE)
                        c = random.randrange(GRID_SIZE)
                        game.seed([(r, c)], random.randint(0, PRIME_FIELD - 1))
                if event.key == K_f:
                    fractal_options = ["circle", "julia", "polynomial"]
                    current_index = fractal_options.index(fractal_equation_type) if fractal_equation_type in fractal_options else -1
                    next_index = (current_index + 1) % len(fractal_options)
                    fractal_equation_type = fractal_options[next_index]
                    if fractal_equation_type == "julia":
                        fractal_parameters = {"c": (-0.7, 0.27015), "power": random.randint(2, 4), "max_iter": 20}
                    elif fractal_equation_type == "polynomial":
                        fractal_parameters = {"a": random.uniform(0.1, 0.5)}
                    else:
                        fractal_parameters = {"radius": random.uniform(0.2, 0.6)}

        # MIDI Input Handling
        if midi_input:
            for msg in midi_input.poll():
                if msg.type == 'note_on' and msg.velocity > 0:
                    midi_notes_buffer.append(msg.note)
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in midi_notes_buffer:
                        midi_notes_buffer.remove(msg.note)

            seed_positions = []
            for note in midi_notes_buffer:
                row = note % GRID_SIZE
                col = (note * 7) % GRID_SIZE
                seed_positions.append((row, col))
            for r, c in seed_positions:
                # Seed with a value influenced by the MIDI note
                seed_value = (note % PRIME_FIELD)
                game.seed([(r, c)], seed_value)

        game.update(fractal_equation_type, fractal_parameters)
        draw_grid(screen, game)
        pygame.display.flip()
        clock.tick(5)

    if midi_input:
        midi_input.close()
    pygame.quit()

if __name__ == '__main__':
    main()
