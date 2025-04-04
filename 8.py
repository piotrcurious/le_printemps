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
ALIVE_COLOR_MAP = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 255, 0)} # Extend for larger p
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
        for _ in range(kwargs.get("max_iter", 10)): # Less iterations for a value
            z = z**power + c
        return abs(z) - 2 # Distance from the boundary
    elif equation_type == "polynomial":
        a = kwargs.get("a", 0.3)
        return normalized_x**3 + normalized_y**3 - a * normalized_x * normalized_y
    return 0

# --- Game of Life in Z_p with Fractal Influence ---
class GameOfLifeZp:
    def __init__(self, grid_size, p):
        self.grid_size = grid_size
        self.p = p
        self.state = np.array([[Zp(0, p) for _ in range(grid_size)] for _ in range(grid_size)])

    def get_neighbors(self, row, col):
        neighbors = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                r, c = (row + i + self.grid_size) % self.grid_size, (col + j + self.grid_size) % self.grid_size
                neighbors.append(self.state[r][c])
        return neighbors

    def update(self, fractal_equation="circle", fractal_params=None):
        new_state = np.copy(self.state)
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                normalized_x = (col - self.grid_size / 2) / (self.grid_size / 2)
                normalized_y = (row - self.grid_size / 2) / (self.grid_size / 2)
                fractal_value = get_fractal_value(normalized_x, normalized_y, fractal_equation, **(fractal_params or {}))

                neighbors = self.get_neighbors(row, col)
                neighbor_sum = sum(n.value for n in neighbors)

                current_state = self.state[row][col]

                # Define the update rule as a polynomial in Z_p, influenced by fractal_value
                # Example: next_state = (a * current_state + b * (neighbor_sum % self.p) + c * f(x, y)) mod p
                # where a, b, c are coefficients in Z_p that might depend on the fractal_value

                # A more direct approach: modify the standard GOL rules based on fractal value
                live_neighbors = sum(1 for n in neighbors if n.value > 0)

                # Coefficients influenced by fractal value (example)
                influence_factor = abs(fractal_value) * 2 # Scale the influence

                if current_state.value > 0:
                    survival_threshold_low = 2 + int(influence_factor) % 2
                    survival_threshold_high = 3 + int(influence_factor) % 2
                    if live_neighbors < survival_threshold_low or live_neighbors > survival_threshold_high:
                        new_state[row][col] = Zp(0, self.p)
                    elif live_neighbors >= survival_threshold_low and live_neighbors <= survival_threshold_high:
                        new_state[row][col] = current_state
                else:
                    birth_threshold = 3 + int(influence_factor) % 2
                    if live_neighbors == birth_threshold:
                        new_state[row][col] = Zp(1, self.p) # Birth into state 1

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
    # ... (same as before)
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

    # Initial random seeding
    initial_seed_count = 100
    for _ in range(initial_seed_count):
        r = random.randrange(GRID_SIZE)
        c = random.randrange(GRID_SIZE)
        game.seed([(r, c)], random.randint(1, PRIME_FIELD - 1))

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
                        game.seed([(r, c)], random.randint(1, PRIME_FIELD - 1))
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
                col = (note * 7) % GRID_SIZE # Spread out the seeding
                seed_positions.append((row, col))
            for r, c in seed_positions:
                game.seed([(r, c)], (note % (PRIME_FIELD - 1)) + 1) # Seed with a value > 0

        game.update(fractal_equation_type, fractal_parameters)
        draw_grid(screen, game)
        pygame.display.flip()
        clock.tick(5) # Adjust speed

    if midi_input:
        midi_input.close()
    pygame.quit()

if __name__ == '__main__':
    main()
