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
ALIVE_COLOR = (255, 255, 255)
FRACTAL_COLOR = (50, 50, 50)
MIDI_DEVICE_NAME = None  # Set your MIDI device name here if needed

# --- The Ring: Z/2Z ---
class Z2:
    def __init__(self, value):
        self.value = value % 2

    def __add__(self, other):
        return Z2((self.value + other.value) % 2)

    def __mul__(self, other):
        return Z2((self.value * other.value) % 2)

    def __int__(self):
        return self.value

    def __eq__(self, other):
        return self.value == other.value

    def __repr__(self):
        return str(self.value)

# --- Algebraic Geometry: Polynomials defining Fractals ---
def create_fractal_mask(grid_size, equation_type="circle", **kwargs):
    """
    Generates a boolean mask for the fractal region based on a polynomial equation.
    """
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    x_coords, y_coords = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    normalized_x = (x_coords - grid_size / 2) / (grid_size / 2)
    normalized_y = (y_coords - grid_size / 2) / (grid_size / 2)

    if equation_type == "circle":
        radius = kwargs.get("radius", 0.5)
        mask = normalized_x**2 + normalized_y**2 <= radius**2
    elif equation_type == "julia":
        c = complex(*kwargs.get("c", (-0.7, 0.27015)))
        power = kwargs.get("power", 2)
        max_iter = kwargs.get("max_iter", 50)
        for i in range(grid_size):
            for j in range(grid_size):
                z = complex(normalized_x[i, j], normalized_y[i, j])
                for _ in range(max_iter):
                    z = z**power + c
                    if abs(z) > 2:
                        mask[i, j] = False
                        break
                else:
                    mask[i, j] = True
    elif equation_type == "polynomial":
        # Example: Folium of Descartes (x^3 + y^3 - a*x*y = 0)
        a = kwargs.get("a", 0.3)
        condition = normalized_x**3 + normalized_y**3 - a * normalized_x * normalized_y
        mask = np.abs(condition) < 0.01  # Adjust threshold
    return mask

# --- Game of Life on the Ring Z/2Z constrained by the Fractal ---
class GameOfLife:
    def __init__(self, grid_size, constraint_mask=None):
        self.grid_size = grid_size
        self.state = np.array([[Z2(0) for _ in range(grid_size)] for _ in range(grid_size)])
        self.constraint_mask = constraint_mask

    def get_neighbors_sum(self, row, col):
        neighbor_sum = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                r, c = (row + i + self.grid_size) % self.grid_size, (col + j + self.grid_size) % self.grid_size
                neighbor_sum += int(self.state[r][c])
        return neighbor_sum

    def update(self):
        new_state = np.copy(self.state)
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.constraint_mask is not None and not self.constraint_mask[row, col]:
                    new_state[row][col] = Z2(0)
                    continue

                current_state = int(self.state[row][col])
                neighbors_sum = self.get_neighbors_sum(row, col)

                # Standard Game of Life rules
                if current_state == 1:
                    if neighbors_sum < 2 or neighbors_sum > 3:
                        new_state[row][col] = Z2(0)
                    elif neighbors_sum == 2 or neighbors_sum == 3:
                        new_state[row][col] = Z2(1)
                else:
                    if neighbors_sum == 3:
                        new_state[row][col] = Z2(1)
        self.state = new_state

    def seed(self, positions):
        for r, c in positions:
            if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                self.state[r][c] = Z2(1)

    def get_display_state(self):
        return np.array([[int(cell) for cell in row] for row in self.state])

# --- MIDI Input Handling ---
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
def draw_grid(screen, game_of_life, fractal_mask):
    display_state = game_of_life.get_display_state()
    for row in range(game_of_life.grid_size):
        for col in range(game_of_life.grid_size):
            x = col * CELL_SIZE
            y = row * CELL_SIZE
            if fractal_mask[row, col]:
                pygame.draw.rect(screen, FRACTAL_COLOR, (x, y, CELL_SIZE, CELL_SIZE))
                if display_state[row, col] == 1:
                    pygame.draw.rect(screen, ALIVE_COLOR, (x, y, CELL_SIZE, CELL_SIZE))
            else:
                pygame.draw.rect(screen, BACKGROUND_COLOR, (x, y, CELL_SIZE, CELL_SIZE))

# --- Main Loop ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Chaos, Fractals, and Life in Z/2Z")

    # Initial fractal shape based on the poem's feeling of spring and lightness
    fractal_mask = create_fractal_mask(GRID_SIZE, equation_type="circle", radius=0.4)
    game = GameOfLife(GRID_SIZE, constraint_mask=fractal_mask)
    game.constraint_mask_type = "circle" # For cycling

    # Initial random seeding within the fractal
    initial_seed_count = 50
    valid_indices = np.argwhere(fractal_mask)
    if len(valid_indices) > 0:
        seed_indices = random.sample(range(len(valid_indices)), min(initial_seed_count, len(valid_indices)))
        initial_positions = [(valid_indices[i][0], valid_indices[i][1]) for i in seed_indices]
        game.seed(initial_positions)

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
                    game.update()
                if event.key == K_r:
                    # Re-seed
                    if len(valid_indices) > 0:
                        seed_indices = random.sample(range(len(valid_indices)), min(initial_seed_count, len(valid_indices)))
                        initial_positions = [(valid_indices[i][0], valid_indices[i][1]) for i in seed_indices]
                        game.seed(initial_positions)
                if event.key == K_f:
                    # Cycle through fractal shapes
                    fractal_options = ["circle", "julia", "polynomial"]
                    current_index = fractal_options.index(game.constraint_mask_type) if hasattr(game, 'constraint_mask_type') and game.constraint_mask_type in fractal_options else -1
                    next_index = (current_index + 1) % len(fractal_options)
                    game.constraint_mask_type = fractal_options[next_index]
                    if game.constraint_mask_type == "circle":
                        fractal_mask = create_fractal_mask(GRID_SIZE, equation_type="circle", radius=0.4)
                    elif game.constraint_mask_type == "julia":
                        fractal_mask = create_fractal_mask(GRID_SIZE, equation_type="julia", c=(-0.7, 0.27015), power=random.randint(2, 5))
                    elif game.constraint_mask_type == "polynomial":
                        fractal_mask = create_fractal_mask(GRID_SIZE, equation_type="polynomial", a=random.uniform(0.1, 0.5))
                    game.constraint_mask = fractal_mask
                    game.state = np.array([[Z2(0) for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]) # Reset

        # MIDI Input Handling
        if midi_input:
            for msg in midi_input.poll():
                if msg.type == 'note_on' and msg.velocity > 0:
                    midi_notes_buffer.append(msg.note)
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in midi_notes_buffer:
                        midi_notes_buffer.remove(msg.note)

            seed_positions = []
            valid_indices = np.argwhere(fractal_mask)
            if len(valid_indices) > 0:
                num_seeds = len(midi_notes_buffer)
                if num_seeds > 0:
                    note_values = np.array(midi_notes_buffer)
                    for i in range(min(num_seeds, len(valid_indices))):
                        note = midi_notes_buffer[i % len(midi_notes_buffer)]
                        grid_index = note % len(valid_indices)
                        row, col = valid_indices[grid_index]
                        seed_positions.append((row, col))
                    game.seed(seed_positions)

        draw_grid(screen, game, fractal_mask)
        pygame.display.flip()
        clock.tick(10)

    if midi_input:
        midi_input.close()
    pygame.quit()

if __name__ == '__main__':
    main()
