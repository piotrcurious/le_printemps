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

# --- Mathematical Structures ---

# Field: GF(2) - Represents the state of a cell (0: dead, 1: alive)
class GF2:
    def __init__(self, value):
        self.value = value % 2

    def __add__(self, other):
        return GF2((self.value + other.value) % 2)

    def __mul__(self, other):
        return GF2((self.value * other.value) % 2)

    def __eq__(self, other):
        return self.value == other.value

    def __int__(self):
        return self.value

# Ring: Polynomial Ring over GF(2) - Can be used to define rules (conceptually)
# For simplicity, we'll use standard Python functions for the Game of Life rules

# --- Fractal Generation using Algebraic Geometry ---

def create_fractal(grid_size, equation_type="circle", center=None, radius=None, power=None):
    """
    Generates a mask representing a fractal shape based on an algebraic equation.
    """
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    x_coords, y_coords = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    normalized_x = (x_coords - grid_size / 2) / (grid_size / 2)
    normalized_y = (y_coords - grid_size / 2) / (grid_size / 2)

    if center is None:
        center_x = 0
        center_y = 0
    else:
        center_x = (center[0] - grid_size / 2) / (grid_size / 2)
        center_y = (center[1] - grid_size / 2) / (grid_size / 2)

    if equation_type == "circle":
        if radius is None:
            radius_sq = 0.5**2
        else:
            radius_sq = (radius / (grid_size / 2))**2
        mask = normalized_x**2 + normalized_y**2 <= radius_sq
    elif equation_type == "julia":
        if power is None:
            power = 2
        c = complex(-0.7, 0.27015)
        max_iter = 50
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
    elif equation_type == "burning_ship":
        max_iter = 50
        for i in range(grid_size):
            for j in range(grid_size):
                c = complex(normalized_x[i, j], normalized_y[i, j])
                z = 0
                for _ in range(max_iter):
                    z = complex(abs(z.real), abs(z.imag))**2 + c
                    if abs(z) > 2:
                        mask[i, j] = False
                        break
                else:
                    mask[i, j] = True
    elif equation_type == "polynomial":
        # Example polynomial: x^3 + y^3 - a*x*y = 0 (Folium of Descartes)
        if power is None:
            power = 1
        a = 0.3 * power
        condition = normalized_x**3 + normalized_y**3 - a * normalized_x * normalized_y
        mask = np.abs(condition) < 0.01 # Adjust threshold for thickness
    return mask

# --- Game of Life Implementation ---

class GameOfLife:
    def __init__(self, grid_size, initial_state=None, constraint_mask=None):
        self.grid_size = grid_size
        if initial_state is None:
            self.state = np.zeros((grid_size, grid_size), dtype=int)
        else:
            self.state = initial_state
        self.constraint_mask = constraint_mask

    def get_neighbors(self, row, col):
        neighbors = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                r, c = (row + i + self.grid_size) % self.grid_size, (col + j + self.grid_size) % self.grid_size
                neighbors.append(self.state[r, c])
        return neighbors

    def update(self):
        new_state = np.copy(self.state)
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.constraint_mask is not None and not self.constraint_mask[row, col]:
                    new_state[row, col] = 0 # Constrain to the fractal
                    continue

                neighbors = self.get_neighbors(row, col)
                live_neighbors = sum(neighbors)
                cell = self.state[row, col]

                if cell == 1:
                    if live_neighbors < 2 or live_neighbors > 3:
                        new_state[row, col] = 0
                    elif live_neighbors == 2 or live_neighbors == 3:
                        new_state[row, col] = 1
                else:
                    if live_neighbors == 3:
                        new_state[row, col] = 1
        self.state = new_state

    def seed(self, positions):
        for r, c in positions:
            if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                self.state[r, c] = 1

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
    for row in range(game_of_life.grid_size):
        for col in range(game_of_life.grid_size):
            x = col * CELL_SIZE
            y = row * CELL_SIZE
            if fractal_mask[row, col]:
                pygame.draw.rect(screen, FRACTAL_COLOR, (x, y, CELL_SIZE, CELL_SIZE))
                if game_of_life.state[row, col] == 1:
                    pygame.draw.rect(screen, ALIVE_COLOR, (x, y, CELL_SIZE, CELL_SIZE))
            else:
                pygame.draw.rect(screen, BACKGROUND_COLOR, (x, y, CELL_SIZE, CELL_SIZE))

# --- Main Loop ---

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Chaotic Fractal Life")

    # --- Poem Interpretation for Initial Settings ---
    # "Le printemps réveille en moi un désir léger, sans racine ni chemin,"
    # -> Initial light seeding, possibly random

    # "comme une âme bohémienne vagabonde,"
    # -> Wandering, dynamic behavior

    # "cherchant l’instant, sans destin."
    # -> Fleeting moments, unpredictable evolution

    # "Les mots se dissolvent au matin,"
    # -> Patterns may fade

    # "les projets se perdent en l’air du temps,"
    # -> Structures might dissolve

    # "et je reste, incertain,"
    # -> Uncertainty in the evolution

    # "face à l’éclat d’un moment fuyant."
    # -> Capture the beauty of transient states

    # "Tel un cheval libre apprivoisé, vendu aux ombres d’un conte désenchanté,"
    # -> Initial freedom constrained

    # "la raison finit par l’enchaîner"
    # -> Algebraic geometry as the constraint

    # "à une tristesse de plus, à jamais."
    # -> Potentially settling into stable or periodic patterns within the constraint

    # Initial fractal shape based on the poem's feeling of spring and lightness
    fractal_mask = create_fractal(GRID_SIZE, equation_type="circle", radius=0.4)
    game = GameOfLife(GRID_SIZE, constraint_mask=fractal_mask)

    # Initial random seeding (inspired by the first line of the poem)
    initial_seed_count = 50
    initial_positions = random.sample([(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if fractal_mask[r, c]], min(initial_seed_count, np.sum(fractal_mask)))
    game.seed(initial_positions)

    midi_input = open_midi_input(MIDI_DEVICE_NAME)
    midi_notes_buffer = []

    running = True
    clock = pygame.time.Clock()
    generation_counter = 0

    while running:
        screen.fill(BACKGROUND_COLOR)

        # --- Handle Events ---
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            if event.type == KEYDOWN:
                if event.key == K_SPACE:
                    game.update()
                    generation_counter += 1
                if event.key == K_r:
                    # Re-seed based on the poem's initial feeling
                    initial_positions = random.sample([(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if fractal_mask[r, c]], min(initial_seed_count, np.sum(fractal_mask)))
                    game.seed(initial_positions)
                    generation_counter = 0
                if event.key == K_f:
                    # Cycle through different fractal shapes, reflecting the poem's progression
                    fractal_options = ["circle", "julia", "burning_ship", "polynomial"]
                    current_index = fractal_options.index(game.constraint_mask_type) if hasattr(game, 'constraint_mask_type') and game.constraint_mask_type in fractal_options else -1
                    next_index = (current_index + 1) % len(fractal_options)
                    game.constraint_mask_type = fractal_options[next_index]
                    if game.constraint_mask_type == "circle":
                        fractal_mask = create_fractal(GRID_SIZE, equation_type="circle", radius=0.4)
                    elif game.constraint_mask_type == "julia":
                        fractal_mask = create_fractal(GRID_SIZE, equation_type="julia", power=random.randint(2, 5))
                    elif game.constraint_mask_type == "burning_ship":
                        fractal_mask = create_fractal(GRID_SIZE, equation_type="burning_ship")
                    elif game.constraint_mask_type == "polynomial":
                        fractal_mask = create_fractal(GRID_SIZE, equation_type="polynomial", power=random.uniform(0.5, 2))
                    game.constraint_mask = fractal_mask
                    game.state = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int) # Reset game state

        # --- Handle MIDI Input ---
        if midi_input:
            for msg in midi_input.poll():
                if msg.type == 'note_on' and msg.velocity > 0:
                    midi_notes_buffer.append(msg.note)
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in midi_notes_buffer:
                        midi_notes_buffer.remove(msg.note)

            # Seed Game of Life based on currently pressed MIDI notes
            seed_positions = []
            valid_fractal_indices = np.argwhere(fractal_mask)
            if valid_fractal_indices.size > 0:
                num_seeds = len(midi_notes_buffer)
                if num_seeds > 0:
                    note_values = np.array(midi_notes_buffer)
                    # Map MIDI notes to grid coordinates (simple mapping)
                    for i in range(min(num_seeds, len(valid_fractal_indices))):
                        note = midi_notes_buffer[i % len(midi_notes_buffer)]
                        grid_index = note % len(valid_fractal_indices)
                        row, col = valid_fractal_indices[grid_index]
                        seed_positions.append((row, col))
                    game.seed(seed_positions)

        # --- Update and Draw ---
        game.update()
        draw_grid(screen, game, fractal_mask)

        pygame.display.flip()
        clock.tick(10) # Adjust speed

    if midi_input:
        midi_input.close()
    pygame.quit()

if __name__ == '__main__':
    main()
