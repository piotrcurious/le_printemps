import numpy as np
import pygame
import pygame.midi
import math
import random
import time
from scipy.ndimage import convolve
from sympy import symbols, simplify, Poly
from numba import jit

"""
Integrated Chaos Theory, Fractals, Algebraic Geometry, and Game of Life Visualization
Inspired by the poem "Le printemps réveille en moi"

The demo creates a visual representation where:
- Algebraic geometry defines boundaries and structures
- Game of Life evolves within these constraints
- Fractal patterns emerge through recursive processes
- Chaos theory principles guide transitions and evolution
- MIDI input influences parameter changes
"""

# Initialize pygame and display
pygame.init()
pygame.midi.init()

# Constants
WIDTH, HEIGHT = 1280, 720
CELL_SIZE = 4
GRID_WIDTH = WIDTH // CELL_SIZE
GRID_HEIGHT = HEIGHT // CELL_SIZE

# Colors based on poem themes
BACKGROUND = (10, 10, 20)
LIFE_COLOR = (100, 200, 100)  # Spring green
FRACTAL_COLOR = (200, 150, 50)  # Earth tones
GEOMETRY_COLOR = (50, 150, 200)  # Sky blue
CHAOS_COLOR = (200, 80, 100)  # Passion red

# Set up display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chaos-Fractal-Life Symphony")

# Algebraic Geometry System using Ring Theory
class AlgebraicField:
    def __init__(self):
        # Define symbolic variables for our field
        self.x, self.y, self.t = symbols('x y t')
        # Create a set of polynomial rings that will define our geometric constraints
        self.update_polynomials()
        self.t_value = 0
        
    def update_polynomials(self):
        # Create a family of polynomials that evolve over time
        self.poly1 = Poly((self.x**2 + self.y**2 - 0.7)**2 - 0.01*self.x*self.y, self.x, self.y)
        self.poly2 = Poly(self.x**3 - 3*self.x*self.y**2 + 0.1*self.t, self.x, self.y)
        self.poly3 = Poly(self.y**3 - 3*self.y*self.x**2 - 0.1*self.t, self.x, self.y)
        
    def evaluate(self, x, y):
        # Evaluate the polynomial at given coordinates
        # This creates algebraic varieties (geometric shapes defined by polynomials)
        try:
            val1 = float(self.poly1.eval(x/GRID_WIDTH*4-2, y/GRID_HEIGHT*4-2))
            val2 = float(self.poly2.eval(x/GRID_WIDTH*4-2, y/GRID_HEIGHT*4-2).subs(self.t, self.t_value))
            val3 = float(self.poly3.eval(x/GRID_WIDTH*4-2, y/GRID_HEIGHT*4-2).subs(self.t, self.t_value))
            
            # Combine the polynomials to create interesting varieties
            result = np.sin(val1) * np.cos(val2) * np.sin(val3)
            return result
        except:
            return 0
            
    def increment_time(self, amount=0.01):
        self.t_value += amount
        # Periodically update polynomials to create new forms
        if int(self.t_value * 10) % 50 == 0:
            self.update_polynomials()


# Fractal System using Julia sets
@jit(nopython=True)
def julia_set(z, c, max_iter):
    # Compute Julia set fractal
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z*z + c
        n += 1
    return n

def generate_julia(c, max_iter=20):
    # Create Julia set array
    julia = np.zeros((GRID_HEIGHT, GRID_WIDTH))
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            # Map coordinates to complex plane
            zx = 3.0 * (x - GRID_WIDTH/2) / (GRID_WIDTH/2)
            zy = 2.0 * (y - GRID_HEIGHT/2) / (GRID_HEIGHT/2)
            z = complex(zx, zy)
            # Calculate fractal value
            julia[y, x] = julia_set(z, c, max_iter)
    return julia / max_iter


# Game of Life with algebraic constraints
class FractalLifeAutomaton:
    def __init__(self):
        # Initialize grid for Conway's Game of Life
        self.grid = np.zeros((GRID_HEIGHT, GRID_WIDTH))
        self.next_grid = np.zeros((GRID_HEIGHT, GRID_WIDTH))
        self.algebraic_field = AlgebraicField()
        self.c_param = complex(-0.7, 0.27)  # Initial Julia parameter
        self.julia_array = generate_julia(self.c_param)
        self.kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        
        # Randomize initial state
        self.randomize(density=0.3)
        
    def randomize(self, density=0.3):
        # Create random initial state with given cell density
        self.grid = np.random.choice([0, 1], size=(GRID_HEIGHT, GRID_WIDTH), 
                                   p=[1-density, density])
                                   
    def constrain_with_algebraic_geometry(self):
        # Apply algebraic constraints to the grid
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                # Only allow life in areas defined by our algebraic variety
                constraint = self.algebraic_field.evaluate(x, y)
                if abs(constraint) > 0.1:
                    self.grid[y, x] = 0
                    
    def constrain_with_fractal(self):
        # Use fractal pattern to influence life grid
        fractal_mask = self.julia_array > 0.7
        self.grid = self.grid * fractal_mask
        
    def update(self):
        # Standard Game of Life rules
        # Count neighbors using convolution
        neighbors = convolve(self.grid, self.kernel, mode='wrap')
        
        # Apply Game of Life rules
        birth = (neighbors == 3) & (self.grid == 0)
        survive = ((neighbors == 2) | (neighbors == 3)) & (self.grid == 1)
        self.next_grid = np.zeros_like(self.grid)
        self.next_grid[birth | survive] = 1
        
        # Apply constraints
        self.grid = self.next_grid
        self.constrain_with_algebraic_geometry()
        self.constrain_with_fractal()
        
        # Update the mathematical field
        self.algebraic_field.increment_time()
        
    def draw(self, surface):
        # Clear the screen
        surface.fill(BACKGROUND)
        
        # Draw Game of Life cells
        life_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        life_surface.fill((0,0,0,0))
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.grid[y, x] > 0:
                    pygame.draw.rect(life_surface, (*LIFE_COLOR, 200), 
                                    (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        # Draw algebraic geometry field
        geometry_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        geometry_surface.fill((0,0,0,0))
        for y in range(0, GRID_HEIGHT, 2):
            for x in range(0, GRID_WIDTH, 2):
                value = self.algebraic_field.evaluate(x, y)
                if abs(value) < 0.05:
                    alpha = int(255 * (1 - abs(value) * 20))
                    pygame.draw.rect(geometry_surface, (*GEOMETRY_COLOR, alpha), 
                                   (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE*2, CELL_SIZE*2), 1)
        
        # Draw Julia fractal
        fractal_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        fractal_surface.fill((0,0,0,0))
        for y in range(0, GRID_HEIGHT, 1):
            for x in range(0, GRID_WIDTH, 1):
                if self.julia_array[y, x] > 0.7:
                    alpha = int(128 * self.julia_array[y, x])
                    pygame.draw.rect(fractal_surface, (*FRACTAL_COLOR, alpha), 
                                   (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE), 0)
        
        # Blend the layers
        surface.blit(fractal_surface, (0, 0))
        surface.blit(geometry_surface, (0, 0))
        surface.blit(life_surface, (0, 0))
        
    def update_fractal(self, cr, ci):
        # Update the Julia parameter
        self.c_param = complex(cr, ci)
        self.julia_array = generate_julia(self.c_param)


# Chaos Theory System: Lorenz Attractor
class LorenzAttractor:
    def __init__(self):
        # Lorenz system parameters
        self.sigma = 10.0
        self.rho = 28.0
        self.beta = 8.0 / 3.0
        
        # Current state
        self.x, self.y, self.z = 0.1, 0.0, 0.0
        
        # History of points for the attractor
        self.points = []
        self.max_points = 1000
        
    def update(self, dt=0.01):
        # Lorenz equations
        dx = self.sigma * (self.y - self.x)
        dy = self.x * (self.rho - self.z) - self.y
        dz = self.x * self.y - self.beta * self.z
        
        # Update state
        self.x += dx * dt
        self.y += dy * dt
        self.z += dz * dt
        
        # Save point
        self.points.append((self.x, self.y, self.z))
        if len(self.points) > self.max_points:
            self.points.pop(0)
            
    def draw(self, surface):
        # Project and draw the Lorenz attractor
        if len(self.points) < 2:
            return
            
        chaos_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        chaos_surface.fill((0,0,0,0))
        
        # Scale factors for projection
        scale_x = 5
        scale_y = 5
        offset_x = WIDTH // 2
        offset_y = HEIGHT // 2
        
        # Draw lines connecting the points
        for i in range(1, len(self.points)):
            x1, y1, z1 = self.points[i-1]
            x2, y2, z2 = self.points[i]
            
            # Project 3D to 2D (simple perspective)
            px1 = int(x1 * scale_x + offset_x)
            py1 = int(y1 * scale_y + offset_y)
            px2 = int(x2 * scale_x + offset_x)
            py2 = int(y2 * scale_y + offset_y)
            
            # Color based on z-coordinate
            alpha = min(255, 100 + int(abs(z2) * 5))
            color = (CHAOS_COLOR[0], CHAOS_COLOR[1], CHAOS_COLOR[2], alpha)
            
            # Draw the line
            pygame.draw.line(chaos_surface, color, (px1, py1), (px2, py2), 2)
            
        surface.blit(chaos_surface, (0, 0))
        
    def set_parameters(self, sigma=None, rho=None, beta=None):
        # Update Lorenz parameters
        if sigma is not None:
            self.sigma = sigma
        if rho is not None:
            self.rho = rho
        if beta is not None:
            self.beta = beta


# MIDI Handler
class MidiHandler:
    def __init__(self):
        # Initialize MIDI input
        self.midi_input = None
        try:
            input_id = pygame.midi.get_default_input_id()
            if input_id != -1:
                self.midi_input = pygame.midi.Input(input_id)
                print(f"MIDI input device initialized: {pygame.midi.get_device_info(input_id)}")
            else:
                print("No default MIDI input device available")
        except:
            print("Error initializing MIDI input")
            
    def poll(self):
        # Check for MIDI events
        if self.midi_input and self.midi_input.poll():
            midi_events = self.midi_input.read(10)
            return midi_events
        return []
        
    def close(self):
        # Clean up
        if self.midi_input:
            self.midi_input.close()


# Main program
def main():
    clock = pygame.time.Clock()
    running = True
    paused = False
    
    # Initialize systems
    life_system = FractalLifeAutomaton()
    chaos_system = LorenzAttractor()
    midi_handler = MidiHandler()
    
    # Parameters for visualization evolution
    evolution_phase = 0
    evolution_counter = 0
    
    # Font for text
    font = pygame.font.SysFont('Arial', 16)
    
    while running:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    life_system.randomize()
                elif event.key == pygame.K_1:
                    # Change fractal parameter
                    life_system.update_fractal(-0.7, 0.27)
                elif event.key == pygame.K_2:
                    # Change fractal parameter
                    life_system.update_fractal(-0.4, 0.6)
                elif event.key == pygame.K_3:
                    # Change fractal parameter
                    life_system.update_fractal(0.285, 0.01)
        
        # Process MIDI input
        midi_events = midi_handler.poll()
        for event in midi_events:
            # event[0] is the status and data
            # For note on: [[[status, note, velocity, 0], timestamp], ...]
            midi_event = event[0]
            if len(midi_event) >= 3:  # Ensure we have enough data
                status, note, velocity = midi_event[0], midi_event[1], midi_event[2]
                
                # Note On event (typically status 144-159)
                if 144 <= status <= 159 and velocity > 0:
                    print(f"Note On: {note}, velocity: {velocity}")
                    
                    # Map MIDI notes to system parameters
                    if note < 48:  # Low notes affect chaos parameters
                        chaos_system.set_parameters(
                            sigma=10 + (note - 36) * 0.5,
                            rho=28 + (velocity / 127) * 10
                        )
                    elif note < 72:  # Mid notes affect fractal parameters
                        cr = -0.7 + (note - 48) / 24
                        ci = 0.27 + (velocity / 127) * 0.3
                        life_system.update_fractal(cr, ci)
                    else:  # High notes affect Game of Life
                        density = velocity / 127
                        life_system.randomize(density)
        
        # Update systems if not paused
        if not paused:
            life_system.update()
            chaos_system.update()
            
            # Evolutionary phases based on poem structure
            evolution_counter += 1
            if evolution_counter > 100:
                evolution_counter = 0
                evolution_phase = (evolution_phase + 1) % 4
                
                # Each phase represents a stanza of the poem
                if evolution_phase == 0:  # Desire without root or path
                    life_system.update_fractal(-0.7, 0.27)
                    chaos_system.set_parameters(sigma=10, rho=28, beta=8/3)
                elif evolution_phase == 1:  # Words dissolving in morning
                    life_system.update_fractal(-0.4, 0.6)
                    chaos_system.set_parameters(sigma=12, rho=25, beta=8/3)
                elif evolution_phase == 2:  # Free horse tamed
                    life_system.update_fractal(0.285, 0.01)
                    chaos_system.set_parameters(sigma=8, rho=30, beta=9/3)
                elif evolution_phase == 3:  # Reason chained to sadness
                    life_system.update_fractal(-0.8, -0.2)
                    chaos_system.set_parameters(sigma=15, rho=20, beta=7/3)
        
        # Draw everything
        life_system.draw(screen)
        chaos_system.draw(screen)
        
        # Display current phase (poem stanza)
        phase_text = [
            "Le printemps réveille en moi un désir léger, sans racine ni chemin",
            "Les mots se dissolvent au matin, les projets se perdent en l'air",
            "Tel un cheval libre apprivoisé, vendu aux ombres d'un conte",
            "La raison finit par l'enchaîner à une tristesse de plus"
        ]
        
        text = font.render(phase_text[evolution_phase], True, (255, 255, 255))
        screen.blit(text, (10, HEIGHT - 30))
        
        # Update display
        pygame.display.flip()
        clock.tick(30)
    
    # Clean up
    midi_handler.close()
    pygame.midi.quit()
    pygame.quit()


if __name__ == "__main__":
    main()
