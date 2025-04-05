import pygame import numpy as np import math import random

Initialize Pygame

pygame.init() screen_size = (800, 800) screen = pygame.display.set_mode(screen_size) pygame.display.set_caption("Le printemps réveille en moi") clock = pygame.time.Clock()

Colors (Spring tones)

BG_COLOR = (240, 255, 245) LIFE_COLOR = (80, 200, 120) CURVE_COLOR = (120, 180, 255) FRACTAL_COLOR = (255, 220, 180) POEM_COLOR = (50, 60, 70)

Font setup

font = pygame.font.SysFont("serif", 20) poem_lines = [ "Le printemps réveille en moi", "un désir léger, sans racine ni chemin,", "comme une âme bohémienne vagabonde,", "cherchant l’instant, sans destin.", "", "Les mots se dissolvent au matin,", "les projets se perdent en l’air du temps,", "et je reste, incertain,", "face à l’éclat d’un moment fuyant.", "", "Tel un cheval libre apprivoisé,", "vendu aux ombres d’un conte désenchanté,", "la raison finit par l’enchaîner", "à une tristesse de plus, à jamais." ]

Game of Life setup

grid_size = 100 cell_size = screen_size[0] // grid_size grid = np.random.choice([0, 1], size=(grid_size, grid_size))

Algebraic mask (circle + rose)

def algebraic_mask(x, y, t): cx, cy = grid_size // 2, grid_size // 2 dx = x - cx dy = y - cy r = math.sqrt(dx2 + dy2) theta = math.atan2(dy, dx) rose = math.sin(4 * theta + 0.5 * math.sin(t)) * 20 + 40 return r < rose

Chaos controller (logistic map)

chaos_r = 3.95 chaos_x = 0.4 def chaos_step(): global chaos_x chaos_x = chaos_r * chaos_x * (1 - chaos_x) return chaos_x

Fractal bloom (simple recursive tree)

def draw_fractal(x, y, angle, depth): if depth == 0: return length = depth * 6 x2 = x + int(math.cos(angle) * length) y2 = y - int(math.sin(angle) * length) pygame.draw.line(screen, FRACTAL_COLOR, (x, y), (x2, y2), 1) draw_fractal(x2, y2, angle - 0.3 + chaos_step() * 0.6, depth - 1) draw_fractal(x2, y2, angle + 0.3 - chaos_step() * 0.6, depth - 1)

Game of Life update

def update_life(): global grid new_grid = np.copy(grid) for i in range(grid_size): for j in range(grid_size): total = sum([ grid[(i-1)%grid_size, (j-1)%grid_size], grid[(i-1)%grid_size, j], grid[(i-1)%grid_size, (j+1)%grid_size], grid[i, (j-1)%grid_size], grid[i, (j+1)%grid_size], grid[(i+1)%grid_size, (j-1)%grid_size], grid[(i+1)%grid_size, j], grid[(i+1)%grid_size, (j+1)%grid_size] ]) alive = grid[i, j] == 1 if alive and (total < 2 or total > 3): new_grid[i, j] = 0 elif not alive and total == 3: new_grid[i, j] = 1

# Apply algebraic geometry mask
        if not algebraic_mask(i, j, pygame.time.get_ticks() * 0.001):
            new_grid[i, j] = 0
grid = new_grid

Poem fade engine

def get_poem_line_index(): t = pygame.time.get_ticks() * 0.001 return int((math.sin(t * 0.1) + 1) / 2 * (len(poem_lines) - 1))

Main loop

running = True while running: screen.fill(BG_COLOR)

for event in pygame.event.get():
    if event.type == pygame.QUIT:
        running = False

update_life()

# Draw Game of Life
for i in range(grid_size):
    for j in range(grid_size):
        if grid[i, j] == 1:
            rect = pygame.Rect(i * cell_size, j * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, LIFE_COLOR, rect)

# Draw fractals from life activity
entropy = np.sum(grid)
if entropy > grid_size * 10:
    for _ in range(5):
        angle = random.uniform(-0.5, 0.5)
        depth = random.randint(3, 5)
        x = random.randint(200, 600)
        y = random.randint(400, 700)
        draw_fractal(x, y, angle, depth)

# Draw current poetic line
idx = get_poem_line_index()
line_surface = font.render(poem_lines[idx], True, POEM_COLOR)
screen.blit(line_surface, (20, 20))

pygame.display.flip()
clock.tick(30)

pygame.quit()

