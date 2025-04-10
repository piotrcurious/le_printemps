import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# ---------------------------
# Parameters and Setup
# ---------------------------

# Domain and time discretization
N = 256                   # Number of grid points per dimension
dx = 1.0 / N              # Spatial resolution
dt = 1e-3                 # Time step for integration
n_steps = 10000           # Total integration steps
plot_interval = 2000      # Interval for intermediate plot (if desired)

# Schnakenberg model parameters (a reaction-diffusion system that can yield Turing patterns)
a = 0.1
b = 0.9

# Steady-state approximations (for Schnakenberg, u_ss and v_ss satisfy a - u + u^2 v = 0, etc.)
u_ss = a + b
v_ss = b / (u_ss**2)

# Diffusion coefficients base values:
Dv = 0.5                  # Constant inhibitor diffusion coefficient
Du_base = 0.1             # Base activator diffusion coefficient

# An “irrational” constant from field theory / algebraic geometry – using the golden ratio φ
phi = (1 + np.sqrt(5)) / 2  

# Spatial grid for modulation: create coordinates normalized to [0,1]
x = np.linspace(0, 1, N, endpoint=False)
y = np.linspace(0, 1, N, endpoint=False)
X, Y = np.meshgrid(x, y)

# ---------------------------
# Advanced Diffusion Modulation via Algebraic Geometry
# ---------------------------
#
# The following function embodies our “algebraic geometry” modulation.
# Here we define a quadratic form:
#
#    P(x,y) = X^2 + Y^2 - (1/φ)*X*Y
#
# which produces a spatial pattern of values. We then use a sinusoidal function
# to modulate the base diffusion coefficient:
#
#    Du(x,y) = Du_base * (1 + α * sin(2π * P(x,y) / C))
#
# where α controls the intensity of modulation and C is a scaling factor.
#
alpha = 0.5    # modulation intensity
C = 1.0        # scaling factor for the polynomial field

P = X**2 + Y**2 - (1 / phi) * X * Y
Du = Du_base * (1 + alpha * np.sin(2 * np.pi * P / C))

# ---------------------------
# Initialize Fields with Noise
# ---------------------------
u = u_ss + 0.01 * (np.random.rand(N, N) - 0.5)
v = v_ss + 0.01 * (np.random.rand(N, N) - 0.5)

# ---------------------------
# Laplacian Operator using Finite Differences (with periodic boundaries)
# ---------------------------
laplacian_kernel = np.array([[0.05, 0.2, 0.05],
                             [0.2, -1.0, 0.2],
                             [0.05, 0.2, 0.05]])

def laplacian(Z):
    """Compute the Laplacian using convolution and periodic boundaries."""
    return convolve2d(Z, laplacian_kernel, mode='same', boundary='wrap') / (dx**2)

# ---------------------------
# Reaction Terms
# ---------------------------
def reaction_u(u, v):
    """Activator reaction term: f(u,v) = a - u + u^2 * v"""
    return a - u + u**2 * v

def reaction_v(u, v):
    """Inhibitor reaction term: g(u,v) = b - u^2 * v"""
    return b - u**2 * v

# ---------------------------
# Simulation Loop
# ---------------------------
print("Le printemps réveille en moi\n"
      "un désir léger, sans racine ni chemin,\n"
      "comme une âme bohémienne vagabonde,\n"
      "cherchant l’instant, sans destin.\n\n"
      "Les mots se dissolvent au matin,\n"
      "les projets se perdent en l’air du temps,\n"
      "et je reste, incertain,\n"
      "face à l’éclat d’un moment fuyant.\n\n"
      "Tel un cheval libre apprivoisé,\n"
      "vendu aux ombres d’un conte désenchanté,\n"
      "la raison finit par l’enchaîner\n"
      "à une tristesse de plus, à jamais.\n")

for step in range(n_steps):
    # Compute Laplacians for u and v
    Lu = laplacian(u)
    Lv = laplacian(v)

    # Note: Du is a 2D array containing spatially-varying diffusion coefficients for u.
    # This variation (a field morphism in the spirit of algebraic geometry) brings
    # unexpected instabilities that manifest as complex, evolving Turing patterns.
    u += dt * (reaction_u(u, v) + Du * Lu)
    v += dt * (reaction_v(u, v) + Dv * Lv)
    
    # Optional: output intermediate plots to see development
    if (step % plot_interval == 0) and step > 0:
        plt.figure(figsize=(6, 5))
        plt.imshow(u, cmap='inferno', origin='lower')
        plt.title(f"Turing Pattern at step {step}")
        plt.colorbar()
        plt.show()

# ---------------------------
# Final Visualization
# ---------------------------
plt.figure(figsize=(8, 6))
plt.imshow(u, cmap='inferno', origin='lower')
plt.title("Turing Diffusion Patterns\n« Le printemps réveille en moi »", fontsize=14)
plt.xlabel("Spatial X")
plt.ylabel("Spatial Y")
plt.colorbar(label='u concentration')
plt.show()
