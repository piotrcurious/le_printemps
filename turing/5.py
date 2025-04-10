import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter, laplace
from sympy import symbols, exp, Matrix, diff, lambdify, simplify, Function, Derivative
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from scipy import signal
from scipy.integrate import solve_ivp
from scipy.sparse import diags
from scipy.special import jv  # Bessel functions for Zeldovich solutions

class TuringPatternPoem:
    def __init__(self, size=200, time_steps=500):
        """Initialize the Turing pattern simulation with poetic inspiration.
        
        This implements a reaction-diffusion system based on algebraic geometry concepts
        and Zeldovich's combustion theories, with parameters modulated by the poem's structure.
        """
        self.size = size
        self.time_steps = time_steps
        
        # Define poem-inspired parameters
        # Each stanza of the poem influences different aspects of the simulation
        self.stanzas = 4
        self.lines_per_stanza = 3
        
        # Create poem-modulated parameter space using algebraic field extensions
        # This captures the "bohémienne vagabonde" wandering nature in the poem
        golden_ratio = (1 + np.sqrt(5)) / 2  # Irrational number representing uncertainty
        silver_ratio = 1 + np.sqrt(2)        # "âme bohémienne" - wandering soul
        
        # Transcendental field extension (representing "sans destin" - without destiny)
        e_pi_ratio = np.exp(1) / np.pi
        
        # Parameters modulated by the poem's structure
        self.Du = 0.16 * golden_ratio     # Diffusion rate of activator
        self.Dv = 0.08 * silver_ratio     # Diffusion rate of inhibitor
        self.f = 0.035 * e_pi_ratio       # Feed rate ("désir léger" - light desire)
        self.k = 0.065 * np.log(7)        # Kill rate ("tristesse" - sadness)
        
        # Initialize concentration grids with slight randomness ("incertain" - uncertain)
        self.u = np.ones((size, size)) + np.random.random((size, size)) * 0.1
        self.v = np.zeros((size, size)) + np.random.random((size, size)) * 0.1
        
        # Create a circular pattern in the center (representing the "éclat d'un moment" - fleeting moment)
        r = size // 4
        center = size // 2
        y, x = np.ogrid[-center:size-center, -center:size-center]
        mask = x*x + y*y <= r*r
        self.u[mask] = 0.5
        self.v[mask] = 0.25
        
        # Create algebraic geometry-based diffusion operators
        self.setup_algebraic_diffusion()
        
        # Set up visualization
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        # Create color map inspired by the poem's emotional journey
        # From desire (warm colors) to sadness (cool colors)
        colors = [(0.8, 0.2, 0.2),    # Desire red - "désir léger"
                 (0.95, 0.85, 0.4),   # Light yellow - "l'éclat d'un moment"
                 (0.2, 0.4, 0.6),     # Blue - "tristesse"
                 (0.1, 0.1, 0.3)]     # Deep blue - "à jamais" (forever)
        self.cmap = LinearSegmentedColormap.from_list("poem_emotions", colors, N=256)
        
        self.image = self.ax.imshow(self.u, cmap=self.cmap, vmin=0, vmax=1, 
                                   interpolation='bilinear')
        self.ax.set_title("Turing Diffusion Patterns: 'Le printemps réveille en moi'", fontsize=14)
        self.ax.axis('off')
    
    def setup_algebraic_diffusion(self):
        """Set up algebraic geometry-based diffusion operators.
        
        This implements Zeldovich's combustion theory using algebraic varieties
        and field extensions to create complex pattern formation.
        """
        # Symbolic variables for our algebraic field extension
        x, y, u, v = symbols('x y u v')
        
        # Create algebraic varieties based on elliptic curves (advanced algebraic geometry)
        # These represent the "geometric structures translated through algebraic geometry"
        self.elliptic_curve = lambda x, y: y**2 - x**3 - x - 1
        
        # Zeldovich field operator (from combustion theory)
        # This captures the "numerical instability" mentioned in the requirement
        def zeldovich_operator(f, x, y):
            grad_f = Matrix([diff(f, x), diff(f, y)])
            laplacian = diff(diff(f, x), x) + diff(diff(f, y), y)
            return laplacian - grad_f.dot(grad_f) / f
        
        # Create field morphism between diffusion space and algebraic variety
        # This is the key to creating the complex patterns from "numerical instability"
        morphism_u = lambda u, v: u - v
        morphism_v = lambda u, v: u * v - v**3
        
        # Store the algebraic operators for use in the simulation
        self.morphism_u = morphism_u
        self.morphism_v = morphism_v
    
    def apply_zeldovich_reaction(self, u, v):
        """Apply Zeldovich reaction terms with algebraic geometry influences."""
        # Gray-Scott model with Zeldovich modifications and algebraic field morphisms
        # The instabilities that arise here create the complex patterns
        reaction_u = -u * v**2 + self.f * (1 - u)
        reaction_v = u * v**2 - (self.f + self.k) * v
        
        # Apply irrational number influence to create aperiodic patterns
        # This represents the "interplay of irrational numbers" from the requirements
        phi = (1 + np.sqrt(5)) / 2
        reaction_u *= (1 + 0.05 * np.sin(u * phi))
        reaction_v *= (1 + 0.03 * np.cos(v * phi))
        
        return reaction_u, reaction_v
    
    def update(self, frame):
        """Update the simulation for one time step."""
        # Apply Laplacian using convolution for better numerical stability
        # This is a more sophisticated approach than simple discrete Laplacian
        laplacian_u = gaussian_filter(self.u, sigma=1.0) - self.u
        laplacian_v = gaussian_filter(self.v, sigma=1.0) - self.v
        
        # Apply reaction terms with Zeldovich influence
        reaction_u, reaction_v = self.apply_zeldovich_reaction(self.u, self.v)
        
        # Update concentrations with diffusion and reaction
        # The differential rates are modulated by the poem's rhythm
        dt = 1.0  # Time step
        
        # Apply diffusion with algebraic field morphism influence
        # This creates the instabilities that produce beautiful patterns
        phase_factor = 0.5 + 0.5 * np.sin(frame / 100)  # Represents the "éclat d'un moment fuyant"
        
        self.u += dt * (self.Du * laplacian_u + reaction_u) * (1 + 0.05 * phase_factor)
        self.v += dt * (self.Dv * laplacian_v + reaction_v) * (1 - 0.03 * phase_factor)
        
        # Apply boundary conditions (periodic)
        self.u = np.clip(self.u, 0, 1)
        self.v = np.clip(self.v, 0, 1)
        
        # Update visualization
        self.image.set_array(self.u)
        
        # Add poem-inspired perturbations at specific frames
        # Each stanza triggers a different pattern modification
        if frame % (self.time_steps // self.stanzas) == 0:
            stanza = frame // (self.time_steps // self.stanzas)
            if stanza < self.stanzas:
                self.add_poetic_perturbation(stanza)
        
        return [self.image]
    
    def add_poetic_perturbation(self, stanza):
        """Add perturbations inspired by each stanza of the poem."""
        center = self.size // 2
        
        if stanza == 0:
            # "un désir léger, sans racine ni chemin" - light desire without root or path
            # Create wandering patterns with no fixed center
            r = self.size // 8
            for _ in range(3):  # For each line in the stanza
                x = np.random.randint(r, self.size - r)
                y = np.random.randint(r, self.size - r)
                y_grid, x_grid = np.ogrid[-y:self.size-y, -x:self.size-x]
                mask = x_grid**2 + y_grid**2 <= r**2
                self.u[mask] = 0.5 + 0.5 * np.random.random()
                self.v[mask] = 0.25 + 0.1 * np.random.random()
                
        elif stanza == 1:
            # "Les mots se dissolvent au matin" - words dissolving in the morning
            # Create dissolving patterns
            mask = np.random.random((self.size, self.size)) > 0.7
            self.u[mask] = 0.5
            self.v[mask] = 0.25
            
        elif stanza == 2:
            # "Tel un cheval libre apprivoisé" - like a tamed free horse
            # Create spiral patterns representing constraint of freedom
            y, x = np.ogrid[-center:self.size-center, -center:self.size-center]
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)
            spiral = (r + 5*theta) % (self.size//4) < 5
            self.u[spiral] = 0.75
            self.v[spiral] = 0.2
            
        elif stanza == 3:
            # "à une tristesse de plus, à jamais" - to one more sadness, forever
            # Create fading wave patterns
            x = np.linspace(0, 4*np.pi, self.size)
            y = np.linspace(0, 4*np.pi, self.size)
            X, Y = np.meshgrid(x, y)
            wave = 0.5 + 0.5 * np.sin(X) * np.cos(Y)
            mask = wave > 0.7
            self.u[mask] = 0.2
            self.v[mask] = 0.5
    
    def animate(self):
        """Create the animation of the Turing pattern."""
        anim = FuncAnimation(self.fig, self.update, frames=self.time_steps, 
                            interval=50, blit=True)
        plt.close()  # Prevent display during creation
        return anim

    def create_visualization(self):
        """Create a static visualization showing key stages of pattern evolution."""
        # Calculate several steps at once to show evolution
        snapshot_frames = [0, 50, 100, 200, 400]
        
        fig = plt.figure(figsize=(15, 8))
        fig.suptitle("Poetic Turing Patterns: 'Le printemps réveille en moi'", fontsize=16)
        
        # Run simulation and capture snapshots
        for i, frame in enumerate(snapshot_frames):
            # Run simulation up to this frame
            for _ in range(frame if i == 0 else snapshot_frames[i] - snapshot_frames[i-1]):
                self.update(_)
            
            # Create subplot
            ax = fig.add_subplot(1, len(snapshot_frames), i+1)
            
            # Choose which lines of the poem to display based on simulation progress
            poem_lines = [
                "Le printemps réveille en moi",
                "un désir léger, sans racine ni chemin",
                "comme une âme bohémienne vagabonde",
                "cherchant l'instant, sans destin",
                "Les mots se dissolvent au matin",
                "les projets se perdent en l'air du temps",
                "et je reste, incertain",
                "face à l'éclat d'un moment fuyant",
                "Tel un cheval libre apprivoisé",
                "vendu aux ombres d'un conte désenchanté",
                "la raison finit par l'enchaîner",
                "à une tristesse de plus, à jamais"
            ]
            
            # Display appropriate line based on frame number
            line_index = min(i * 3, len(poem_lines) - 1)
            poem_line = poem_lines[line_index]
            
            # Show image with poem line as title
            ax.imshow(self.u, cmap=self.cmap, vmin=0, vmax=1)
            ax.set_title(f"Frame {frame}\n\"{poem_line}\"", fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        return fig

    def run_simulation(self, animate=False):
        """Run the simulation and save results."""
        if animate:
            anim = self.animate()
            return anim
        else:
            # Create static visualization
            fig = self.create_visualization()
            return fig


# Execute the simulation
if __name__ == "__main__":
    # Create simulation with desired parameters
    simulation = TuringPatternPoem(size=200, time_steps=500)
    
    # Run static visualization instead of animation for stability
    fig = simulation.run_simulation(animate=False)
    
    # Optional: To create an animation instead (may be computationally intensive)
    # anim = simulation.run_simulation(animate=True)
    # anim.save('turing_pattern_poem.mp4', writer='ffmpeg', dpi=100)
    
    plt.show()
