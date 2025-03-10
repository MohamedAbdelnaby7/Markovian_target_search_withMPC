import numpy as np
import matplotlib.pyplot as plt
import random

# ---------------------- Parameters ---------------------- #
GRID_SIZE = 50  # Size of the search grid
NUM_ROBOTS = 8  # Number of searching robots
MAX_STEPS = 300  # Maximum simulation steps
TARGET_MOVE_PROB = 0.8  # Probability of target moving
COMMUNICATION_RADIUS = 50  # How far robots can communicate

# ---------------------- Target (Markov Movement) ---------------------- #
class Target:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.position = np.random.randint(0, grid_size, size=2)
        self.transition_matrix = self.create_transition_matrix()

    def create_transition_matrix(self):
        """Defines the probability of moving in each direction."""
        return {
            (0, 1): 0.25,   # Move Right
            (0, -1): 0.25,  # Move Left
            (1, 0): 0.25,   # Move Down
            (-1, 0): 0.25   # Move Up
        }

    def move(self):
        """Moves based on transition probabilities."""
        if np.random.rand() < TARGET_MOVE_PROB:
            move_choices, probabilities = zip(*self.transition_matrix.items())
            move = move_choices[np.random.choice(len(move_choices), p=probabilities)]
            self.position = np.clip(self.position + move, 0, self.grid_size-1)

# ---------------------- Bayesian Robot ---------------------- #
class Robot:
    def __init__(self, grid_size, robot_id):
        self.grid_size = grid_size
        self.robot_id = robot_id
        self.position = np.random.randint(0, grid_size, size=2)
        self.belief = np.ones((grid_size, grid_size)) / (grid_size**2)  # Uniform belief
        self.false_positive_rate = 0.1
        self.false_negative_rate = 0.2

    def move(self, best_move=None):
        """Move based on MPC if available, else random move."""
        if best_move:
            self.position = np.clip(self.position + best_move, 0, self.grid_size-1)
        else:
            move = random.choice([(1,0), (-1,0), (0,1), (0,-1)])
            self.position = np.clip(self.position + move, 0, self.grid_size-1)

    def sense(self, env):
        """Update belief using Bayesian inference."""
        x, y = self.position
        target_x, target_y = env.position

        # Sensor model: Probability based on true target location
        if (x, y) == (target_x, target_y):
            likelihood = 1 - self.false_negative_rate  # True positive rate
        else:
            likelihood = self.false_positive_rate  # False positive rate

        # Update belief using Bayes’ Rule: P(T|O) ∝ P(O|T) * P(T)
        prior_belief = self.belief.copy()
        self.belief[x, y] = likelihood * prior_belief[x, y]
        self.belief /= np.sum(self.belief)  # Normalize to keep sum = 1

    def communicate(self, other_robots):
        """Share belief if uncertainty is high."""
        for other in other_robots:
            if np.linalg.norm(self.position - other.position) <= COMMUNICATION_RADIUS:
                if np.max(self.belief) < 0.7:  # Uncertainty threshold
                    self.belief = (self.belief + other.belief) / 2  # Merge beliefs
                    self.belief /= np.sum(self.belief)

# ---------------------- Model Predictive Control (MPC) ---------------------- #
def mpc_plan(robot, env, horizon=3):
    """Predicts the best move for the next n steps."""
    best_move = None
    best_expected_belief = -np.inf

    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up

    for move in moves:
        simulated_pos = np.clip(robot.position + move, 0, robot.grid_size-1)
        simulated_belief = robot.belief.copy()

        for _ in range(horizon):  # Simulate n steps ahead
            simulated_belief *= 0.9  # Assume decay in confidence
            simulated_pos = np.clip(simulated_pos + random.choice(moves), 0, robot.grid_size-1)

        expected_belief = np.sum(simulated_belief)

        if expected_belief > best_expected_belief:
            best_expected_belief = expected_belief
            best_move = move

    return best_move
def animate_trajectories(self):
        """
        Create an animation showing the target and agents moving on the grid
        step by step.
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Convert 1D states to 2D (row, col) for plotting
        target_coords = [divmod(pos, self.grid_size[1]) for pos in self.trajectories['target']]
        agent_coords = []
        for agent_path in self.trajectories['agents']:
            agent_coords.append([divmod(pos, self.grid_size[1]) for pos in agent_path])
        
        # We can plot the grid boundaries (optional)
        ax.set_xlim(-0.5, self.grid_size[1] - 0.5)
        ax.set_ylim(-0.5, self.grid_size[0] - 0.5)
        ax.set_xticks(range(self.grid_size[1]))
        ax.set_yticks(range(self.grid_size[0]))
        ax.grid(True)
        plt.gca().invert_yaxis()  # Flip so row 0 is at the top, optional

        # Scatter plots for the target and each agent
        target_scatter = ax.scatter([], [], c='blue', label='Target', s=80)
        agent_scatters = []
        colors = ['green', 'red', 'purple', 'orange']  # For up to 4 agents
        for i in range(self.n_agents):
            sc = ax.scatter([], [], c=colors[i % len(colors)], label=f'Agent {i+1}', s=80)
            agent_scatters.append(sc)

        ax.legend()

        def init():
            """Initialize the animation (empty frame)."""
            target_scatter.set_offsets([])
            for sc in agent_scatters:
                sc.set_offsets([])
            return [target_scatter] + agent_scatters
        
        def update(frame):
            """Update scatter plot for frame 'frame'."""
            # Update target scatter
            tx, ty = target_coords[frame]
            target_scatter.set_offsets([[ty, tx]])  # note: (col, row) => (x, y)
            
            # Update each agent’s scatter
            for i, sc in enumerate(agent_scatters):
                ax_, ay_ = agent_coords[i][frame]
                sc.set_offsets([[ay_, ax_]])
            return [target_scatter] + agent_scatters

        # Number of frames in the animation is the number of recorded steps
        num_frames = len(target_coords)
        
        anim = FuncAnimation(
            fig, update, frames=num_frames, init_func=init,
            interval=200, blit=True, repeat=False
        )
        
        plt.show()

# ---------------------- Simulation ---------------------- #
def run_simulation():
    env = Target(GRID_SIZE)
    robots = [Robot(GRID_SIZE, i) for i in range(NUM_ROBOTS)]
    target_found = False
    robot_positions = {robot.robot_id: [] for robot in robots}  # Track robot positions for trajectories

    plt.figure(figsize=(6,6))
    
    for step in range(MAX_STEPS):
        plt.clf()  # Clear the figure for the next frame
        plt.imshow(np.zeros((GRID_SIZE, GRID_SIZE)), cmap="gray")  # Background

        # Move target
        env.move()

        for robot in robots:
            best_move = mpc_plan(robot, env)
            robot.move(best_move)
            robot.sense(env)

            # Store the robot's position
            robot_positions[robot.robot_id].append(robot.position)

            # Plot robot's current position
            plt.scatter(robot.position[1], robot.position[0], c='blue', s=100, label="Robot" if robot.robot_id == 0 else "")

        # Draw target
        plt.scatter(env.position[1], env.position[0], c='red', s=100, marker="X", label="Target")

        # Robots communicate
        for robot in robots:
            robot.communicate(robots)

        # Check if any robot found the target
        for robot in robots:
            if np.argmax(robot.belief) == np.ravel_multi_index(env.position, (GRID_SIZE, GRID_SIZE)):
                target_found = True

        # Plot trajectories (connecting past positions)
        for robot_id, positions in robot_positions.items():
            robot_positions_array = np.array(positions)
            plt.plot(robot_positions_array[:, 1], robot_positions_array[:, 0], c='blue', alpha=0.6)

        plt.title(f"Step {step + 1}")
        plt.legend()
        plt.pause(0.5)

        if target_found:
            print(f"Target found at {env.position} in {step + 1} steps!")
            np.save("ground_truth_belief.npy", robot.belief)  # Save belief for later training
            break
    plt.show()
run_simulation()
