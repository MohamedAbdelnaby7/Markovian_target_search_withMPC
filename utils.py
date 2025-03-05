import matplotlib.pyplot as plt
import matplotlib.animation as animation

def visualize_trajectories(self, initial_belief):
        """Visualize agent and target trajectories with start/end beliefs"""
        plt.figure(figsize=(12, 6))
        
        # Plot initial belief
        plt.subplot(1, 2, 1)
        plt.imshow(initial_belief.reshape(self.grid_size), cmap='Reds')
        plt.title("Initial Belief")
        plt.colorbar()
        
        # Plot final belief
        plt.subplot(1, 2, 2)
        plt.imshow(self.belief.reshape(self.grid_size), cmap='Reds')
        plt.title("Final Belief")
        plt.colorbar()
        
        # Create trajectory plot
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        
        # Convert positions to coordinates
        target_path = [divmod(s, self.grid_size[1]) for s in self.trajectories['target']]
        agent_paths = [[divmod(s, self.grid_size[1]) for s in path] 
                      for path in self.trajectories['agents']]
        
        # Plot target trajectory
        t_x, t_y = zip(*target_path)
        plt.plot(t_y, t_x, 'b-', marker='o', label='Target Path')
        
        # Plot agent trajectories
        colors = ['green', 'purple', 'orange', 'red']
        for i, path in enumerate(agent_paths):
            a_x, a_y = zip(*path)
            plt.plot(a_y, a_x, colors[i], 
                    label=f'Agent {i+1} Path')
        
        plt.xticks(range(self.grid_size[1]))
        plt.yticks(range(self.grid_size[0]))
        plt.grid(True)
        plt.title("Agent and Target Trajectories")
        plt.legend()
        plt.show()

def visualize_trajectories_MCTS(grid_size, trajectories):
    """
    Visualize agent and target trajectories for MCTS (no belief involved).

    :param grid_size: (rows, cols) specifying the environment grid
    :param trajectories: a dict containing:
         {
            'target': [list of target positions over time],
            'agents': [list of lists of agent positions over time]
         }
    """

    rows, cols = grid_size

    # Create a figure/axis for plotting
    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    # Convert the target’s positions to (x, y) coordinates
    target_positions = trajectories['target']
    target_coords = [divmod(pos, cols) for pos in target_positions]  # (row, col)
    t_x, t_y = zip(*target_coords)  # separate row, col

    # Plot the target’s path in blue
    plt.plot(t_y, t_x, 'b-o', label='Target Path')

    # Convert each agent’s positions to (x, y) coords
    agent_positions_list = trajectories['agents']  # list of agent-specific lists
    colors = ['green', 'purple', 'orange', 'red', 'black']
    for i, agent_traj in enumerate(agent_positions_list):
        agent_coords = [divmod(pos, cols) for pos in agent_traj]
        a_x, a_y = zip(*agent_coords)
        color = colors[i % len(colors)]
        plt.plot(a_y, a_x, color=color, marker='o', label=f'Agent {i+1} Path')

    # Make the grid lines stand out a bit
    plt.xticks(range(cols))
    plt.yticks(range(rows))
    ax.invert_yaxis()  # so row 0 is at top if you prefer typical image-like orientation
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.title("MCTS Search: Agent and Target Trajectories")
    plt.legend()
    plt.show()

def animate_trajectories_MCTS(grid_size, trajectories, interval=500):
    """
    Animates agent and target trajectories for MCTS on a single figure.

    :param grid_size: tuple (rows, cols) defining the grid dimensions.
    :param trajectories: dict containing:
         {
            'target': [list of target positions over time],
            'agents': [list of lists of agent positions over time]
         }
    :param interval: time in milliseconds between frames.
    """
    rows, cols = grid_size

    # Convert the target positions from flat indices to (row, col) coordinates.
    target_positions = trajectories['target']
    target_coords = [divmod(pos, cols) for pos in target_positions]

    # Convert each agent's trajectory from flat indices to (row, col) coordinates.
    agents_coords = []
    for agent_traj in trajectories['agents']:
        coords = [divmod(pos, cols) for pos in agent_traj]
        agents_coords.append(coords)

    # Determine the total number of frames (we assume all trajectories have the same length).
    n_frames = len(target_coords)

    # Create a figure and axis.
    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame):
        ax.clear()
        # Set grid properties and axis limits.
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.invert_yaxis()  # so that row 0 is at the top
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)

        # Plot the target trajectory up to the current frame.
        if frame >= 0:
            current_target = target_coords[:frame + 1]
            if current_target:
                t_rows, t_cols = zip(*current_target)
                ax.plot(t_cols, t_rows, 'b-o', label='Target Path')

        # Plot each agent's trajectory up to the current frame.
        colors = ['green', 'purple', 'orange', 'red', 'black']
        for i, agent_coord in enumerate(agents_coords):
            # Ensure we only use data available up to this frame.
            current_traj = agent_coord[:frame + 1]
            if current_traj:
                a_rows, a_cols = zip(*current_traj)
                ax.plot(a_cols, a_rows, marker='o', color=colors[i % len(colors)],
                        label=f'Agent {i + 1} Path')

        ax.set_title(f"MCTS Search: Trajectories at Step {frame + 1}")
        ax.legend(loc='upper right')
        return ax,

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False, repeat=True)
    plt.show()