import numpy as np
import random
import itertools
import math

class SearchEnvironment:
    def __init__(self, grid_size=(50, 50), n_agents=3,
                 gaussian_bias=False, heatmap_std_dev=5, heatmap_center=None):
        self.grid_size = grid_size
        self.n_states = grid_size[0] * grid_size[1]  # Total states in the grid
        self.n_agents = n_agents
        self.gaussian_bias = gaussian_bias
        self.heatmap_std_dev = heatmap_std_dev
        self.heatmap_center = heatmap_center

        # Initialize agents at random positions
        self.agent_positions = tuple(np.random.choice(self.n_states, self.n_agents, replace=False))

        # Initialize target position randomly
        self.true_position = np.random.choice(self.n_states)

        # Store movement history
        self.trajectories = {
            'target': [self.true_position],
            'agents': [[pos] for pos in self.agent_positions]
        }

    def get_neighbors(self, state):
        """
        Returns valid moves for:
        1) A single-agent state (int)
        2) A multi-agent state (tuple of ints)

        For multi-agent, we return all possible joint moves (Cartesian product).
        For single-agent, we just return a list of neighbors for that agent.
        """
        rows, cols = self.grid_size

        # Case 1: single-agent state (int)
        if isinstance(state, (int, np.integer)):
        # This covers native int and any numpy int type
            row, col = divmod(state, cols)
            neighbors = []
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbors.append(nr * cols + nc)
            return neighbors

        # Case 2: multi-agent state (tuple of ints)
        # e.g., state = (posA, posB, ...)
        neighbors_list = []
        for agent_pos in state:
            row, col = divmod(agent_pos, cols)
            single_neighbors = []
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    single_neighbors.append(nr * cols + nc)
            neighbors_list.append(single_neighbors)

        # Compute the full Cartesian product of moves from each agent.
        # This returns all combinations like:
        # [(agent1_move1, agent2_move1), (agent1_move1, agent2_move2), ...]
        joint_moves = list(itertools.product(*neighbors_list))
        random.shuffle(joint_moves)  # Shuffle to randomize ordering
        return joint_moves


    def move_target(self):
        """
            Moves the target to one of its neighboring states.
            If gaussian_bias is enabled and a heatmap center is provided (as a tuple of (row, col)),
            the target will move toward that center (i.e. choose the neighbor with the smallest Euclidean distance to the center).
            Otherwise, the target moves randomly.
        """
        neighbors = self.get_neighbors(self.true_position)

        if not neighbors:
            return # No valid Movies

        # If gaussian_bias is True and a heatmap center is provided,
        # choose the neighbor that minimizes the Euclidean distance to the heatmap center.
        if self.gaussian_bias and self.heatmap_center is not None:
            # self.heatmap_center should be a tuple (target_row, target_col)
            target_center = self.heatmap_center
            
            def euclidean_distance(state_index):
                r, c = divmod(state_index, self.grid_size[1])
                return math.sqrt((r - target_center[0])**2 + (c - target_center[1])**2)
            
            best_neighbor = min(neighbors, key=euclidean_distance)
            self.true_position = best_neighbor
        else:
            self.true_position = random.choice(neighbors)
    
        # Record the trajectory (assuming self.trajectories is a dict that tracks positions over time).
        self.trajectories['target'].append(self.true_position)

    def is_terminal(self, state):
        """Check if any agent is at the target position."""
        return any(agent_pos == self.true_position for agent_pos in state)

    def update_agents(self, new_positions):
        """Update agent positions after an MCTS step."""
        self.agent_positions = tuple(new_positions)
        for i, pos in enumerate(new_positions):
            self.trajectories['agents'][i].append(pos)