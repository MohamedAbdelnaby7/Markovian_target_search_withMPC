import numpy as np
import random

class SearchEnvironment:
    def __init__(self, grid_size=(50, 50), n_agents=3):
        self.grid_size = grid_size
        self.n_states = grid_size[0] * grid_size[1]  # Total states in the grid
        self.n_agents = n_agents

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

        # For multi-agents, we want the Cartesian product of each agent's neighbors
        # e.g. agent1 has neighbors [n1a, n1b], agent2 has neighbors [n2a, n2b].
        # We'll return [(n1a, n2a), (n1a, n2b), (n1b, n2a), (n1b, n2b)].
        return list(zip(*neighbors_list))


    def move_target(self):
        """Move the target randomly to any of its neighboring states."""
        rows, cols = self.grid_size
        row, col = divmod(self.true_position, cols)
        neighbors = []

        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbors.append(nr * cols + nc)

        if neighbors:
            self.true_position = random.choice(neighbors)
            self.trajectories['target'].append(self.true_position)

    def is_terminal(self, state):
        """Check if any agent is at the target position."""
        return any(agent_pos == self.true_position for agent_pos in state)

    def update_agents(self, new_positions):
        """Update agent positions after an MCTS step."""
        self.agent_positions = tuple(new_positions)
        for i, pos in enumerate(new_positions):
            self.trajectories['agents'][i].append(pos)