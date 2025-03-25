import numpy as np
import random
import itertools
import math

class SearchEnvironment:
    def __init__(self, grid_size=(50, 50), n_agents=3,
                 gaussian_bias=False, heatmap_std_dev=5, heatmap_center=None,
                 initial_agent_positions=None, initial_target_position=None, target_mdp = False, 
                 precomputed_target_trajectory=None):
        self.grid_size = grid_size
        self.n_states = grid_size[0] * grid_size[1]  # Total states in the grid
        self.n_agents = n_agents
        self.gaussian_bias = gaussian_bias
        self.heatmap_std_dev = heatmap_std_dev
        self.heatmap_center = heatmap_center
        self.target_mdp = target_mdp
        # If None, we'll do Markov transitions; otherwise, we use the precomputed list
        self.precomputed_target_trajectory = precomputed_target_trajectory

        if self.target_mdp:
            # Initialize transition matrix (Markov chain)
            self.transition_matrix = self.create_transition_matrix()

        # Initialize agents at given positions or randomly
        if initial_agent_positions is not None:
            self.agent_positions = tuple(initial_agent_positions)
        else:
            self.agent_positions = tuple(np.random.choice(self.n_states, self.n_agents, replace=False))

        # Initialize target at given position or randomly
        if initial_target_position is not None:
            self.true_position = initial_target_position
        else:
            self.true_position = np.random.choice(self.n_states)

        # Store movement history
        self.trajectories = {
            'target': [self.true_position],
            'agents': [[pos] for pos in self.agent_positions]
        }

    def create_transition_matrix(self):
        """Create a simple transition matrix where target moves to adjacent cells"""
        matrix = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            neighbors = self.get_neighbors(i)
            p_stay = 0.2  # Probability of staying in current cell
            p_move = (1 - p_stay) / len(neighbors) if neighbors else 0
            matrix[i, i] = p_stay
            for n in neighbors:
                matrix[i, n] = p_move
        return matrix
    
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


    def move_target(self, current_step=None):
        """
            If precomputed_target_trajectory is available,
            return the position for this step.
            Else, Moves the target to one of its neighboring states.
            If gaussian_bias is enabled and a heatmap center is provided (as a tuple of (row, col)),
            the target will move toward that center (i.e. choose the neighbor with the smallest Euclidean distance to the center).
            Otherwise, the target moves randomly.
        """
        neighbors = self.get_neighbors(self.true_position)

        if not neighbors:
            return # No valid Movies
        if self.precomputed_target_trajectory is not None:
            return self.precomputed_target_trajectory[current_step]
        else:
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
            elif self.target_mdp:
                # Update target position (Markov transition)
                self.true_position = np.random.choice(
                    self.n_states, p=self.transition_matrix[self.true_position]
                )
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

class SearchEnvironmentWithBeliefs(SearchEnvironment):
    def __init__(self, grid_size=(50, 50), n_agents=3, initial_agent_positions=None, proximity_distance=3):
        super().__init__(grid_size, n_agents, initial_agent_positions=initial_agent_positions)
        self.proximity_distance = proximity_distance
        # Initialize agent beliefs (uniform for now)
        self.agent_beliefs = [np.ones(self.n_states) / self.n_states for _ in range(self.n_agents)]

    def get_agent_beliefs(self):
        return self.agent_beliefs

    def merge_beliefs(self, agents_in_range):
        """ Merge the beliefs of the agents within proximity """
        beliefs = [self.agent_beliefs[agent_id] for agent_id in agents_in_range]
        merged_belief = self.merge_beliefs_using_kl(beliefs, agent_weights=np.ones(len(beliefs)))  # Equal weight for now
        return merged_belief

    def merge_beliefs_using_kl(self, beliefs, agent_weights):
        """ Perform belief merging using KL Divergence """
        num_states = beliefs[0].shape[0]
        
        def kl_divergence(p, q):
            p = np.clip(p, 1e-10, 1)
            q = np.clip(q, 1e-10, 1)
            return np.sum(p * np.log(p / q))

        def objective(merged_belief):
            merged_belief = np.clip(merged_belief, 1e-10, 1)
            merged_belief /= np.sum(merged_belief)
            divergence = 0
            for belief, weight in zip(beliefs, agent_weights):
                divergence += weight * kl_divergence(belief, merged_belief)
            return divergence

        initial_guess = np.mean(beliefs, axis=0)
        constraints = ({'type': 'eq', 'fun': lambda b: np.sum(b) - 1})
        bounds = [(0, 1) for _ in range(num_states)]

        result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints, method='SLSQP')

        if result.success:
            return result.x / np.sum(result.x)
        else:
            raise ValueError("Optimization failed: " + result.message)

    def check_proximity(self):
        # Check for proximity-based communication and merge beliefs
        for agent_id in range(self.n_agents):
            agents_in_range = [
                other_agent_id for other_agent_id in range(self.n_agents)
                if self.distance(self.agent_positions[agent_id], self.agent_positions[other_agent_id]) <= self.proximity_distance
            ]
            if len(agents_in_range) > 1:
                merged_belief = self.merge_beliefs(agents_in_range)
                for agent_id in agents_in_range:
                    self.agent_beliefs[agent_id] = merged_belief