import numpy as np
import random
import math
from collections import defaultdict

class MCTSPlanner:
    def __init__(self, env, horizon=3, simulations=100, exploration_constant=1.41, 
                 discount_factor=0.9, gaussian_bias=False, heatmap_std_dev=5, heatmap_center=None):
        """
        env: the environment object, which must provide a method get_neighbors(state)
             that returns a list of possible joint actions given a joint state.
             We assume that the state is a tuple with agents' positions.
        horizon: rollout depth (number of steps to simulate)
        simulations: number of MCTS iterations to perform (controlled in __call__)
        exploration_constant: constant balancing exploration vs. exploitation in UCB
        discount_factor: factor to discount rewards during backpropagation (set to 0.9)
        gaussian_bias: flag to enable additional Gaussian-based biases (not used in this basic version)
        heatmap_std_dev: standard deviation used in the Gaussian reward function.
        """
        self.env = env
        self.horizon = horizon
        self.simulations = simulations
        self.exploration_constant = exploration_constant
        self.discount_factor = discount_factor
        self.gaussian_bias = gaussian_bias
        self.heatmap_std_dev = heatmap_std_dev
        self.heatmap_center = heatmap_center

        # MCTS Tree: Q-values and visit counts for (state, joint action) pairs.
        self.Q = defaultdict(float)
        self.N = defaultdict(int)

    def __call__(self, sensors, num_simulations=None):
        """
        Run the MCTS search.
        
        sensors: a tuple representing the state (agents positions).
                 For example, (agent1_position, agent2_position, ...).
        num_simulations: if provided, overrides the default simulation count.
        """
        if num_simulations is None:
            num_simulations = self.simulations

        # Ensure the state is a tuple.
        root = sorted(self._ensure_tuple(sensors))

        # Run MCTS for the specified number of simulations.
        for _ in range(num_simulations):
            self.mcts_plan(root)

        # Return the best joint action (agents and target move) from the root state.
        # If you prefer to control only the agents' moves externally,
        # you can extract the agents' moves from the joint action.
        return self.select_best_action(root)

    def mcts_plan(self, state):
        """
        Runs MCTS for planning and returns the best joint action (a tuple) for the multi-agent state.
        """
        current_state = self._ensure_tuple(state)
        path = []

        # Expand if needed
        if current_state not in self.Q:
            self.expand(current_state)

        # Selection & Expansion
        while (current_state in self.Q and len(self.env.get_neighbors(current_state)) > 0):
            actions = self.select_action(current_state)
            path.append((current_state, actions))
            current_state = tuple(actions)

        # Simulation (Rollout)
        reward = self.simulate(current_state)

        # Backpropagation
        self.backpropagate(path, reward)

    def select_best_action(self, state):
        """
        Returns the best joint action from the Q-values for the multi-agent 'state'.
        """
        state = self._ensure_tuple(state)
        possible_actions = self.env.get_neighbors(state)
        # Each possible action is also a tuple of new positions
        return max(possible_actions, key=lambda a: self.Q[(state, a)])

    def select_action(self, state):
        """Selects the best action for the given multi-agent state."""
        possible_moves = self.env.get_neighbors(state)

        def ucb_score(action):
            if self.N[(state, action)] == 0:
                return random.uniform(5, 6)
            exploitation = self.Q[(state, action)] / self.N[(state, action)]
            exploration = self.exploration_constant * math.sqrt(
                math.log(sum(self.N[(state, a)] for a in possible_moves) + 1) / self.N[(state, action)]
            )
            return exploitation + exploration

        return max(possible_moves, key=ucb_score)

    def expand(self, state):
        """Expands the MCTS tree by adding new possible single-agent or multi-agent actions."""
        possible_actions = self.env.get_neighbors(state)
        # For multi-agent 'state', possible_actions is e.g. [(posA1, posB1), (posA1, posB2), ...]
        for action in possible_actions:
            self.Q[(state, action)] = 0
            self.N[(state, action)] = 0

    def simulate(self, state):
        """
        Performs a rollout simulation from the given state over a horizon.
        Uses a random rollout policy by sampling a joint action from the environment's neighbors.
        The reward is computed as a constant reward (number of agents) plus a Gaussian reward for each agent.
        
        Here, we assume that the state is a multi-agent state (tuple of agent positions)
        and that self.env.get_neighbors(state) returns a list of possible joint moves.
        """
        total_reward = 0.0
        current_state = state

        for _ in range(self.horizon):
            # Get all possible joint moves from the current state.
            possible_joint_moves = self.env.get_neighbors(current_state)
            if not possible_joint_moves:
                break  # Terminal or no valid moves.
            step_reward = 0
            # Sample a joint move at random.
            current_state = random.choice(possible_joint_moves)
            if self.gaussian_bias:
                # Compute Gaussian reward for each agent.
                # Here we assume that get_gaussian_reward compares each agent's position
                # against a fixed target position (stored in self.heatmap_center) for the sanity check.
                gaussian_rewards = [self.get_gaussian_reward(agent_pos) for agent_pos in current_state]
                step_reward += sum(gaussian_rewards)
            else:
                # Compute the reward for this new state.
                # Assuming state = (agent1, agent2, ..., agentN)
                num_agents = len(current_state)
                # Use a constant reward per agent (as a simple sanity check) plus Gaussian rewards.
                step_reward = num_agents
            total_reward += step_reward
        return total_reward

    def backpropagate(self, path, reward):
        """Backpropagates the reward through the tree."""
        for state, actions in reversed(path):
            # Convert actions list to a tuple so we can index Q properly
            joint_action = tuple(actions)
            self.Q[(state, joint_action)] += reward
            self.N[(state, joint_action)] += 1
            reward *=.9

    def get_gaussian_reward(self, position):
        """
        Computes a Gaussian reward based on the Euclidean distance between the given state position and the heatmap center.
        
        - 'position' is a flattened index.
        - The grid size is (rows, cols), so we convert the index to (row, col).
        - 'self.heatmap_center' should be a tuple (center_row, center_col).
        - The reward is highest (1.0) when the agent is exactly at the heatmap center,
        and decays as the Euclidean distance increases.
        """
        if self.heatmap_center is None:
            return 0  # No Gaussian influence if no center is specified.
        
        rows, cols = self.env.grid_size
        row, col = divmod(position, cols)
        center_row, center_col = self.heatmap_center
        
        # Compute squared Euclidean distance.
        dist_squared = (row - center_row)**2 + (col - center_col)**2
        
        # Compute the Gaussian function; maximum value of 1 when dist_squared is 0.
        reward = math.exp(-dist_squared / (2 * self.heatmap_std_dev ** 2))
        return reward


    def _ensure_tuple(self, state):
        """
        Helper method to ensure 'state' is a tuple of agent positions.
        If it's int, we wrap it in a tuple.
        If it's a list, convert to tuple.
        """
        if isinstance(state, int):
            return (state,)
        elif isinstance(state, (list, np.ndarray)):
            return tuple(state)
        elif isinstance(state, tuple):
            return state
        else:
            raise ValueError(f"MCTSPlanner received unexpected state type: {type(state)}")