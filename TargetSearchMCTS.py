import numpy as np
import random
import math
from collections import defaultdict
from utils import visualize_trajectories

class MCTSPlanner:
    def __init__(self, env, horizon=3, simulations=100, exploration_constant=1.41):
        self.env = env
        self.horizon = horizon  # Number of steps to simulate
        self.simulations = simulations  # Number of MCTS iterations
        self.exploration_constant = exploration_constant  # Exploration-exploitation balance

        # MCTS Tree Structure
        self.Q = defaultdict(float)  # Q-values for (state, action) pairs
        self.N = defaultdict(int)    # Visit counts for (state, action) pairs

    def __call__(self, sensors, num_simulations=None):
        """
        Allows calling MCTSPlanner as a function to run the MCTS search.
        'sensors' should be a tuple of agent positions, e.g. (42, 75, 90) for 3 agents.
        """
        if num_simulations is None:
            num_simulations = self.simulations

        # Convert 'sensors' to a multi-agent state (tuple of ints).
        root = self._ensure_tuple(sensors)

        # Run MCTS for the specified number of simulations
        for _ in range(num_simulations):
            self.mcts_plan(root)

        # Return the best action for all agents after all simulations
        return self.select_best_action(root)

    def mcts_plan(self, state):
        """
        Runs MCTS for planning and returns the best joint action (a tuple) for the multi-agent state.
        """
        # Make sure state is a tuple of agent positions
        root = self._ensure_tuple(state)
        root = tuple(sorted(root))

        for _ in range(self.simulations):
            current_state = root
            path = []

            # Selection & Expansion
            while (all(pos in self.Q for pos in current_state)
                   and len(self.env.get_neighbors(current_state)) > 0):
                # 'select_action' here is called on each agent's position individually
                actions = [self.select_action(pos) for pos in current_state]
                path.append((current_state, actions))
                current_state = self.get_next_state(current_state, actions)

            # Expand if needed
            if any(pos not in self.Q for pos in current_state):
                self.expand(current_state)

            # Simulation (Rollout)
            reward = self.simulate(current_state)

            # Backpropagation
            self.backpropagate(path, reward)

        # Now pick the best joint action from the root state
        possible_actions = self.env.get_neighbors(root)
        best_joint_action = max(possible_actions, key=lambda a: sum(self.Q[(root, a)]))
        return best_joint_action

    def select_best_action(self, state):
        """
        Returns the best joint action from the Q-values for the multi-agent 'state'.
        """
        root = self._ensure_tuple(state)
        root = tuple(sorted(root))
        possible_actions = self.env.get_neighbors(root)

        # Each possible action is also a tuple of new positions
        return max(possible_actions, key=lambda a: sum(self.Q[(root, a)]))

    def select_action(self, single_agent_pos):
        """
        Selects the best next move for a single agent's position.
        We treat single_agent_pos as an int.
        """
        # We call get_neighbors(int) → returns a list of single-agent neighbor positions
        possible_moves = self.env.get_neighbors(single_agent_pos)

        total_visits = sum(self.N[((single_agent_pos,), move)] for move in possible_moves)

        def ucb_score(move):
            # For Q/N, we store the state as a tuple of length 1, e.g. (single_agent_pos,)
            if self.N[((single_agent_pos,), move)] == 0:
                return float('inf')
            exploitation = self.Q[((single_agent_pos,), move)] / self.N[((single_agent_pos,), move)]
            exploration = self.exploration_constant * math.sqrt(
                math.log(total_visits + 1) / self.N[((single_agent_pos,), move)]
            )
            return exploitation + exploration

        return max(possible_moves, key=ucb_score)

    def get_next_state(self, old_state, actions):
        """
        Moves each agent to the chosen next position from 'actions'.
        'old_state': tuple of agent positions
        'actions': a list of the chosen next single-agent positions
        """
        # actions has the same length as old_state
        new_positions = []
        for next_pos in actions:
            # next_pos is an int representing the single agent's next cell
            new_positions.append(next_pos)
        return tuple(new_positions)

    def expand(self, state):
        """Expands the MCTS tree by adding new possible single-agent or multi-agent actions."""
        possible_actions = self.env.get_neighbors(state)
        # For multi-agent 'state', possible_actions is e.g. [(posA1, posB1), (posA1, posB2), ...]
        for action in possible_actions:
            self.Q[(state, action)] = 0
            self.N[(state, action)] = 0

    def simulate(self, state):
        """Runs a random rollout and returns an estimated reward."""
        current_state = state
        total_reward = 0

        for _ in range(self.horizon):
            # For each agent in the multi-agent state, pick a random neighbor
            next_positions = []
            for agent_pos in current_state:
                # single-agent call
                possible_moves = self.env.get_neighbors(agent_pos)
                next_positions.append(random.choice(possible_moves))

            current_state = tuple(next_positions)
            # Simple reward: number of agents
            total_reward += len(current_state)

        return total_reward

    def backpropagate(self, path, reward):
        """Backpropagates the reward through the tree."""
        for state, actions in reversed(path):
            # Convert actions list to a tuple so we can index Q properly
            joint_action = tuple(actions)
            self.Q[(state, joint_action)] += reward
            self.N[(state, joint_action)] += 1
            reward *= 0.9  # Discount factor

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