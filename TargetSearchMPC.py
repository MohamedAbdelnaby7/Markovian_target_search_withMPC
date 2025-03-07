import itertools
import numpy as np
from scipy.stats import entropy
from utils import visualize_trajectories

class TargetSearchMPC:
    def __init__(self, env, alpha=0.1, beta=0.1, horizon=2, fast_mode=False):
        self.env = env
        self.alpha = alpha  # False alarm probability
        self.beta = beta    # Missed detection probability
        self.horizon = horizon
        self.fast_mode = fast_mode #flag for optimized execution
        
        # Initialize transition matrix (Markov chain)
        self.transition_matrix = self.create_transition_matrix()
        
        # Initialize uniform belief
        self.belief = np.ones(self.env.n_states) / self.env.n_states
    
        self.correct_detections = 0  # Track successful detections
        self.map_estimates = []  # Store MAP estimates

    def create_transition_matrix(self):
        """Create a simple transition matrix where target moves to adjacent cells"""
        matrix = np.zeros((self.env.n_states, self.env.n_states))
        for i in range(self.env.n_states):
            neighbors = self.env.get_neighbors(i)
            p_stay = 0.2  # Probability of staying in current cell
            p_move = (1 - p_stay) / len(neighbors) if neighbors else 0
            matrix[i, i] = p_stay
            for n in neighbors:
                matrix[i, n] = p_move
        return matrix

    def get_neighbors(self, state):
        """Get adjacent states (including diagonal)"""
        row, col = divmod(state, self.env.grid_size[1])
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.env.grid_size[0] and 0 <= nc < self.env.grid_size[1]:
                    neighbors.append(nr * self.env.grid_size[1] + nc)
        return neighbors

    def update_belief(self, sensors, observations):
        """Update belief based on sensor readings"""
        # Predict step (Markov transition)
        predicted_belief = self.transition_matrix.T @ self.belief
        
        # Update step (Bayesian update)
        likelihood = np.ones(self.env.n_states)
        for s, obs in zip(sensors, observations):
            if obs == 1:
                likelihood[s] *= (1 - self.beta)  # True positive
                likelihood[:s] *= self.alpha       # False alarm
                likelihood[s+1:] *= self.alpha
            else:
                likelihood[s] *= self.beta         # Missed detection
                likelihood[:s] *= (1 - self.alpha)  # True negative
                likelihood[s+1:] *= (1 - self.alpha)
        
        updated_belief = predicted_belief * likelihood
        return updated_belief / updated_belief.sum()

    def simulate_observation(self, sensors, horizon):
        """Generate simulated sensor observations"""
        """Vectorized Sensor Observations (Avoids Loop-Based Sampling)"""
        predicted_target_pos = self.env.true_position  # Start from the current position
        observations_over_horizon = []  # Store observations at each time step

        if horizon == 0:
            return np.where(np.array(sensors) == self.env.true_position,
                np.random.choice([0, 1], p=[self.beta, 1 - self.beta], size=len(sensors)),
                np.random.choice([0, 1], p=[1 - self.alpha, self.alpha], size=len(sensors))
                ).tolist()
        for _ in range(horizon):  # Simulate for each future step
            predicted_target_pos = np.random.choice(self.env.n_states, p=self.transition_matrix[predicted_target_pos])
            observations_over_horizon.append(
                np.where(np.array(sensors) == predicted_target_pos,
                np.random.choice([0, 1], p=[self.beta, 1 - self.beta], size=len(sensors)),
                np.random.choice([0, 1], p=[1 - self.alpha, self.alpha], size=len(sensors))
                ).tolist()
            )
        return observations_over_horizon
    
    def objective_function(self, objective_type, belief):
        """Dynamically execute the selected objective function."""
        objective_functions = {
            "entropy": lambda b: -entropy(b),
            "greedy_map": lambda b: np.argsort(b)[-self.env.n_agents:][::-1],
            "variance_reducing": lambda b: np.argsort(b)[- (self.env.n_agents + 1):][::-1][1:]
        }
        return objective_functions.get(objective_type, lambda _: None)(belief)
    
    def mpc_plan(self, objective_type):
        """Uses either brute-force full search or fast greedy strategy based on `fast_mode`."""
        if self.fast_mode:
            # Greedy movement selection
            best_action = []
            for agent_pos in self.env.agent_positions:
                neighbors = self.env.get_neighbors(agent_pos) + [agent_pos]
                best_move = max(neighbors, key=lambda n: self.belief[n])
                best_action.append(best_move)
            return best_action
        else:
            """MPC planning with receding horizon"""
            best_value = -np.inf
            best_action = None
            
            # Generate reachable positions for each agent
            candidate_actions = []
            for agent_pos in self.env.agent_positions:
                neighbors = self.env.get_neighbors(agent_pos)
                candidate_actions.append(neighbors + [agent_pos])
            
            # Evaluate all possible combinations of agent movements
            for action_seq in itertools.product(*candidate_actions):
                current_belief = self.belief.copy()
                total_value = 0
                
                # Get future observations over the horizon
                obs_horizon = self.simulate_observation(action_seq, self.horizon)

                for t in range(self.horizon):
                    current_belief = self.update_belief(action_seq, obs_horizon[t])
                    total_value += self.objective_function(objective_type, current_belief)
                    if isinstance(total_value, np.ndarray):  # Handle array case when using greedy_map
                        total_value = np.max(total_value)  # Convert array to a single value
                
                if total_value > best_value:
                    best_value = total_value
                    best_action = action_seq
        return best_action
    
    def run_simulation(self, steps=20, detection_threshold=0.8, objective_type = "greedy_map"):
        initial_belief = self.belief.copy()
        
        for step in range(steps):
            print(f"step {step}")
            # Plan and move agents
            new_positions = self.mpc_plan(objective_type)
            self.env.agent_positions = list(new_positions)
            
            # Update target position (Markov transition)
            self.env.true_position = np.random.choice(
                self.env.n_states, p=self.transition_matrix[self.env.true_position]
            )

            # Record trajectories
            self.env.trajectories['target'].append(self.env.true_position)
            for i, pos in enumerate(new_positions):
                self.env.trajectories['agents'][i].append(pos)
            
            # Get sensor observations
            obs = self.simulate_observation(new_positions, 0)

            # Update belief
            self.belief = self.update_belief(new_positions, obs)

            # --- Stopping Criterion ---
            # Check each agent’s position: if sensor=1 and posterior>threshold
            # in that cell => "found" the target
            for (agent_idx, agent_cell) in enumerate(new_positions):
                if obs[agent_idx] == 1 and self.belief[agent_cell] > detection_threshold:
                    print(f"Target detected by Agent {agent_idx} at step {step}, cell {divmod(self.env.true_position, 50)}.")
                    # Optionally store final results
                    self.visualize_trajectories(initial_belief)
                    return self.env.trajectories
        
        # If we exit the loop without detection, just visualize and return
        print("Target not detected within the given steps.")
        self.visualize_trajectories(initial_belief)
        return self.env.trajectories