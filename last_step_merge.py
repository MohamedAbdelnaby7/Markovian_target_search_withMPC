# improved_last_step_merge_with_comparisons.py

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as patches
import itertools
import time
from scipy.optimize import minimize
from Environment import SearchEnvironment

class RegionRestrictedAgent:
    def __init__(self, agent_id, region_states, grid_size, initial_position=None, transition_matrix=None):
        """
        Agent restricted to a specific region of the grid
        
        Parameters:
        -----------
        agent_id : int
            Unique identifier for this agent
        region_states : list
            List of state indices that belong to this agent's region
        grid_size : tuple
            Size of the grid (rows, cols)
        initial_position : int, optional
            Initial position of the agent (must be in region_states)
        transition_matrix : ndarray, optional
            Transition matrix for target movement prediction
        """
        self.agent_id = agent_id
        self.region_states = set(region_states)  # Convert to set for faster lookups
        self.grid_size = grid_size
        self.total_states = grid_size[0] * grid_size[1]
        self.transition_matrix = transition_matrix
        
        # Initialize position randomly within region if not specified
        if initial_position is None or initial_position not in self.region_states:
            self.position = np.random.choice(list(self.region_states))
        else:
            self.position = initial_position
            
        # Initialize belief (uniform across the agent's region only)
        self.belief = np.zeros(self.total_states)
        region_prob = 1.0 / len(self.region_states)
        for state in self.region_states:
            self.belief[state] = region_prob
            
        # Initialize trajectory
        self.trajectory = [self.position]
        
    def get_neighbors(self):
        """Get valid neighboring states within agent's region"""
        rows, cols = self.grid_size
        row, col = divmod(self.position, cols)
        neighbors = []
        
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                new_pos = nr * cols + nc
                if new_pos in self.region_states:
                    neighbors.append(new_pos)
                    
        # Always include current position as an option
        if self.position not in neighbors:
            neighbors.append(self.position)
            
        return neighbors
    
    def update_position(self, new_position):
        """Update agent position and trajectory"""
        if new_position in self.region_states:
            self.position = new_position
            self.trajectory.append(new_position)
        else:
            raise ValueError(f"Position {new_position} is outside agent {self.agent_id}'s region")
            
    def update_belief(self, observation, alpha=0.1, beta=0.2):
        """
        Update belief based on observation
        
        Parameters:
        -----------
        observation : int
            1 if detection, 0 if no detection
        alpha : float
            False alarm probability
        beta : float
            Missed detection probability
        """
        # Predict step - propagate belief using target motion model
        predicted_belief = np.zeros_like(self.belief)
        
        if self.transition_matrix is not None:
            # Use provided transition matrix
            predicted_belief = self.transition_matrix.T @ self.belief
        else:
            # Simple local movement model
            rows, cols = self.grid_size
            for state in range(self.total_states):
                if self.belief[state] > 0:
                    # Find valid neighbors
                    r, c = divmod(state, cols)
                    neighbors = []
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (0,0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            neighbors.append(nr * cols + nc)
                    
                    # Distribute probability
                    prob = self.belief[state] / len(neighbors)
                    for neighbor in neighbors:
                        predicted_belief[neighbor] += prob
                    
        # Update step - apply Bayes rule
        likelihood = np.ones(self.total_states)
        
        if observation == 1:  # Detection
            # Likelihood is (1-beta) at current position, alpha elsewhere
            likelihood[:] = alpha
            likelihood[self.position] = 1 - beta
        else:  # No detection
            # Likelihood is beta at current position, (1-alpha) elsewhere
            likelihood[:] = 1 - alpha
            likelihood[self.position] = beta
            
        # Apply Bayes rule
        updated_belief = predicted_belief * likelihood
        
        # Constrain belief to agent's region
        for state in range(self.total_states):
            if state not in self.region_states:
                updated_belief[state] = 0
                
        # Normalize
        if np.sum(updated_belief) > 0:
            self.belief = updated_belief / np.sum(updated_belief)
        
        # Ensure belief is strictly in the agent's region
        for state in range(self.total_states):
            if state not in self.region_states:
                self.belief[state] = 0


class LastStepMergeExperiment:
    def __init__(self, grid_size=(20, 20), n_agents=4, segmentation_type='equal', steps=3000, 
                 alpha=0.1, beta=0.2, target_mdp=True, custom_regions = None):
        """
        Experiment to compare individual search with final belief merging
        
        Parameters:
        -----------
        grid_size : tuple
            Size of the grid (rows, cols)
        n_agents : int
            Number of agents
        segmentation_type : str
            Type of segmentation ('equal', 'quadrants', or 'custom')
        steps : int
            Number of simulation steps
        alpha : float
            False alarm probability
        beta : float
            Missed detection probability
        target_mdp : bool
            Whether to use MDP for target movement
        """
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.steps = steps
        self.alpha = alpha
        self.beta = beta
        self.target_mdp = target_mdp
        
        # Create reference environment for target dynamics
        self.env = SearchEnvironment(grid_size=grid_size, n_agents=1, target_mdp=target_mdp)
        
        if custom_regions:
            self.segments = custom_regions
        else:
            self.segments = self._create_segments(segmentation_type)
        
        # Generate target trajectory
        self._generate_target_trajectory()
        
        # Create agents
        self.agents = []
        for i in range(min(n_agents, len(self.segments))):
            # Create a transition matrix for this agent's belief
            trans_matrix = self.env.transition_matrix if hasattr(self.env, 'transition_matrix') else None
            agent = RegionRestrictedAgent(i, self.segments[i], grid_size, transition_matrix=trans_matrix)
            self.agents.append(agent)
    
    def _create_segments(self, segmentation_type):
        """Create segments based on the specified type"""
        rows, cols = self.grid_size
        segments = []
        
        if segmentation_type == 'equal':
            # Create roughly equal-sized horizontal segments
            segment_height = rows // self.n_agents
            remain = rows % self.n_agents
            
            start_row = 0
            for i in range(self.n_agents):
                # Add an extra row to some segments if division isn't even
                height = segment_height + (1 if i < remain else 0)
                segment = []
                
                for r in range(start_row, start_row + height):
                    for c in range(cols):
                        segment.append(r * cols + c)
                
                segments.append(segment)
                start_row += height
        
        elif segmentation_type == 'quadrants':
            # Divide the grid into 4 quadrants
            mid_row = rows // 2
            mid_col = cols // 2
            
            # Top-left quadrant
            quadrant1 = [r * cols + c for r in range(mid_row) for c in range(mid_col)]
            # Top-right quadrant
            quadrant2 = [r * cols + c for r in range(mid_row) for c in range(mid_col, cols)]
            # Bottom-left quadrant
            quadrant3 = [r * cols + c for r in range(mid_row, rows) for c in range(mid_col)]
            # Bottom-right quadrant
            quadrant4 = [r * cols + c for r in range(mid_row, rows) for c in range(mid_col, cols)]
            
            segments = [quadrant1, quadrant2, quadrant3, quadrant4]
            
            # If we have fewer than 4 agents, combine some segments
            while len(segments) > self.n_agents:
                segments[0].extend(segments.pop())
        
        else:  # 'custom' or unrecognized type - create a default segmentation
            segment_size = (self.grid_size[0] * self.grid_size[1]) // self.n_agents
            for i in range(self.n_agents):
                start = i * segment_size
                end = (i + 1) * segment_size if i < self.n_agents - 1 else self.grid_size[0] * self.grid_size[1]
                segments.append(list(range(start, end)))
        
        return segments
    
    def _generate_target_trajectory(self):
        """Generate a target trajectory for the entire simulation"""
        rows, cols = self.grid_size
        
        # Random starting position
        target_pos = np.random.randint(0, rows * cols)
        trajectory = [target_pos]
        
        # Use environment's move_target method for consistency
        self.env.true_position = target_pos
        
        for step in range(self.steps):
            self.env.move_target()  # This will update env.true_position
            trajectory.append(self.env.true_position)
        
        self.target_trajectory = trajectory
    
    def run_simulation(self):
        """Run the search simulation with region-restricted agents"""
        
        for step in range(self.steps):
            if step % 500 == 0:
                print(f"Simulation step {step}/{self.steps}")
                
            target_pos = self.target_trajectory[step]
            
            # Each agent makes an observation and updates its belief
            for agent in self.agents:
                # Generate observation based on agent position and target position
                if agent.position == target_pos:
                    # Target is at agent's position
                    observation = 1 if random.random() > self.beta else 0
                else:
                    # Target is not at agent's position
                    observation = 1 if random.random() < self.alpha else 0
                
                # Update belief based on observation
                agent.update_belief(observation, self.alpha, self.beta)
                
                # Simple greedy policy: move to highest probability neighbor
                neighbors = agent.get_neighbors()
                
                # Add small random noise to break ties and encourage exploration
                belief_values = np.array([agent.belief[pos] for pos in neighbors])
                noise = np.random.uniform(0, 0.01, size=len(neighbors))
                belief_values += noise
                
                best_idx = np.argmax(belief_values)
                best_pos = neighbors[best_idx]
                agent.update_position(best_pos)
    
    def merge_beliefs_kl(self):
        """Merge agent beliefs at end of simulation using KL divergence minimization"""
        # Extract agent beliefs
        all_beliefs = np.array([agent.belief for agent in self.agents])
        
        # Define KL divergence function
        def kl_divergence(p, q):
            # Add small epsilon to avoid division by zero
            p = np.clip(p, 1e-10, 1)
            q = np.clip(q, 1e-10, 1)
            return np.sum(p * np.log(p / q))
        
        # Define objective function for minimization
        def objective(merged_belief_flat):
            merged_belief = merged_belief_flat.reshape(all_beliefs[0].shape)
            
            # Normalize to ensure it's a valid probability distribution
            merged_belief = merged_belief / np.sum(merged_belief)
            
            # Calculate total KL divergence from all agents
            total_divergence = 0
            for agent_belief in all_beliefs:
                total_divergence += kl_divergence(agent_belief, merged_belief)
                
            return total_divergence
        
        # Constraints: probabilities must sum to 1
        def constraint(x):
            return np.sum(x) - 1
        
        # Initial guess: average of all beliefs
        initial_guess = np.mean(all_beliefs, axis=0)
        
        # Normalize initial guess
        initial_guess = initial_guess / np.sum(initial_guess)
        
        # Setup bounds and constraints
        bounds = [(0, 1) for _ in range(len(initial_guess))]
        constraints = {'type': 'eq', 'fun': constraint}
        
        # Perform optimization
        print("Optimizing KL divergence for belief merging...")
        result = minimize(
            objective, 
            initial_guess.flatten(), 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100}
        )
        
        if result.success:
            merged_belief = result.x.reshape(initial_guess.shape)
            return merged_belief / np.sum(merged_belief)
        else:
            print("Optimization failed:", result.message)
            return initial_guess / np.sum(initial_guess)
    
    def merge_beliefs_average(self):
        """Merge agent beliefs using simple averaging"""
        all_beliefs = np.array([agent.belief for agent in self.agents])
        merged_belief = np.mean(all_beliefs, axis=0)
        return merged_belief / np.sum(merged_belief)
    
    def merge_beliefs_consensus(self, iterations=50):
        """
        Merge agent beliefs using consensus approach
        
        Parameters:
        -----------
        iterations : int
            Number of consensus iterations
        """
        # Get initial beliefs
        n_agents = len(self.agents)
        all_beliefs = [agent.belief.copy() for agent in self.agents]
        n_states = len(all_beliefs[0])
        
        # Define adjacency matrix (all agents connected with equal weight)
        adjacency = np.ones((n_agents, n_agents)) - np.eye(n_agents)
        
        # Normalize weights
        weights = adjacency / np.sum(adjacency, axis=1, keepdims=True)
        
        # Run consensus iterations
        for _ in range(iterations):
            new_beliefs = []
            for i in range(n_agents):
                # Weighted average of neighboring beliefs
                belief_i = np.zeros(n_states)
                for j in range(n_agents):
                    if weights[i, j] > 0:
                        belief_i += weights[i, j] * all_beliefs[j]
                
                # Normalize
                if np.sum(belief_i) > 0:
                    belief_i = belief_i / np.sum(belief_i)
                    
                new_beliefs.append(belief_i)
            
            all_beliefs = new_beliefs
        
        # Final belief is the average of all converged beliefs
        merged_belief = np.mean(all_beliefs, axis=0)
        return merged_belief / np.sum(merged_belief)
    
    def visualize_results(self, merged_beliefs=None):
        """
        Visualize results with individual agent beliefs, merged belief, and trajectories
        
        Parameters:
        -----------
        merged_beliefs : dict, optional
            Dictionary of merged beliefs using different methods
        """
        if merged_beliefs is None:
            # Generate merged beliefs with different methods
            merged_beliefs = {
                "KL Divergence": self.merge_beliefs_kl(),
                "Simple Average": self.merge_beliefs_average(),
                "Consensus": self.merge_beliefs_consensus()
            }
        
        rows, cols = self.grid_size
        
        # Create a figure for agent beliefs
        n_agents = len(self.agents)
        fig_agents = plt.figure(figsize=(15, 5 * ((n_agents+1) // 2)))
        
        # Plot individual agent beliefs
        for i, agent in enumerate(self.agents):
            ax = fig_agents.add_subplot((n_agents+1) // 2, 2, i+1)
            belief_grid = agent.belief.reshape(rows, cols)
            im = ax.imshow(belief_grid, cmap='hot', interpolation='nearest', vmin=0, vmax=np.max(belief_grid))
            
            # Convert region states set to list for processing
            region_states_list = list(agent.region_states)
            
            # Add segment boundary
            min_row = min([r for r, _ in [divmod(s, cols) for s in region_states_list]])
            max_row = max([r for r, _ in [divmod(s, cols) for s in region_states_list]])
            min_col = min([c for _, c in [divmod(s, cols) for s in region_states_list]])
            max_col = max([c for _, c in [divmod(s, cols) for s in region_states_list]])
            
            rect = patches.Rectangle((min_col-0.5, min_row-0.5), max_col-min_col+1, max_row-min_row+1, 
                                     linewidth=2, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)
            
            # Plot agent trajectory
            agent_traj = [divmod(pos, cols) for pos in agent.trajectory]
            agent_rows, agent_cols = zip(*agent_traj)
            ax.plot(agent_cols, agent_rows, 'b-', linewidth=1.5, alpha=0.7)
            
            # Mark final agent position
            final_row, final_col = divmod(agent.position, cols)
            ax.plot(final_col, final_row, 'bo', markersize=8)
            
            ax.set_title(f'Agent {i} Belief')
            fig_agents.colorbar(im, ax=ax)
        
        # Create a figure for target trajectory
        fig_target = plt.figure(figsize=(10, 8))
        ax_target = fig_target.add_subplot(111)
        
        target_traj = [divmod(pos, cols) for pos in self.target_trajectory]
        target_rows, target_cols = zip(*target_traj)
        
        # Plot the full trajectory line
        ax_target.plot(target_cols, target_rows, 'r-', linewidth=1.5, alpha=0.7)
        
        # Add direction arrows
        arrow_indices = np.linspace(0, len(target_traj)-1, min(20, len(target_traj))).astype(int)
        for i in arrow_indices:
            if i < len(target_traj) - 1:
                dx = target_cols[i+1] - target_cols[i]
                dy = target_rows[i+1] - target_rows[i]
                ax_target.arrow(target_cols[i], target_rows[i], dx, dy, 
                               head_width=0.3, head_length=0.5, fc='red', ec='red')
        
        # Mark start and end positions
        ax_target.plot(target_cols[0], target_rows[0], 'ro', markersize=10, label='Start')
        ax_target.plot(target_cols[-1], target_rows[-1], 'r*', markersize=12, label='End')
        
        # Add segment boundaries
        for i, agent in enumerate(self.agents):
            region_states_list = list(agent.region_states)
            min_row = min([r for r, _ in [divmod(s, cols) for s in region_states_list]])
            max_row = max([r for r, _ in [divmod(s, cols) for s in region_states_list]])
            min_col = min([c for _, c in [divmod(s, cols) for s in region_states_list]])
            max_col = max([c for _, c in [divmod(s, cols) for s in region_states_list]])
            
            rect = patches.Rectangle((min_col-0.5, min_row-0.5), max_col-min_col+1, max_row-min_row+1, 
                                    linewidth=2, edgecolor='blue', facecolor='none')
            ax_target.add_patch(rect)
            
        # Mark final positions of all agents
        for i, agent in enumerate(self.agents):
            final_row, final_col = divmod(agent.position, cols)
            ax_target.plot(final_col, final_row, 'bo', markersize=8, label=f'Agent {i}' if i==0 else "")
        
        ax_target.set_title('Target Trajectory')
        ax_target.legend()
        
        # Create a figure for merged beliefs
        fig_merged = plt.figure(figsize=(15, 5 * ((len(merged_beliefs)+1) // 2)))
        
        # Plot each merged belief
        for i, (method_name, merged_belief) in enumerate(merged_beliefs.items()):
            ax = fig_merged.add_subplot((len(merged_beliefs)+1) // 2, 2, i+1)
            merged_grid = merged_belief.reshape(rows, cols)
            im = ax.imshow(merged_grid, cmap='hot', interpolation='nearest', vmin=0, vmax=np.max(merged_grid))
            
            # Add segment boundaries
            for agent in self.agents:
                region_states_list = list(agent.region_states)
                min_row = min([r for r, _ in [divmod(s, cols) for s in region_states_list]])
                max_row = max([r for r, _ in [divmod(s, cols) for s in region_states_list]])
                min_col = min([c for _, c in [divmod(s, cols) for s in region_states_list]])
                max_col = max([c for _, c in [divmod(s, cols) for s in region_states_list]])
                
                rect = patches.Rectangle((min_col-0.5, min_row-0.5), max_col-min_col+1, max_row-min_row+1, 
                                        linewidth=2, edgecolor='blue', facecolor='none')
                ax.add_patch(rect)
            
            # Mark final positions of all agents
            for agent in self.agents:
                final_row, final_col = divmod(agent.position, cols)
                ax.plot(final_col, final_row, 'bo', markersize=8)
                
            # Mark target final position
            final_target_row, final_target_col = divmod(self.target_trajectory[-1], cols)
            ax.plot(final_target_col, final_target_row, 'r*', markersize=12)
                
            ax.set_title(f'Merged Belief: {method_name}')
            fig_merged.colorbar(im, ax=ax)
            
        # Save figures
        plt.figure(fig_agents.number)
        plt.tight_layout()
        plt.savefig('agent_beliefs.png', dpi=300)
        
        plt.figure(fig_target.number)
        plt.tight_layout()
        plt.savefig('target_trajectory.png', dpi=300)
        
        plt.figure(fig_merged.number)
        plt.tight_layout()
        plt.savefig('merged_beliefs.png', dpi=300)
        
        plt.show()
        
        return merged_beliefs


def run_experiment(grid_size=(20, 20), n_agents=4, segmentation='equal', steps=3000, alpha=0.1, beta=0.2):
    """Run the last-step belief merging experiment"""
    experiment = LastStepMergeExperiment(
        grid_size=grid_size,
        n_agents=n_agents,
        segmentation_type=segmentation,
        steps=steps,
        alpha=alpha,
        beta=beta,
        target_mdp=True
    )
    
    print(f"Running simulation with {n_agents} agents for {steps} steps...")
    start_time = time.time()
    experiment.run_simulation()
    sim_time = time.time() - start_time
    print(f"Simulation completed in {sim_time:.2f} seconds")
    
    print("Merging beliefs with different methods...")
    merged_beliefs = {
        "KL Divergence": experiment.merge_beliefs_kl(),
        "Simple Average": experiment.merge_beliefs_average(),
        "Consensus": experiment.merge_beliefs_consensus(iterations=50)
    }
    
    print("Visualizing results...")
    experiment.visualize_results(merged_beliefs)
    
    # Calculate comparison metrics
    rows, cols = grid_size
    final_target_pos = experiment.target_trajectory[-1]
    
    # Create a "true belief" (point mass at target's final position)
    true_belief = np.zeros(rows * cols)
    true_belief[final_target_pos] = 1.0
    
    # KL divergence from true belief
    kl_scores = {}
    for method, belief in merged_beliefs.items():
        # Add small epsilon to avoid division by zero
        true = np.clip(true_belief, 1e-10, 1)
        pred = np.clip(belief, 1e-10, 1)
        
        # KL(true || pred)
        kl = np.sum(true * np.log(true / pred))
        kl_scores[method] = kl
        
        # Also calculate Euclidean distance between belief mode and target
        max_prob_pos = np.argmax(belief)
        max_prob_row, max_prob_col = divmod(max_prob_pos, cols)
        target_row, target_col = divmod(final_target_pos, cols)
        euclidean_dist = np.sqrt((max_prob_row - target_row)**2 + (max_prob_col - target_col)**2)
        
        print(f"{method} - KL Divergence: {kl:.6f}, Euclidean Distance: {euclidean_dist:.2f}")
    
    return experiment, merged_beliefs, kl_scores

def calculate_comparison_metrics(experiment, merged_beliefs):
    """Calculate and print comparison metrics for different merging methods"""
    grid_size = experiment.grid_size
    final_target_pos = experiment.target_trajectory[-1]
    
    # Create a "true belief" (point mass at target's final position)
    true_belief = np.zeros(grid_size[0] * grid_size[1])
    true_belief[final_target_pos] = 1.0
    
    print("\n=== Comparison Metrics ===")
    
    # KL divergence from true belief
    kl_scores = {}
    for method, belief in merged_beliefs.items():
        # Add small epsilon to avoid division by zero
        true = np.clip(true_belief, 1e-10, 1)
        pred = np.clip(belief, 1e-10, 1)
        
        # KL(true || pred)
        kl = np.sum(true * np.log(true / pred))
        kl_scores[method] = kl
        
        # Entropy of belief (measure of uncertainty)
        entropy = -np.sum(pred * np.log(pred))
        
        # Calculate Euclidean distance between belief mode and target
        max_prob_pos = np.argmax(belief)
        max_prob_row, max_prob_col = divmod(max_prob_pos, grid_size[1])
        target_row, target_col = divmod(final_target_pos, grid_size[1])
        euclidean_dist = np.sqrt((max_prob_row - target_row)**2 + (max_prob_col - target_col)**2)
        
        # Probability assigned to true position
        prob_at_true = belief[final_target_pos]
        
        print(f"{method} Metrics:")
        print(f"  - KL Divergence: {kl:.6f}")
        print(f"  - Entropy: {entropy:.6f}")
        print(f"  - Euclidean Distance: {euclidean_dist:.2f}")
        print(f"  - Probability at True Position: {prob_at_true:.6f}")
    
    # Calculate KL divergences between methods
    print("\n=== Method Comparisons ===")
    methods = list(merged_beliefs.keys())
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            method1 = methods[i]
            method2 = methods[j]
            belief1 = np.clip(merged_beliefs[method1], 1e-10, 1)
            belief2 = np.clip(merged_beliefs[method2], 1e-10, 1)
            
            # KL(method1 || method2)
            kl12 = np.sum(belief1 * np.log(belief1 / belief2))
            # KL(method2 || method1)
            kl21 = np.sum(belief2 * np.log(belief2 / belief1))
            # Jensen-Shannon Divergence
            m = 0.5 * (belief1 + belief2)
            js = 0.5 * (np.sum(belief1 * np.log(belief1 / m)) + np.sum(belief2 * np.log(belief2 / m)))
            
            print(f"{method1} vs {method2}:")
            print(f"  - KL({method1}||{method2}): {kl12:.6f}")
            print(f"  - KL({method2}||{method1}): {kl21:.6f}")
            print(f"  - Jensen-Shannon Divergence: {js:.6f}")

def run_conflicting_beliefs_experiment():
    """Test case with overlapping regions and conflicting beliefs"""
    # Create a shared region where multiple agents can operate
    grid_size = (20, 20)
    n_agents = 3
    
    # Custom regions with overlap in the center
    regions = [
        # Agent 0: Left side + center
        [r * grid_size[1] + c for r in range(grid_size[0]) 
                              for c in range(15)],
        # Agent 1: Right side + center  
        [r * grid_size[1] + c for r in range(grid_size[0]) 
                              for c in range(5, grid_size[1])],
        # Agent 2: Middle region
        [r * grid_size[1] + c for r in range(5, 15) 
                              for c in range(5, 15)]
    ]
    
    # Create experiment with custom regions
    experiment = LastStepMergeExperiment(
        grid_size=grid_size,
        n_agents=n_agents,
        segmentation_type='custom',
        steps=1000,
        alpha=0.2,  # Higher false alarm rate
        beta=0.3,   # Higher missed detection rate
        custom_regions=regions
    )
    
    # Create artificial beliefs for each agent (instead of running simulation)
    # Agent 0 believes target is in left-center
    belief0 = np.zeros(grid_size[0] * grid_size[1])
    for r in range(8, 12):
        for c in range(3, 7):
            state = r * grid_size[1] + c
            belief0[state] = 1.0
    experiment.agents[0].belief = belief0 / np.sum(belief0)
    
    # Agent 1 believes target is in right-center
    belief1 = np.zeros(grid_size[0] * grid_size[1])
    for r in range(8, 12):
        for c in range(13, 17):
            state = r * grid_size[1] + c
            belief1[state] = 1.0
    experiment.agents[1].belief = belief1 / np.sum(belief1)
    
    # Agent 2 believes target is in center
    belief2 = np.zeros(grid_size[0] * grid_size[1])
    for r in range(8, 12):
        for c in range(8, 12):
            state = r * grid_size[1] + c
            belief2[state] = 1.0
    experiment.agents[2].belief = belief2 / np.sum(belief2)
    
    # Set a known target position in one of the belief regions
    experiment.target_trajectory = [10 * grid_size[1] + 10]  # Center position
    
    # Merge beliefs with different methods
    merged_beliefs = {
        "KL Divergence": experiment.merge_beliefs_kl(),
        "Simple Average": experiment.merge_beliefs_average(),
        "Consensus": experiment.merge_beliefs_consensus()
    }
    
    # Visualize results
    experiment.visualize_results(merged_beliefs)
    
    # Calculate metrics
    calculate_comparison_metrics(experiment, merged_beliefs)
    
    return experiment, merged_beliefs

def run_multimodal_experiment():
    """Test case with agents having multi-modal belief distributions"""
    grid_size = (20, 20)
    n_agents = 3
    
    # Create experiment with default segmentation
    experiment = LastStepMergeExperiment(
        grid_size=grid_size,
        n_agents=n_agents,
        segmentation_type='equal',
        steps=500
    )
    
    # Agent 0: Bimodal belief in top region
    belief0 = np.zeros(grid_size[0] * grid_size[1])
    # First mode (left)
    for r in range(1, 3):
        for c in range(5, 8):
            state = r * grid_size[1] + c
            belief0[state] = 1.0
    # Second mode (right)
    for r in range(1, 3):
        for c in range(12, 15):
            state = r * grid_size[1] + c
            belief0[state] = 1.0
    experiment.agents[0].belief = belief0 / np.sum(belief0)
    
    # Agent 1: Gaussian-like belief in middle region
    belief1 = np.zeros(grid_size[0] * grid_size[1])
    center_r, center_c = 9, 10
    for r in range(7, 13):
        for c in range(5, 15):
            dist_sq = (r - center_r)**2 + (c - center_c)**2
            if dist_sq < 25:  # Within radius of 5
                state = r * grid_size[1] + c
                belief1[state] = np.exp(-dist_sq/10)  # Gaussian decay
    experiment.agents[1].belief = belief1 / np.sum(belief1)
    
    # Agent 2: Uniform belief along horizontal band in bottom region
    belief2 = np.zeros(grid_size[0] * grid_size[1])
    for r in range(16, 18):
        for c in range(3, 17):
            state = r * grid_size[1] + c
            belief2[state] = 1.0
    experiment.agents[2].belief = belief2 / np.sum(belief2)
    
    # Set a known target position
    experiment.target_trajectory = [9 * grid_size[1] + 10]  # Middle of the grid
    
    # Merge beliefs with different methods
    merged_beliefs = {
        "KL Divergence": experiment.merge_beliefs_kl(),
        "Simple Average": experiment.merge_beliefs_average(),
        "Consensus": experiment.merge_beliefs_consensus()
    }
    
    # Visualize results
    experiment.visualize_results(merged_beliefs)
    
    # Calculate metrics
    calculate_comparison_metrics(experiment, merged_beliefs)
    
    return experiment, merged_beliefs

def run_varying_confidence_experiment():
    """Test case with agents having different levels of confidence"""
    grid_size = (20, 20)
    n_agents = 3
    
    # Create experiment
    experiment = LastStepMergeExperiment(
        grid_size=grid_size,
        n_agents=n_agents,
        segmentation_type='equal',
        steps=500
    )
    
    # Agent 0: Very high confidence but wrong location
    belief0 = np.zeros(grid_size[0] * grid_size[1])
    for r in range(2, 4):
        for c in range(9, 11):
            state = r * grid_size[1] + c
            belief0[state] = 5.0  # Very high confidence
    experiment.agents[0].belief = belief0 / np.sum(belief0)
    
    # Agent 1: Medium confidence and correct location
    belief1 = np.zeros(grid_size[0] * grid_size[1])
    true_pos = 10 * grid_size[1] + 10  # Center position
    for r in range(9, 12):
        for c in range(9, 12):
            state = r * grid_size[1] + c
            dist_sq = (r - 10)**2 + (c - 10)**2
            belief1[state] = np.exp(-dist_sq/2)  # Gaussian around true position
    experiment.agents[1].belief = belief1 / np.sum(belief1)
    
    # Agent 2: Low confidence, diffused belief
    belief2 = np.zeros(grid_size[0] * grid_size[1])
    for r in range(13, 18):
        for c in range(5, 15):
            state = r * grid_size[1] + c
            belief2[state] = 0.2 + 0.05 * np.random.rand()  # Low, noisy confidence
    experiment.agents[2].belief = belief2 / np.sum(belief2)
    
    # Set target position
    experiment.target_trajectory = [true_pos]
    
    # Merge beliefs with different methods
    merged_beliefs = {
        "KL Divergence": experiment.merge_beliefs_kl(),
        "Simple Average": experiment.merge_beliefs_average(),
        "Consensus": experiment.merge_beliefs_consensus()
    }
    
    # Visualize results
    experiment.visualize_results(merged_beliefs)
    
    # Calculate metrics
    calculate_comparison_metrics(experiment, merged_beliefs)
    
    return experiment, merged_beliefs

def run_sparse_dense_experiment():
    """Test case with sparse and dense belief distributions"""
    grid_size = (20, 20)
    n_agents = 3
    
    # Use equal segments
    experiment = LastStepMergeExperiment(
        grid_size=grid_size,
        n_agents=n_agents,
        segmentation_type='equal',
        steps=500
    )
    
    # Agent 0: Sparse, concentrated belief
    belief0 = np.zeros(grid_size[0] * grid_size[1])
    state1 = 2 * grid_size[1] + 5
    state2 = 3 * grid_size[1] + 15
    belief0[state1] = 0.7
    belief0[state2] = 0.3
    experiment.agents[0].belief = belief0
    
    # Agent 1: Medium density belief
    belief1 = np.zeros(grid_size[0] * grid_size[1])
    for r in range(8, 11):
        for c in range(8, 13):
            state = r * grid_size[1] + c
            belief1[state] = 1.0 / (15) # 15 cells
    experiment.agents[1].belief = belief1
    
    # Agent 2: Dense, uniform belief
    belief2 = np.zeros(grid_size[0] * grid_size[1])
    for r in range(13, 19):
        for c in range(3, 17):
            state = r * grid_size[1] + c
            belief2[state] = 1.0 / (6 * 14)  # 84 cells
    experiment.agents[2].belief = belief2
    
    # Set target position
    experiment.target_trajectory = [9 * grid_size[1] + 10]  # In Agent 1's region
    
    # Merge beliefs with different methods
    merged_beliefs = {
        "KL Divergence": experiment.merge_beliefs_kl(),
        "Simple Average": experiment.merge_beliefs_average(),
        "Consensus": experiment.merge_beliefs_consensus()
    }
    
    # Visualize results
    experiment.visualize_results(merged_beliefs)
    
    # Calculate metrics
    calculate_comparison_metrics(experiment, merged_beliefs)
    
    return experiment, merged_beliefs

# # Run the experiment if executed directly
# if __name__ == "__main__":
#     # Run the experiment with parameters
#     experiment, merged_beliefs, metrics = run_experiment(
#         grid_size=(20, 20), 
#         n_agents=4, 
#         steps=3000
#     )

def run_all_experiments():
    """Run all test cases and display results"""
    print("\n=== TEST CASE 1: CONFLICTING BELIEFS ===")
    exp1, beliefs1 = run_conflicting_beliefs_experiment()
    
    print("\n=== TEST CASE 2: MULTI-MODAL DISTRIBUTIONS ===")
    exp2, beliefs2 = run_multimodal_experiment()
    
    print("\n=== TEST CASE 3: VARYING CONFIDENCE LEVELS ===")
    exp3, beliefs3 = run_varying_confidence_experiment()
    
    print("\n=== TEST CASE 4: SPARSE AND DENSE BELIEFS ===")
    exp4, beliefs4 = run_sparse_dense_experiment()
    
    return {
        "conflicting": (exp1, beliefs1),
        "multimodal": (exp2, beliefs2),
        "confidence": (exp3, beliefs3),
        "sparse_dense": (exp4, beliefs4)
    }

if __name__ == "__main__":
    results = run_all_experiments()