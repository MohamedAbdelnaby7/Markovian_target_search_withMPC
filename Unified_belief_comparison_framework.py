import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import random
import time
import itertools
from scipy.optimize import minimize
from scipy.stats import entropy
import json
from Environment import SearchEnvironment, SearchEnvironmentWithBeliefs

class UnifiedBeliefMergingFramework:
    """
    Unified framework for testing different belief merging approaches.
    Combines functionality from why_all_same_test.py, last_step_merge.py, and overlapping_test.py
    """
    
    def __init__(self, grid_size=(20, 20), n_agents=4):
        self.grid_size = grid_size
        self.total_states = grid_size[0] * grid_size[1]
        self.n_agents = n_agents
        
    # =========================== BELIEF CREATION METHODS ===========================
    
    def create_conflicting_beliefs(self):
        """Create agents with explicitly conflicting beliefs"""
        beliefs = []
        for i in range(self.n_agents):
            belief = np.zeros(self.total_states)
            
            # Each agent believes target is in a different quadrant
            if i == 0:  # Top-left
                for r in range(5, 10):
                    for c in range(5, 10):
                        state = r * self.grid_size[1] + c
                        dist = np.sqrt((r - 7.5)**2 + (c - 7.5)**2)
                        belief[state] = np.exp(-dist)
            elif i == 1:  # Top-right
                for r in range(5, 10):
                    for c in range(10, 15):
                        state = r * self.grid_size[1] + c
                        dist = np.sqrt((r - 7.5)**2 + (c - 12.5)**2)
                        belief[state] = np.exp(-dist)
            elif i == 2:  # Bottom-left
                for r in range(10, 15):
                    for c in range(5, 10):
                        state = r * self.grid_size[1] + c
                        dist = np.sqrt((r - 12.5)**2 + (c - 7.5)**2)
                        belief[state] = np.exp(-dist)
            else:  # Bottom-right
                for r in range(10, 15):
                    for c in range(10, 15):
                        state = r * self.grid_size[1] + c
                        dist = np.sqrt((r - 12.5)**2 + (c - 12.5)**2)
                        belief[state] = np.exp(-dist)
            
            belief = belief / np.sum(belief)
            beliefs.append(belief)
        
        return beliefs
    
    def create_overlapping_beliefs(self):
        """Create beliefs with overlapping regions"""
        beliefs = []
        for i in range(self.n_agents):
            belief = np.zeros(self.total_states)
            
            # Create overlapping Gaussian distributions
            center_r = 10 + (i - self.n_agents/2) * 2
            center_c = 10 + (i - self.n_agents/2) * 2
            
            for r in range(self.grid_size[0]):
                for c in range(self.grid_size[1]):
                    dist = np.sqrt((r - center_r)**2 + (c - center_c)**2)
                    if dist < 8:  # Overlapping radius
                        state = r * self.grid_size[1] + c
                        belief[state] = np.exp(-dist**2/10)
            
            belief = belief / np.sum(belief)
            beliefs.append(belief)
        
        return beliefs
    
    def create_different_confidence_beliefs(self):
        """Create agents with different confidence levels"""
        beliefs = []
        confidences = [0.5, 1.0, 2.0, 3.0]  # Different spread parameters
        
        for i in range(self.n_agents):
            belief = np.zeros(self.total_states)
            center_r, center_c = 10 + i*2, 10 + i*2
            confidence = confidences[i % len(confidences)]
            
            for r in range(self.grid_size[0]):
                for c in range(self.grid_size[1]):
                    dist = np.sqrt((r - center_r)**2 + (c - center_c)**2)
                    state = r * self.grid_size[1] + c
                    belief[state] = np.exp(-dist**2/confidence)
            
            belief = belief / np.sum(belief)
            beliefs.append(belief)
        
        return beliefs
    
    # =========================== BELIEF MERGING METHODS ===========================
    
    def merge_beliefs_average(self, beliefs, agent_weights=None):
        """Simple averaging of beliefs"""
        if agent_weights is None:
            agent_weights = np.ones(len(beliefs)) / len(beliefs)
        
        merged = np.zeros_like(beliefs[0])
        for i, belief in enumerate(beliefs):
            merged += agent_weights[i] * belief
        
        return merged / np.sum(merged)
    
    def merge_beliefs_consensus(self, beliefs, iterations=50):
        """Consensus-based merging"""
        n_agents = len(beliefs)
        all_beliefs = [belief.copy() for belief in beliefs]
        
        # Define adjacency matrix
        adjacency = np.ones((n_agents, n_agents)) - np.eye(n_agents)
        weights = adjacency / np.sum(adjacency, axis=1, keepdims=True)
        
        # Run consensus iterations
        for _ in range(iterations):
            new_beliefs = []
            for i in range(n_agents):
                belief_i = np.zeros_like(all_beliefs[0])
                for j in range(n_agents):
                    if weights[i, j] > 0:
                        belief_i += weights[i, j] * all_beliefs[j]
                
                belief_i = belief_i / np.sum(belief_i)
                new_beliefs.append(belief_i)
            
            all_beliefs = new_beliefs
        
        merged = np.mean(all_beliefs, axis=0)
        return merged / np.sum(merged)
    
    def merge_beliefs_kl(self, beliefs, agent_weights=None):
        """KL divergence-based merging with multiple initializations"""
        if agent_weights is None:
            agent_weights = np.ones(len(beliefs))
        
        def kl_divergence(p, q):
            p = np.clip(p, 1e-10, 1)
            q = np.clip(q, 1e-10, 1)
            return np.sum(p * np.log(p / q))
        
        def objective(merged_flat):
            merged = merged_flat.reshape(beliefs[0].shape)
            merged = merged / np.sum(merged)
            
            total_divergence = 0
            for i, belief in enumerate(beliefs):
                total_divergence += agent_weights[i] * kl_divergence(belief, merged)
            
            return total_divergence
        
        # Initial guess: weighted average
        initial_guess = self.merge_beliefs_average(beliefs, agent_weights)
        
        # Constraints and bounds
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = [(0, 1) for _ in range(len(initial_guess))]
        
        result = minimize(
            objective,
            initial_guess.flatten(),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )
        
        if result.success:
            merged = result.x.reshape(initial_guess.shape)
            return merged / np.sum(merged)
        else:
            return initial_guess
    
    def merge_beliefs_geometric_mean(self, beliefs):
        """Geometric mean merging"""
        epsilon = 1e-10
        product = np.ones_like(beliefs[0])
        
        for belief in beliefs:
            product *= np.clip(belief, epsilon, 1)
        
        geo_mean = product ** (1.0 / len(beliefs))
        return geo_mean / np.sum(geo_mean)
    
    def merge_beliefs_weighted_by_entropy(self, beliefs):
        """Weighted average based on belief entropy (lower entropy = higher weight)"""
        entropies = []
        for belief in beliefs:
            p = np.clip(belief, 1e-10, 1)
            entropy_val = -np.sum(p * np.log(p))
            entropies.append(entropy_val)
        
        # Convert to weights (lower entropy = higher confidence = higher weight)
        max_entropy = max(entropies)
        weights = [1.0 - e/max_entropy for e in entropies]
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        return self.merge_beliefs_average(beliefs, weights)
    
    # =========================== COMPARISON METHODS ===========================
    
    def jensen_shannon_divergence(self, p, q):
        """Calculate Jensen-Shannon divergence between two distributions"""
        p = np.clip(p, 1e-10, 1)
        q = np.clip(q, 1e-10, 1)
        m = 0.5 * (p + q)
        return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))
    
    def kl_objective_value(self, beliefs, merged):
        """Calculate the KL divergence objective value"""
        total = 0
        merged = np.clip(merged, 1e-10, 1)
        for belief in beliefs:
            belief = np.clip(belief, 1e-10, 1)
            total += np.sum(belief * np.log(belief / merged))
        return total
    
    def compare_merging_methods(self, beliefs, test_name="test"):
        """Compare different belief merging methods"""
        methods = {
            "Simple Average": self.merge_beliefs_average,
            "Consensus": self.merge_beliefs_consensus,
            "KL Divergence": self.merge_beliefs_kl,
            "Geometric Mean": self.merge_beliefs_geometric_mean,
            "Entropy Weighted": self.merge_beliefs_weighted_by_entropy
        }
        
        results = {}
        print(f"\nComparing belief merging methods for: {test_name}")
        
        for name, method in methods.items():
            start_time = time.time()
            merged = method(beliefs)
            elapsed = time.time() - start_time
            results[name] = merged
            print(f"  - {name}: {elapsed:.4f} seconds")
        
        # Calculate comparison metrics
        print("\nMethod Comparison Metrics:")
        method_names = list(results.keys())
        for i in range(len(method_names)):
            for j in range(i+1, len(method_names)):
                name1, name2 = method_names[i], method_names[j]
                js_div = self.jensen_shannon_divergence(results[name1], results[name2])
                print(f"  - {name1} vs {name2}: JS Divergence = {js_div:.6f}")
        
        return results
    
    # =========================== VISUALIZATION METHODS ===========================
    
    def visualize_beliefs(self, beliefs, merged_beliefs=None, test_name="test"):
        """Visualize individual and merged beliefs"""
        rows, cols = self.grid_size
        n_agents = len(beliefs)
        
        # Create figure for individual beliefs
        fig_individual = plt.figure(figsize=(15, 5 * ((n_agents+1) // 2)))
        
        for i, belief in enumerate(beliefs):
            ax = fig_individual.add_subplot((n_agents+1) // 2, 2, i+1)
            belief_grid = belief.reshape(rows, cols)
            im = ax.imshow(belief_grid, cmap='hot', interpolation='nearest')
            ax.set_title(f'Agent {i+1} Belief')
            fig_individual.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig(f'individual_beliefs_{test_name}.png', dpi=300)
        
        # Create figure for merged beliefs if provided
        if merged_beliefs is not None:
            n_methods = len(merged_beliefs)
            fig_merged = plt.figure(figsize=(15, 5 * ((n_methods+1) // 2)))
            
            for i, (method_name, merged_belief) in enumerate(merged_beliefs.items()):
                ax = fig_merged.add_subplot((n_methods+1) // 2, 2, i+1)
                merged_grid = merged_belief.reshape(rows, cols)
                im = ax.imshow(merged_grid, cmap='hot', interpolation='nearest')
                ax.set_title(f'Merged Belief: {method_name}')
                fig_merged.colorbar(im, ax=ax)
            
            plt.tight_layout()
            plt.savefig(f'merged_beliefs_{test_name}.png', dpi=300)
        
        plt.show()


class CommunicationScenarioComparison:
    """
    Main class for comparing full communication vs. last-step merging scenarios
    """
    
    def __init__(self, grid_size=(20, 20), n_agents=4, steps=1000, 
                 proximity_distance=5, alpha=0.1, beta=0.2):
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.steps = steps
        self.proximity_distance = proximity_distance
        self.alpha = alpha  # False alarm probability
        self.beta = beta    # Missed detection probability
        self.merger = UnifiedBeliefMergingFramework(grid_size, n_agents)
        
        # Generate shared trajectory and initial positions for fair comparison
        self.shared_target_trajectory = self._generate_shared_target_trajectory()
        self.shared_initial_positions = self._generate_shared_initial_positions()
        
    def _generate_shared_target_trajectory(self):
        """Generate a single target trajectory to be used by both scenarios"""
        env = SearchEnvironment(
            grid_size=self.grid_size, 
            n_agents=1,  # Only need target movement
            target_mdp=True
        )
        
        trajectory = [env.true_position]  # Starting position
        
        for step in range(self.steps):
            env.move_target(step)
            trajectory.append(env.true_position)
        
        return trajectory
    
    def _generate_shared_initial_positions(self):
        """Generate initial agent positions to be used by both scenarios"""
        total_states = self.grid_size[0] * self.grid_size[1]
        # Ensure agents don't start at the same position as target
        available_positions = list(range(total_states))
        if self.shared_target_trajectory[0] in available_positions:
            available_positions.remove(self.shared_target_trajectory[0])
        
        initial_positions = np.random.choice(
            available_positions, 
            self.n_agents, 
            replace=False
        )
        return list(initial_positions)
        """Run scenario with continuous communication and belief merging"""
        print("Running Full Communication Scenario...")
        
        # Use SearchEnvironmentWithBeliefs for continuous communication
        env = SearchEnvironmentWithBeliefs(
            grid_size=self.grid_size, 
            n_agents=self.n_agents,
            target_mdp=True, 
            proximity_distance=self.proximity_distance
        )
        
        # Generate target trajectory
        target_trajectory = []
        for step in range(self.steps):
            if step % 100 == 0:
                print(f"  Step {step}/{self.steps}")
                
            env.move_target(step)
            target_trajectory.append(env.true_position)
            
            # Each agent makes observations and updates beliefs
            for agent_id in range(self.n_agents):
                agent_pos = env.agent_positions[agent_id]
                
                # Generate observation
                if agent_pos == env.true_position:
                    observation = 1 if random.random() > self.beta else 0
                else:
                    observation = 1 if random.random() < self.alpha else 0
                
                # Update individual belief
                self._update_agent_belief(env.agent_beliefs[agent_id], agent_pos, observation)
                
                # Move agent (simple greedy policy)
                neighbors = env.get_neighbors(agent_pos)
                if neighbors:
                    belief_values = [env.agent_beliefs[agent_id][pos] for pos in neighbors]
                    best_pos = neighbors[np.argmax(belief_values)]
                    env.agent_positions = list(env.agent_positions)
                    env.agent_positions[agent_id] = best_pos
                    env.agent_positions = tuple(env.agent_positions)
            
            # Check proximity and merge beliefs
            env.check_proximity()
        
        # Final merged belief is the average of all agent beliefs
        final_beliefs = env.get_agent_beliefs()
        final_merged = np.mean(final_beliefs, axis=0)
        final_merged = final_merged / np.sum(final_merged)
        
        return {
            'target_trajectory': target_trajectory,
            'agent_trajectories': env.trajectories['agents'],
            'final_beliefs': final_beliefs,
            'final_merged': final_merged,
            'communication_events': []  # Could track when agents communicated
        }
    
    def run_full_communication_scenario(self):
        """Run scenario with continuous communication and belief merging"""
        print("Running Full Communication Scenario...")
        
        # Use shared initial conditions
        env = SearchEnvironmentWithBeliefs(
            grid_size=self.grid_size, 
            n_agents=self.n_agents,
            target_mdp=True, 
            proximity_distance=self.proximity_distance,
            initial_agent_positions=self.shared_initial_positions.copy(),
            initial_target_position=self.shared_target_trajectory[0],
            precomputed_target_trajectory=self.shared_target_trajectory.copy()
        )
        
        target_trajectory = self.shared_target_trajectory.copy()
        
        for step in range(self.steps):
            if step % 100 == 0:
                print(f"  Step {step}/{self.steps}")
                
            # Use precomputed target position
            env.true_position = target_trajectory[step + 1] if step + 1 < len(target_trajectory) else target_trajectory[-1]
            
            # Each agent makes observations and updates beliefs
            for agent_id in range(self.n_agents):
                agent_pos = env.agent_positions[agent_id]
                
                # Generate observation
                if agent_pos == env.true_position:
                    observation = 1 if random.random() > self.beta else 0
                else:
                    observation = 1 if random.random() < self.alpha else 0
                
                # Update individual belief
                self._update_agent_belief(env.agent_beliefs[agent_id], agent_pos, observation)
                
                # Move agent (simple greedy policy)
                neighbors = env.get_neighbors(agent_pos)
                if neighbors:
                    belief_values = [env.agent_beliefs[agent_id][pos] for pos in neighbors]
                    best_pos = neighbors[np.argmax(belief_values)]
                    env.agent_positions = list(env.agent_positions)
                    env.agent_positions[agent_id] = best_pos
                    env.agent_positions = tuple(env.agent_positions)
            
            # Check proximity and merge beliefs
            env.check_proximity()
        
        # Final merged belief is the average of all agent beliefs
        final_beliefs = env.get_agent_beliefs()
        final_merged = np.mean(final_beliefs, axis=0)
        final_merged = final_merged / np.sum(final_merged)
        
        return {
            'target_trajectory': target_trajectory,
            'agent_trajectories': env.trajectories['agents'],
            'final_beliefs': final_beliefs,
            'final_merged': final_merged,
            'communication_events': []  # Could track when agents communicated
        }
    
    def run_last_step_merge_scenario(self):
        """Run scenario with completely independent agents and final belief merging"""
        print("Running Independent Agents + Last-Step Merge Scenario...")
        
        # Create separate independent environments for each agent
        # Each agent thinks it's the only agent in the world
        independent_agents = []
        
        for agent_id in range(self.n_agents):
            # Each agent gets its own environment and believes it's alone
            agent_env = SearchEnvironment(
                grid_size=self.grid_size, 
                n_agents=1,  # Each agent thinks there's only 1 agent (itself)
                target_mdp=True,
                initial_agent_positions=[self.shared_initial_positions[agent_id]],
                initial_target_position=self.shared_target_trajectory[0],
                precomputed_target_trajectory=self.shared_target_trajectory.copy()
            )
            
            # Initialize belief for this agent
            belief = np.ones(self.grid_size[0] * self.grid_size[1]) / (self.grid_size[0] * self.grid_size[1])
            
            independent_agents.append({
                'env': agent_env,
                'belief': belief,
                'trajectory': [self.shared_initial_positions[agent_id]]
            })
        
        # Run independent simulations for each agent
        for step in range(self.steps):
            if step % 100 == 0:
                print(f"  Step {step}/{self.steps}")
            
            # Each agent operates in complete isolation
            for agent_id, agent_data in enumerate(independent_agents):
                env = agent_data['env']
                belief = agent_data['belief']
                
                # Use precomputed target position (same for all agents)
                target_pos = self.shared_target_trajectory[step + 1] if step + 1 < len(self.shared_target_trajectory) else self.shared_target_trajectory[-1]
                env.true_position = target_pos
                
                # Agent makes observation at its current position
                agent_pos = env.agent_positions[0]  # Only one agent in this environment
                
                # Generate observation
                if agent_pos == target_pos:
                    observation = 1 if random.random() > self.beta else 0
                else:
                    observation = 1 if random.random() < self.alpha else 0
                
                # Update individual belief (no knowledge of other agents)
                self._update_agent_belief(belief, agent_pos, observation)
                
                # Move agent using its own belief (MDP-like policy)
                neighbors = env.get_neighbors(agent_pos)
                if neighbors:
                    # Add exploration noise to encourage diverse search strategies
                    belief_values = [belief[pos] for pos in neighbors]
                    noise = np.random.uniform(0, 0.02, len(neighbors))  # Independent exploration
                    belief_values = np.array(belief_values) + noise
                    best_pos = neighbors[np.argmax(belief_values)]
                    
                    # Update environment
                    env.agent_positions = [best_pos]
                    agent_data['trajectory'].append(best_pos)
        
        # Extract final beliefs from all independent agents
        final_beliefs = [agent_data['belief'] for agent_data in independent_agents]
        agent_trajectories = [agent_data['trajectory'] for agent_data in independent_agents]
        
        # Final step: merge all beliefs using different methods
        print("  Merging independent beliefs at final step...")
        merged_beliefs = self.merger.compare_merging_methods(final_beliefs, "independent_final_merge")
        
        return {
            'target_trajectory': self.shared_target_trajectory,
            'agent_trajectories': agent_trajectories,
            'final_beliefs': final_beliefs,
            'merged_beliefs': merged_beliefs,
            'final_merged': merged_beliefs['KL Divergence']  # Use KL as default
        }
    
    def _update_agent_belief(self, belief, agent_pos, observation):
        """Update an agent's belief based on observation"""
        # Simple Bayesian update
        likelihood = np.ones_like(belief)
        
        if observation == 1:  # Detection
            likelihood[:] = self.alpha
            likelihood[agent_pos] = 1 - self.beta
        else:  # No detection
            likelihood[:] = 1 - self.alpha
            likelihood[agent_pos] = self.beta
        
        # Apply Bayes rule
        updated_belief = belief * likelihood
        
        # Normalize
        if np.sum(updated_belief) > 0:
            belief[:] = updated_belief / np.sum(updated_belief)
    
    def compare_scenarios(self):
        """Compare both scenarios and analyze results"""
        print("=" * 80)
        print("COMPARING FULL COMMUNICATION VS. LAST-STEP MERGING")
        print(f"Using SHARED target trajectory and initial positions")
        print(f"Target starts at: {self.shared_target_trajectory[0]}")
        print(f"Target ends at: {self.shared_target_trajectory[-1]}")
        print(f"Agent initial positions: {self.shared_initial_positions}")
        print("=" * 80)
        
        # Run both scenarios with shared conditions
        full_comm_results = self.run_full_communication_scenario()
        last_step_results = self.run_last_step_merge_scenario()
        
        # Verify we used the same trajectory
        assert full_comm_results['target_trajectory'] == last_step_results['target_trajectory'], \
            "Target trajectories should be identical!"
        
        # Calculate comparison metrics
        final_target_pos = self.shared_target_trajectory[-1]
        
        # Create "ground truth" belief (point mass at final target position)
        ground_truth = np.zeros(self.grid_size[0] * self.grid_size[1])
        ground_truth[final_target_pos] = 1.0
        
        # Compare final merged beliefs
        full_comm_belief = full_comm_results['final_merged']
        last_step_belief = last_step_results['final_merged']
        
        # Calculate metrics
        metrics = {}
        
        for scenario, belief in [("Full Communication", full_comm_belief), 
                                ("Last-Step Merge", last_step_belief)]:
            
            # KL divergence from ground truth
            belief_clipped = np.clip(belief, 1e-10, 1)
            ground_truth_clipped = np.clip(ground_truth, 1e-10, 1)
            kl_div = np.sum(ground_truth_clipped * np.log(ground_truth_clipped / belief_clipped))
            
            # Entropy (uncertainty)
            entropy_val = -np.sum(belief_clipped * np.log(belief_clipped))
            
            # Probability at true position
            prob_at_true = belief[final_target_pos]
            
            # Distance of belief mode from true position
            mode_pos = np.argmax(belief)
            mode_row, mode_col = divmod(mode_pos, self.grid_size[1])
            true_row, true_col = divmod(final_target_pos, self.grid_size[1])
            distance = np.sqrt((mode_row - true_row)**2 + (mode_col - true_col)**2)
            
            metrics[scenario] = {
                'kl_divergence': kl_div,
                'entropy': entropy_val,
                'prob_at_true': prob_at_true,
                'distance_to_true': distance
            }
        
        # Print comparison results
        print("\n=== COMPARISON RESULTS ===")
        for scenario, metric_dict in metrics.items():
            print(f"\n{scenario}:")
            for metric, value in metric_dict.items():
                print(f"  {metric}: {value:.6f}")
        
        # Compare between scenarios
        print(f"\n=== RELATIVE PERFORMANCE ===")
        kl_improvement = (metrics['Last-Step Merge']['kl_divergence'] - 
                         metrics['Full Communication']['kl_divergence']) / metrics['Last-Step Merge']['kl_divergence']
        print(f"KL Divergence improvement (Full Comm vs Last-Step): {kl_improvement*100:.2f}%")
        
        entropy_diff = metrics['Full Communication']['entropy'] - metrics['Last-Step Merge']['entropy']
        print(f"Entropy difference (Full Comm - Last-Step): {entropy_diff:.6f}")
        
        # Visualize results
        self._visualize_comparison_results(full_comm_results, last_step_results, metrics)
        
        return full_comm_results, last_step_results, metrics
    
    def _visualize_comparison_results(self, full_comm_results, last_step_results, metrics):
        """Visualize comparison results"""
        rows, cols = self.grid_size
        
        # Create comparison figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot target trajectory
        target_traj = full_comm_results['target_trajectory']
        target_coords = [divmod(pos, cols) for pos in target_traj]
        target_rows, target_cols = zip(*target_coords)
        
        for i in range(2):
            axes[i, 0].plot(target_cols, target_rows, 'r-', alpha=0.7, linewidth=2)
            axes[i, 0].plot(target_cols[0], target_rows[0], 'ro', markersize=8, label='Start')
            axes[i, 0].plot(target_cols[-1], target_rows[-1], 'r*', markersize=12, label='End')
            axes[i, 0].set_title('Target Trajectory')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
        
        # Plot final beliefs
        full_comm_belief = full_comm_results['final_merged'].reshape(rows, cols)
        last_step_belief = last_step_results['final_merged'].reshape(rows, cols)
        
        im1 = axes[0, 1].imshow(full_comm_belief, cmap='hot', interpolation='nearest')
        axes[0, 1].set_title('Full Communication - Final Belief')
        plt.colorbar(im1, ax=axes[0, 1])
        
        im2 = axes[1, 1].imshow(last_step_belief, cmap='hot', interpolation='nearest')
        axes[1, 1].set_title('Last-Step Merge - Final Belief')
        plt.colorbar(im2, ax=axes[1, 1])
        
        # Plot metrics comparison
        scenarios = list(metrics.keys())
        metric_names = list(metrics[scenarios[0]].keys())
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        values1 = [metrics[scenarios[0]][metric] for metric in metric_names]
        values2 = [metrics[scenarios[1]][metric] for metric in metric_names]
        
        axes[0, 2].bar(x - width/2, values1, width, label=scenarios[0], alpha=0.8)
        axes[0, 2].bar(x + width/2, values2, width, label=scenarios[1], alpha=0.8)
        axes[0, 2].set_xlabel('Metrics')
        axes[0, 2].set_ylabel('Values')
        axes[0, 2].set_title('Performance Comparison')
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(metric_names, rotation=45)
        axes[0, 2].legend()
        
        # Plot difference heatmap
        diff = full_comm_belief - last_step_belief
        im3 = axes[1, 2].imshow(diff, cmap='RdBu', interpolation='nearest', vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
        axes[1, 2].set_title('Belief Difference\n(Full Comm - Last Step)')
        plt.colorbar(im3, ax=axes[1, 2])
        
        plt.tight_layout()
        plt.savefig('scenario_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()


def run_comprehensive_experiments():
    """Run comprehensive experiments comparing different scenarios"""
    
    # Test different configurations
    configs = [
        {"grid_size": (15, 15), "n_agents": 3, "steps": 500, "name": "small_grid"},
        {"grid_size": (20, 20), "n_agents": 4, "steps": 1000, "name": "medium_grid"},
        {"grid_size": (25, 25), "n_agents": 5, "steps": 1500, "name": "large_grid"}
    ]
    
    all_results = {}
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"RUNNING EXPERIMENT: {config['name']}")
        print(f"Grid: {config['grid_size']}, Agents: {config['n_agents']}, Steps: {config['steps']}")
        print(f"{'='*60}")
        
        comparison = CommunicationScenarioComparison(
            grid_size=config['grid_size'],
            n_agents=config['n_agents'],
            steps=config['steps']
        )
        
        full_comm, last_step, metrics = comparison.compare_scenarios()
        all_results[config['name']] = {
            'config': config,
            'full_comm': full_comm,
            'last_step': last_step,
            'metrics': metrics
        }
    
    # Summary analysis
    print(f"\n{'='*80}")
    print("SUMMARY ANALYSIS ACROSS ALL EXPERIMENTS")
    print(f"{'='*80}")
    
    for exp_name, results in all_results.items():
        metrics = results['metrics']
        print(f"\n{exp_name.upper()}:")
        
        full_comm_kl = metrics['Full Communication']['kl_divergence']
        last_step_kl = metrics['Last-Step Merge']['kl_divergence']
        improvement = (last_step_kl - full_comm_kl) / last_step_kl * 100
        
        print(f"  KL Divergence - Full Comm: {full_comm_kl:.6f}, Last-Step: {last_step_kl:.6f}")
        print(f"  Improvement: {improvement:.2f}%")
        print(f"  Winner: {'Full Communication' if improvement > 0 else 'Last-Step Merge'}")
    
    return all_results


def test_belief_merging_methods():
    """Test different belief merging methods on various scenarios"""
    merger = UnifiedBeliefMergingFramework(grid_size=(20, 20), n_agents=4)
    
    test_cases = [
        ("Conflicting Beliefs", merger.create_conflicting_beliefs()),
        ("Overlapping Beliefs", merger.create_overlapping_beliefs()),
        ("Different Confidence", merger.create_different_confidence_beliefs())
    ]
    
    for test_name, beliefs in test_cases:
        print(f"\n{'='*60}")
        print(f"TESTING: {test_name}")
        print(f"{'='*60}")
        
        merged_results = merger.compare_merging_methods(beliefs, test_name)
        merger.visualize_beliefs(beliefs, merged_results, test_name)


class MDPCommunicationPolicy:
    """
    Enhanced MDP-based policy for deciding when and what to communicate
    """
    def __init__(self, communication_cost=0.1, truth_reward=1.0, learning_rate=0.1):
        self.communication_cost = communication_cost
        self.truth_reward = truth_reward
        self.learning_rate = learning_rate
        self.q_table = {}  # State-action value function
        self.state_visits = {}  # Track state visits for exploration
        
    def get_state_key(self, agent_belief, other_beliefs, step):
        """Convert agent state to hashable key for Q-table"""
        # Discretize belief into bins for state representation
        belief_mode = np.argmax(agent_belief)
        belief_entropy = -np.sum(agent_belief * np.log(np.clip(agent_belief, 1e-10, 1)))
        confidence = np.max(agent_belief)
        
        # Discretize values
        entropy_bin = min(int(belief_entropy), 9)
        confidence_bin = min(int(confidence * 10), 9)
        step_bin = min(step // 10, 99)
        
        return (belief_mode, entropy_bin, confidence_bin, step_bin)
    
    def should_communicate(self, agent_belief, other_beliefs, step, epsilon=0.1):
        """Decide whether to communicate using epsilon-greedy Q-learning"""
        state_key = self.get_state_key(agent_belief, other_beliefs, step)
        
        # Initialize Q-values if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = {'communicate': 0.0, 'silent': 0.0}
            self.state_visits[state_key] = 0
        
        self.state_visits[state_key] += 1
        
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = np.random.choice(['communicate', 'silent'])
        else:
            action = max(self.q_table[state_key], key=self.q_table[state_key].get)
        
        return action == 'communicate'
    
    def update_q_value(self, state_key, action, reward):
        """Update Q-value based on observed reward"""
        if state_key not in self.q_table:
            self.q_table[state_key] = {'communicate': 0.0, 'silent': 0.0}
        
        current_q = self.q_table[state_key][action]
        self.q_table[state_key][action] = current_q + self.learning_rate * (reward - current_q)


class EnhancedCommunicationScenarioComparison(CommunicationScenarioComparison):
    """
    Enhanced version with MDP-based communication policy and more sophisticated analysis
    """
    
    def __init__(self, grid_size=(20, 20), n_agents=4, steps=1000, 
                 proximity_distance=5, alpha=0.1, beta=0.2, use_mdp_policy=True):
        super().__init__(grid_size, n_agents, steps, proximity_distance, alpha, beta)
        self.use_mdp_policy = use_mdp_policy
        self.mdp_policies = [MDPCommunicationPolicy() for _ in range(n_agents)]
        
    def run_enhanced_full_communication_scenario(self):
        """Enhanced version with MDP-based communication decisions"""
        print("Running Enhanced Full Communication Scenario with MDP Policy...")
        
        # Use shared initial conditions
        env = SearchEnvironmentWithBeliefs(
            grid_size=self.grid_size, 
            n_agents=self.n_agents,
            target_mdp=True, 
            proximity_distance=self.proximity_distance,
            initial_agent_positions=self.shared_initial_positions.copy(),
            initial_target_position=self.shared_target_trajectory[0],
            precomputed_target_trajectory=self.shared_target_trajectory.copy()
        )
        
        target_trajectory = self.shared_target_trajectory.copy()
        communication_events = []
        belief_history = []
        
        for step in range(self.steps):
            if step % 100 == 0:
                print(f"  Step {step}/{self.steps}")
            
            # Use precomputed target position
            env.true_position = target_trajectory[step + 1] if step + 1 < len(target_trajectory) else target_trajectory[-1]
            
            # Store current beliefs for analysis
            current_beliefs = [belief.copy() for belief in env.agent_beliefs]
            belief_history.append(current_beliefs)
            
            # Each agent decides whether to communicate
            communications_this_step = []
            
            for agent_id in range(self.n_agents):
                agent_pos = env.agent_positions[agent_id]
                
                # Generate observation
                if agent_pos == env.true_position:
                    observation = 1 if random.random() > self.beta else 0
                else:
                    observation = 1 if random.random() < self.alpha else 0
                
                # Update individual belief
                self._update_agent_belief(env.agent_beliefs[agent_id], agent_pos, observation)
                
                # Decide whether to communicate using MDP policy
                if self.use_mdp_policy:
                    other_beliefs = [env.agent_beliefs[i] for i in range(self.n_agents) if i != agent_id]
                    should_comm = self.mdp_policies[agent_id].should_communicate(
                        env.agent_beliefs[agent_id], other_beliefs, step
                    )
                    
                    if should_comm:
                        communications_this_step.append(agent_id)
                        # Calculate reward for communication decision
                        reward = self._calculate_communication_reward(env.agent_beliefs[agent_id], env.true_position)
                        
                        # Update Q-value (simplified - in practice you'd wait for outcome)
                        state_key = self.mdp_policies[agent_id].get_state_key(
                            env.agent_beliefs[agent_id], other_beliefs, step
                        )
                        self.mdp_policies[agent_id].update_q_value(state_key, 'communicate', reward)
                
                # Move agent
                neighbors = env.get_neighbors(agent_pos)
                if neighbors:
                    belief_values = [env.agent_beliefs[agent_id][pos] for pos in neighbors]
                    noise = np.random.uniform(0, 0.01, len(neighbors))  # Exploration noise
                    belief_values = np.array(belief_values) + noise
                    best_pos = neighbors[np.argmax(belief_values)]
                    env.agent_positions = list(env.agent_positions)
                    env.agent_positions[agent_id] = best_pos
                    env.agent_positions = tuple(env.agent_positions)
            
            # Process communications
            if communications_this_step:
                communication_events.append((step, communications_this_step))
                self._process_communications(env, communications_this_step)
        
        # Calculate final metrics
        final_beliefs = env.get_agent_beliefs()
        final_merged = np.mean(final_beliefs, axis=0)
        final_merged = final_merged / np.sum(final_merged)
        
        return {
            'target_trajectory': target_trajectory,
            'agent_trajectories': env.trajectories['agents'],
            'final_beliefs': final_beliefs,
            'final_merged': final_merged,
            'communication_events': communication_events,
            'belief_history': belief_history,
            'mdp_policies': self.mdp_policies
        }
    
    def _calculate_communication_reward(self, agent_belief, true_position):
        """Calculate reward for communication decision"""
        # Reward based on information gain and accuracy
        prob_at_true = agent_belief[true_position]
        entropy_val = -np.sum(agent_belief * np.log(np.clip(agent_belief, 1e-10, 1)))
        
        # Higher reward for high probability at true position, penalty for high entropy
        reward = prob_at_true - 0.1 * entropy_val - self.mdp_policies[0].communication_cost
        return reward
    
    def _process_communications(self, env, communicating_agents):
        """Process communications between agents"""
        if len(communicating_agents) < 2:
            return
        
        # Merge beliefs of communicating agents
        communicating_beliefs = [env.agent_beliefs[i] for i in communicating_agents]
        merged_belief = np.mean(communicating_beliefs, axis=0)
        merged_belief = merged_belief / np.sum(merged_belief)
        
        # Update beliefs of communicating agents
        for agent_id in communicating_agents:
            # Blend with existing belief (partial update)
            alpha = 0.7  # Communication influence factor
            env.agent_beliefs[agent_id] = (alpha * merged_belief + 
                                         (1 - alpha) * env.agent_beliefs[agent_id])
            env.agent_beliefs[agent_id] = env.agent_beliefs[agent_id] / np.sum(env.agent_beliefs[agent_id])
    
    def analyze_communication_patterns(self, results):
        """Analyze communication patterns from the full communication scenario"""
        communication_events = results['communication_events']
        
        print("\n=== COMMUNICATION PATTERN ANALYSIS ===")
        
        # Count total communications
        total_communications = sum(len(agents) for _, agents in communication_events)
        print(f"Total communication events: {len(communication_events)}")
        print(f"Total agent communications: {total_communications}")
        print(f"Average communications per step: {total_communications / self.steps:.3f}")
        
        # Agent-wise communication frequency
        agent_comm_counts = [0] * self.n_agents
        for _, agents in communication_events:
            for agent in agents:
                agent_comm_counts[agent] += 1
        
        print("\nAgent communication frequencies:")
        for i, count in enumerate(agent_comm_counts):
            print(f"  Agent {i}: {count} communications ({count/self.steps:.3f} per step)")
        
        # Communication over time
        comm_by_step = [0] * self.steps
        for step, agents in communication_events:
            comm_by_step[step] = len(agents)
        
        # Plot communication patterns
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Communication frequency over time
        axes[0, 0].plot(comm_by_step)
        axes[0, 0].set_title('Communications per Step')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Number of Communicating Agents')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Agent communication distribution
        axes[0, 1].bar(range(self.n_agents), agent_comm_counts)
        axes[0, 1].set_title('Total Communications per Agent')
        axes[0, 1].set_xlabel('Agent ID')
        axes[0, 1].set_ylabel('Communication Count')
        
        # Belief convergence over time
        if 'belief_history' in results:
            belief_history = results['belief_history']
            convergence_metrics = []
            
            for step_beliefs in belief_history:
                # Calculate pairwise JS divergences between agents
                js_divs = []
                for i in range(len(step_beliefs)):
                    for j in range(i+1, len(step_beliefs)):
                        js_div = self.merger.jensen_shannon_divergence(step_beliefs[i], step_beliefs[j])
                        js_divs.append(js_div)
                convergence_metrics.append(np.mean(js_divs))
            
            axes[1, 0].plot(convergence_metrics)
            axes[1, 0].set_title('Belief Convergence Over Time\n(Lower = More Converged)')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Average JS Divergence Between Agents')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Q-learning statistics
        if self.use_mdp_policy:
            q_values_comm = []
            q_values_silent = []
            
            for policy in self.mdp_policies:
                for state_key, actions in policy.q_table.items():
                    q_values_comm.append(actions['communicate'])
                    q_values_silent.append(actions['silent'])
            
            axes[1, 1].hist([q_values_comm, q_values_silent], bins=20, alpha=0.7, 
                           label=['Communicate', 'Silent'])
            axes[1, 1].set_title('Q-Value Distribution')
            axes[1, 1].set_xlabel('Q-Value')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('communication_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_enhanced_scenarios(self):
        """Enhanced comparison with detailed analysis"""
        print("=" * 80)
        print("ENHANCED COMMUNICATION SCENARIO COMPARISON")
        print(f"Using SHARED target trajectory and initial positions")
        print(f"Target starts at: {self.shared_target_trajectory[0]}")
        print(f"Target ends at: {self.shared_target_trajectory[-1]}")
        print(f"Agent initial positions: {self.shared_initial_positions}")
        print("=" * 80)
        
        # Run enhanced scenarios
        full_comm_results = self.run_enhanced_full_communication_scenario()
        last_step_results = self.run_last_step_merge_scenario()
        
        # Verify we used the same trajectory
        assert full_comm_results['target_trajectory'] == last_step_results['target_trajectory'], \
            "Target trajectories should be identical!"
        
        # Analyze communication patterns
        self.analyze_communication_patterns(full_comm_results)
        
        # Calculate metrics for enhanced results
        final_target_pos = self.shared_target_trajectory[-1]
        ground_truth = np.zeros(self.grid_size[0] * self.grid_size[1])
        ground_truth[final_target_pos] = 1.0
        
        full_comm_belief = full_comm_results['final_merged']
        last_step_belief = last_step_results['final_merged']
        
        # Calculate metrics (same logic as base class)
        metrics = {}
        for scenario, belief in [("Enhanced Full Communication", full_comm_belief), 
                                ("Last-Step Merge", last_step_belief)]:
            
            belief_clipped = np.clip(belief, 1e-10, 1)
            ground_truth_clipped = np.clip(ground_truth, 1e-10, 1)
            kl_div = np.sum(ground_truth_clipped * np.log(ground_truth_clipped / belief_clipped))
            entropy_val = -np.sum(belief_clipped * np.log(belief_clipped))
            prob_at_true = belief[final_target_pos]
            
            mode_pos = np.argmax(belief)
            mode_row, mode_col = divmod(mode_pos, self.grid_size[1])
            true_row, true_col = divmod(final_target_pos, self.grid_size[1])
            distance = np.sqrt((mode_row - true_row)**2 + (mode_col - true_col)**2)
            
            metrics[scenario] = {
                'kl_divergence': kl_div,
                'entropy': entropy_val,
                'prob_at_true': prob_at_true,
                'distance_to_true': distance
            }
        
        # Print results
        print("\n=== ENHANCED COMPARISON RESULTS ===")
        for scenario, metric_dict in metrics.items():
            print(f"\n{scenario}:")
            for metric, value in metric_dict.items():
                print(f"  {metric}: {value:.6f}")
        
        # Compare between scenarios
        print(f"\n=== RELATIVE PERFORMANCE ===")
        kl_improvement = (metrics['Last-Step Merge']['kl_divergence'] - 
                         metrics['Enhanced Full Communication']['kl_divergence']) / metrics['Last-Step Merge']['kl_divergence']
        print(f"KL Divergence improvement (Enhanced Full Comm vs Last-Step): {kl_improvement*100:.2f}%")
        
        entropy_diff = metrics['Enhanced Full Communication']['entropy'] - metrics['Last-Step Merge']['entropy']
        print(f"Entropy difference (Enhanced Full Comm - Last-Step): {entropy_diff:.6f}")
        
        # Additional analysis for MDP policies
        if self.use_mdp_policy:
            self._analyze_mdp_learning(full_comm_results)
        
        # Visualize results (reuse the parent class method)
        self._visualize_comparison_results(full_comm_results, last_step_results, metrics)
        
        return full_comm_results, last_step_results, metrics
    
    def _analyze_mdp_learning(self, results):
        """Analyze MDP learning results"""
        print("\n=== MDP LEARNING ANALYSIS ===")
        
        for i, policy in enumerate(self.mdp_policies):
            print(f"\nAgent {i} MDP Statistics:")
            print(f"  States explored: {len(policy.q_table)}")
            print(f"  Total state visits: {sum(policy.state_visits.values())}")
            
            if policy.q_table:
                comm_values = [actions['communicate'] for actions in policy.q_table.values()]
                silent_values = [actions['silent'] for actions in policy.q_table.values()]
                
                print(f"  Avg Q(communicate): {np.mean(comm_values):.4f}")
                print(f"  Avg Q(silent): {np.mean(silent_values):.4f}")
                print(f"  Communication preference: {np.mean(comm_values) > np.mean(silent_values)}")


def run_parameter_sensitivity_analysis():
    """Run sensitivity analysis on key parameters"""
    print("\n" + "="*80)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("="*80)
    
    base_config = {
        'grid_size': (20, 20),
        'n_agents': 4,
        'steps': 500,
        'proximity_distance': 5,
        'alpha': 0.1,
        'beta': 0.2
    }
    
    # Parameters to test
    parameter_tests = [
        ('proximity_distance', [3, 5, 8, 12]),
        ('alpha', [0.05, 0.1, 0.2, 0.3]),
        ('beta', [0.1, 0.2, 0.3, 0.4]),
        ('n_agents', [3, 4, 5, 6])
    ]
    
    results_summary = {}
    
    for param_name, param_values in parameter_tests:
        print(f"\nTesting parameter: {param_name}")
        param_results = []
        
        for value in param_values:
            print(f"  Testing {param_name} = {value}")
            
            # Create config with modified parameter
            test_config = base_config.copy()
            test_config[param_name] = value
            
            # Run comparison
            comparison = EnhancedCommunicationScenarioComparison(**test_config)
            full_comm, last_step, metrics = comparison.compare_enhanced_scenarios()
            
            # Extract key metrics
            full_comm_kl = metrics['Full Communication']['kl_divergence']
            last_step_kl = metrics['Last-Step Merge']['kl_divergence']
            improvement = (last_step_kl - full_comm_kl) / last_step_kl * 100
            
            param_results.append({
                'value': value,
                'full_comm_kl': full_comm_kl,
                'last_step_kl': last_step_kl,
                'improvement': improvement
            })
        
        results_summary[param_name] = param_results
    
    # Visualize sensitivity analysis
    _visualize_sensitivity_analysis(results_summary)
    
    return results_summary


def _visualize_sensitivity_analysis(results_summary):
    """Visualize parameter sensitivity analysis results"""
    n_params = len(results_summary)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (param_name, results) in enumerate(results_summary.items()):
        if i >= len(axes):
            break
            
        values = [r['value'] for r in results]
        improvements = [r['improvement'] for r in results]
        
        axes[i].plot(values, improvements, 'o-', linewidth=2, markersize=8)
        axes[i].set_title(f'Performance vs {param_name}')
        axes[i].set_xlabel(param_name)
        axes[i].set_ylabel('Improvement (%)')
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Multi-Agent Belief Map Comparison Framework")
    print("=" * 80)
    
    # Choose what to run
    print("Available experiments:")
    print("1. Test belief merging methods")
    print("2. Compare communication scenarios (basic)")
    print("3. Run comprehensive experiments")
    print("4. Enhanced communication comparison with MDP")
    print("5. Parameter sensitivity analysis")
    print("6. Run all experiments")
    
    choice = input("Enter choice (1-6): ")
    
    if choice == "1":
        test_belief_merging_methods()
    elif choice == "2":
        comparison = CommunicationScenarioComparison()
        comparison.compare_scenarios()
    elif choice == "3":
        run_comprehensive_experiments()
    elif choice == "4":
        enhanced_comparison = EnhancedCommunicationScenarioComparison()
        enhanced_comparison.compare_enhanced_scenarios()
    elif choice == "5":
        run_parameter_sensitivity_analysis()
    elif choice == "6":
        print("Running all experiments...")
        test_belief_merging_methods()
        enhanced_comparison = EnhancedCommunicationScenarioComparison()
        enhanced_comparison.compare_enhanced_scenarios()
        run_comprehensive_experiments()
        run_parameter_sensitivity_analysis()
    else:
        print("Invalid choice. Running enhanced communication comparison by default.")
        enhanced_comparison = EnhancedCommunicationScenarioComparison()
        enhanced_comparison.compare_enhanced_scenarios()