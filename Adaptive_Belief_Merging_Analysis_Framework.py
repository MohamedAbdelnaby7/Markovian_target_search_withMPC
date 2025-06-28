import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import json
import time
import pickle
from datetime import datetime
import os
from scipy.stats import entropy, pearsonr, spearmanr
from scipy.optimize import minimize
import pandas as pd
import seaborn as sns
from typing import Dict, List, Tuple, Any, Union
import itertools
import warnings
warnings.filterwarnings('ignore')


class UnifiedBeliefMergingFramework:
    """
    Framework for different belief merging approaches
    """
    def __init__(self, grid_size=(20, 20), n_agents=4):
        self.grid_size = grid_size
        self.total_states = grid_size[0] * grid_size[1]
        self.n_agents = n_agents
        
    def merge_beliefs_average(self, beliefs, agent_weights=None):
        """Simple averaging of beliefs"""
        if agent_weights is None:
            agent_weights = np.ones(len(beliefs)) / len(beliefs)
        
        merged = np.zeros_like(beliefs[0])
        for i, belief in enumerate(beliefs):
            merged += agent_weights[i] * belief
        
        return merged / np.sum(merged)
    
    def merge_beliefs_kl(self, beliefs, agent_weights=None):
        """KL divergence-based merging"""
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
    
    def jensen_shannon_divergence(self, p, q):
        """Calculate Jensen-Shannon divergence between two distributions"""
        p = np.clip(p, 1e-10, 1)
        q = np.clip(q, 1e-10, 1)
        m = 0.5 * (p + q)
        return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))


class TargetMovementPolicy:
    """
    Target movement policy - simple patterns without MPC/MDP
    """
    def __init__(self, grid_size, movement_pattern='random'):
        self.grid_size = grid_size
        self.rows, self.cols = grid_size
        self.movement_pattern = movement_pattern
        self.step_count = 0
        
    def get_next_position(self, current_pos, step=None):
        """Get next position based on movement pattern"""
        if step is not None:
            self.step_count = step
            
        r, c = divmod(current_pos, self.cols)
        
        if self.movement_pattern == 'random':
            # Random walk with 0.8 prob of moving, 0.2 of staying
            if np.random.random() < 0.2:
                return current_pos
                
            moves = []
            if r > 0: moves.append(current_pos - self.cols)
            if r < self.rows-1: moves.append(current_pos + self.cols)
            if c > 0: moves.append(current_pos - 1)
            if c < self.cols-1: moves.append(current_pos + 1)
            
            return np.random.choice(moves) if moves else current_pos
            
        elif self.movement_pattern == 'evasive':
            # Try to move away from center
            center_r, center_c = self.rows // 2, self.cols // 2
            
            # Calculate direction away from center
            dr = 1 if r > center_r else -1 if r < center_r else 0
            dc = 1 if c > center_c else -1 if c < center_c else 0
            
            # Preferred moves
            preferred = []
            if 0 <= r + dr < self.rows:
                preferred.append(current_pos + dr * self.cols)
            if 0 <= c + dc < self.cols:
                preferred.append(current_pos + dc)
                
            if preferred and np.random.random() < 0.7:
                return np.random.choice(preferred)
            else:
                # Random move
                moves = []
                if r > 0: moves.append(current_pos - self.cols)
                if r < self.rows-1: moves.append(current_pos + self.cols)
                if c > 0: moves.append(current_pos - 1)
                if c < self.cols-1: moves.append(current_pos + 1)
                return np.random.choice(moves) if moves else current_pos
                
        elif self.movement_pattern == 'patrol':
            # Circular patrol pattern
            corners = [
                0,  # top-left
                self.cols - 1,  # top-right
                (self.rows - 1) * self.cols + self.cols - 1,  # bottom-right
                (self.rows - 1) * self.cols  # bottom-left
            ]
            
            # Find closest corner
            min_dist = float('inf')
            target_corner = corners[0]
            for corner in corners:
                corner_r, corner_c = divmod(corner, self.cols)
                dist = abs(r - corner_r) + abs(c - corner_c)
                if dist < min_dist and dist > 0:  # Don't stay at current corner
                    min_dist = dist
                    target_corner = corner
                    
            # Move towards target corner
            target_r, target_c = divmod(target_corner, self.cols)
            dr = 1 if target_r > r else -1 if target_r < r else 0
            dc = 1 if target_c > c else -1 if target_c < c else 0
            
            new_r = r + dr
            new_c = c + dc
            
            if 0 <= new_r < self.rows and 0 <= new_c < self.cols:
                return new_r * self.cols + new_c
                
        return current_pos


class MultiAgentMPC:
    """
    Correct MPC implementation for multi-agent search
    """
    def __init__(self, grid_size, n_agents, horizon=2, alpha=0.1, beta=0.2):
        self.grid_size = grid_size
        self.rows, self.cols = grid_size
        self.n_agents = n_agents
        self.horizon = horizon
        self.alpha = alpha  # False positive rate
        self.beta = beta    # False negative rate
        self.n_states = grid_size[0] * grid_size[1]
        
    def get_joint_action(self, beliefs: Union[np.ndarray, List[np.ndarray]], 
                        agent_positions: List[int], 
                        fast_mode: bool = False) -> List[int]:
        """
        Get optimal joint action for all agents
        """
        if fast_mode:
            return self._get_greedy_joint_action(beliefs, agent_positions)
        
        best_joint_action = agent_positions.copy()
        best_value = -float('inf')
        
        # Generate candidate actions for each agent
        candidate_actions = []
        for pos in agent_positions:
            neighbors = self._get_neighbors(pos)
            candidate_actions.append(neighbors)
        
        # Limit search if too many combinations
        max_combinations = 625  # 5^4 for 4 agents
        all_combinations = list(itertools.product(*candidate_actions))
        
        if len(all_combinations) > max_combinations:
            # Sample a subset for large action spaces
            sampled_combinations = [all_combinations[i] for i in 
                                   np.random.choice(len(all_combinations), max_combinations, replace=False)]
        else:
            sampled_combinations = all_combinations
        
        # Evaluate sampled joint actions
        for joint_action in sampled_combinations:
            # Evaluate this joint action over the horizon
            if isinstance(beliefs, np.ndarray):  # Shared belief
                total_value = self._evaluate_shared_belief(
                    beliefs.copy(), list(joint_action), self.horizon
                )
            else:  # Independent beliefs
                total_value = self._evaluate_independent_beliefs(
                    [b.copy() for b in beliefs], list(joint_action), self.horizon
                )
            
            if total_value > best_value:
                best_value = total_value
                best_joint_action = list(joint_action)
        
        return best_joint_action
    
    def _get_greedy_joint_action(self, beliefs: Union[np.ndarray, List[np.ndarray]], 
                                 agent_positions: List[int]) -> List[int]:
        """
        Fast greedy approximation: each agent moves to highest belief neighbor
        """
        joint_action = []
        
        if isinstance(beliefs, np.ndarray):  # Shared belief
            for pos in agent_positions:
                neighbors = self._get_neighbors(pos)
                best_pos = max(neighbors, key=lambda n: beliefs[n])
                joint_action.append(best_pos)
        else:  # Independent beliefs
            for i, pos in enumerate(agent_positions):
                neighbors = self._get_neighbors(pos)
                best_pos = max(neighbors, key=lambda n: beliefs[i][n])
                joint_action.append(best_pos)
        
        return joint_action
    
    def _evaluate_shared_belief(self, belief: np.ndarray, 
                               joint_action: List[int], 
                               horizon: int) -> float:
        """
        Evaluate joint action with shared belief over horizon
        """
        total_value = 0.0
        current_belief = belief.copy()
        
        for h in range(horizon):
            # Simulate joint observations
            simulated_obs = self._simulate_joint_observation(joint_action, current_belief)
            
            # Update shared belief with all observations
            current_belief = self._update_belief_joint(
                current_belief, joint_action, simulated_obs
            )
            
            # Calculate objective (negative entropy for information gain)
            total_value += -entropy(current_belief)
            
            # For future horizons, should consider future movements
            # Simplified: agents stay in place
        
        return total_value
    
    def _evaluate_independent_beliefs(self, beliefs: List[np.ndarray], 
                                    joint_action: List[int], 
                                    horizon: int) -> float:
        """
        Evaluate joint action with independent beliefs over horizon
        """
        total_value = 0.0
        current_beliefs = [b.copy() for b in beliefs]
        
        for h in range(horizon):
            # Each agent has its own belief and makes its own observation
            for i, (action, belief) in enumerate(zip(joint_action, current_beliefs)):
                # Simulate observation for this agent
                obs = self._simulate_single_observation(action, belief)
                
                # Update this agent's belief
                current_beliefs[i] = self._update_belief_single(
                    belief, action, obs
                )
            
            # Calculate total objective across all agents
            for belief in current_beliefs:
                total_value += -entropy(belief)
        
        return total_value
    
    def _simulate_joint_observation(self, joint_positions: List[int], 
                               belief: np.ndarray) -> List[int]:
        """
        Simulate observations for all agents at their positions
        """
        # Normalize belief to ensure it sums to 1
        belief_normalized = belief / np.sum(belief)
        
        # Sample target position from belief (for simulation)
        target_pos = np.random.choice(self.n_states, p=belief_normalized)
        
        observations = []
        for pos in joint_positions:
            if pos == target_pos:
                # True positive with probability (1-beta)
                obs = 1 if np.random.random() < (1 - self.beta) else 0
            else:
                # False positive with probability alpha
                obs = 1 if np.random.random() < self.alpha else 0
            observations.append(obs)
        
        return observations
    
    def _simulate_single_observation(self, position: int, belief: np.ndarray) -> int:
        """
        Simulate observation for single agent
        """
        # Sample target position from belief
        target_pos = np.random.choice(self.n_states, p=belief)
        
        if position == target_pos:
            return 1 if np.random.random() < (1 - self.beta) else 0
        else:
            return 1 if np.random.random() < self.alpha else 0
    
    def _update_belief_joint(self, belief: np.ndarray, 
                           positions: List[int], 
                           observations: List[int]) -> np.ndarray:
        """
        Update belief with joint observations from all agents
        """
        # Likelihood for each state
        likelihood = np.ones(self.n_states)
        
        for pos, obs in zip(positions, observations):
            if obs == 1:  # Detection
                likelihood[pos] *= (1 - self.beta)  # True positive at position
                # False positive elsewhere
                mask = np.ones(self.n_states, dtype=bool)
                mask[pos] = False
                likelihood[mask] *= self.alpha
            else:  # No detection
                likelihood[pos] *= self.beta  # False negative at position
                # True negative elsewhere
                mask = np.ones(self.n_states, dtype=bool)
                mask[pos] = False
                likelihood[mask] *= (1 - self.alpha)
        
        # Bayesian update
        posterior = belief * likelihood
        return posterior / (np.sum(posterior) + 1e-10)
    
    def _update_belief_single(self, belief: np.ndarray, 
                            position: int, 
                            observation: int) -> np.ndarray:
        """
        Update single agent's belief
        """
        likelihood = np.ones(self.n_states)
        
        if observation == 1:  # Detection
            likelihood[position] = 1 - self.beta
            likelihood[np.arange(self.n_states) != position] = self.alpha
        else:  # No detection
            likelihood[position] = self.beta
            likelihood[np.arange(self.n_states) != position] = 1 - self.alpha
        
        # Bayesian update
        posterior = belief * likelihood
        return posterior / (np.sum(posterior) + 1e-10)
    
    def _get_neighbors(self, position: int) -> List[int]:
        """Get valid neighboring positions including current position"""
        r, c = divmod(position, self.cols)
        neighbors = [position]  # Include current position
        
        if r > 0:
            neighbors.append(position - self.cols)
        if r < self.rows - 1:
            neighbors.append(position + self.cols)
        if c > 0:
            neighbors.append(position - 1)
        if c < self.cols - 1:
            neighbors.append(position + 1)
            
        return neighbors


class ControlledMergingExperiment:
    """
    Main experiment class with proper MPC implementation
    """
    
    def __init__(self, grid_size=(20, 20), n_agents=4, alpha=0.1, beta=0.2, horizon=2):
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.alpha = alpha  # False positive rate
        self.beta = beta   # False negative rate
        self.merger = UnifiedBeliefMergingFramework(grid_size, n_agents)
        self.mpc = MultiAgentMPC(grid_size, n_agents, horizon, alpha, beta)
        
    def run_controlled_experiment(self, merge_intervals, n_trials=10, max_steps=1000, 
                                target_pattern='random', verbose=True, fast_mode=True):
        """
        Run controlled experiment where each merging strategy faces IDENTICAL conditions
        """
        all_results = []
        
        for trial in range(n_trials):
            if verbose:
                print(f"\nTrial {trial + 1}/{n_trials}")
                
            # Generate trial conditions
            trial_seed = trial * 1000
            np.random.seed(trial_seed)
            
            # Generate target trajectory
            target_policy = TargetMovementPolicy(self.grid_size, target_pattern)
            initial_target = np.random.randint(0, self.grid_size[0] * self.grid_size[1])
            
            target_trajectory = [initial_target]
            current_pos = initial_target
            for step in range(max_steps):
                current_pos = target_policy.get_next_position(current_pos, step)
                target_trajectory.append(current_pos)
            
            # Generate initial agent positions
            total_states = self.grid_size[0] * self.grid_size[1]
            available_positions = list(range(total_states))
            if initial_target in available_positions:
                available_positions.remove(initial_target)
                
            initial_positions = np.random.choice(
                available_positions, 
                self.n_agents, 
                replace=False
            ).tolist()
            
            # Pre-generate ALL random numbers for this trial
            observation_randoms = np.random.random((max_steps, self.n_agents))
            
            # Store trial configuration
            trial_config = {
                'trial_id': trial,
                'seed': trial_seed,
                'target_trajectory': target_trajectory,
                'initial_positions': initial_positions,
                'observation_randoms': observation_randoms,
                'target_pattern': target_pattern
            }
            
            # Run each merging strategy on this EXACT scenario
            trial_results = {}
            
            for interval in merge_intervals:
                if verbose:
                    print(f"  Testing interval: {interval if interval != float('inf') else 'No merging'}")
                    
                result = self._run_single_experiment(
                    trial_config, 
                    interval, 
                    max_steps,
                    fast_mode
                )
                
                if interval == 0:
                    trial_results['full_comm'] = result
                elif interval == float('inf'):
                    trial_results['no_comm'] = result
                else:
                    trial_results[f'interval_{interval}'] = result
                    
            all_results.append({
                'trial_id': trial,
                'config': trial_config,
                'results': trial_results
            })
            
        return all_results
    
    def _run_single_experiment(self, trial_config, merge_interval, max_steps, fast_mode=True):
        """Run a single experiment with specified merge interval"""
        # Special case for full communication
        if merge_interval == 0:
            return self._run_centralized_full_communication(trial_config, max_steps, fast_mode)
        
        # For other strategies
        start_time = time.time()
        
        # Initialize agents with independent beliefs
        agent_beliefs = [
            np.ones(self.grid_size[0] * self.grid_size[1]) / (self.grid_size[0] * self.grid_size[1])
            for _ in range(self.n_agents)
        ]
        
        agent_positions = trial_config['initial_positions'].copy()
        agent_trajectories = [[pos] for pos in agent_positions]
        
        # Tracking variables
        target_found = False
        first_discovery_step = max_steps
        discovery_count = 0
        entropy_history = []
        divergence_history = []
        merge_events = []
        
        # Run simulation
        for step in range(max_steps):
            # Target position for this step
            target_pos = trial_config['target_trajectory'][step]
            
            # Calculate and store metrics BEFORE actions
            entropies = [entropy(b) for b in agent_beliefs]
            entropy_history.append({
                'mean': np.mean(entropies),
                'std': np.std(entropies),
                'max': np.max(entropies),
                'min': np.min(entropies)
            })
            
            # Calculate divergence between agents
            divergences = []
            for i in range(len(agent_beliefs)):
                for j in range(i+1, len(agent_beliefs)):
                    div = self.merger.jensen_shannon_divergence(agent_beliefs[i], agent_beliefs[j])
                    divergences.append(div)
            
            divergence_history.append({
                'mean': np.mean(divergences) if divergences else 0,
                'std': np.std(divergences) if divergences else 0,
                'max': np.max(divergences) if divergences else 0
            })
            
            # Check if it's time to merge
            if merge_interval != float('inf') and step > 0 and step % merge_interval == 0:
                # Save beliefs BEFORE merge
                beliefs_before = [b.copy() for b in agent_beliefs]
                
                # Perform belief merging
                merged_belief = self.merger.merge_beliefs_kl(agent_beliefs)
                
                # Update all agents with merged belief
                for i in range(self.n_agents):
                    agent_beliefs[i] = merged_belief.copy()
                
                # Calculate entropy change
                entropy_before = np.mean([entropy(b) for b in beliefs_before])
                entropy_after = entropy(merged_belief)
                
                merge_events.append({
                    'step': step,
                    'entropy_before': entropy_before,
                    'entropy_after': entropy_after,
                    'entropy_reduction': entropy_before - entropy_after
                })
            
            # Get joint action using MPC
            joint_action = self.mpc.get_joint_action(
                agent_beliefs,
                agent_positions,
                fast_mode=fast_mode
            )
            
            # Make observations BEFORE moving
            for i, pos in enumerate(agent_positions):
                obs_rand = trial_config['observation_randoms'][step, i]
                
                if pos == target_pos:
                    observation = 1 if obs_rand > self.beta else 0
                    if observation == 1 and not target_found:
                        target_found = True
                        first_discovery_step = step
                    if observation == 1:
                        discovery_count += 1
                else:
                    observation = 1 if obs_rand < self.alpha else 0
                
                # Update individual belief
                agent_beliefs[i] = self.mpc._update_belief_single(
                    agent_beliefs[i], pos, observation
                )
            
            # Execute joint action
            agent_positions = joint_action
            for i, new_pos in enumerate(joint_action):
                agent_trajectories[i].append(new_pos)
        
        # Final merge for no_comm strategy
        if merge_interval == float('inf'):
            final_beliefs = agent_beliefs
            final_merged = self.merger.merge_beliefs_kl(final_beliefs)
        else:
            final_beliefs = agent_beliefs
            final_merged = np.mean(final_beliefs, axis=0)
            final_merged = final_merged / np.sum(final_merged)
        
        # Calculate final metrics
        final_target_pos = trial_config['target_trajectory'][-1]
        
        # Performance metrics
        prob_at_true_target = final_merged[final_target_pos]
        
        # Find position with highest belief
        predicted_pos = np.argmax(final_merged)
        pred_r, pred_c = divmod(predicted_pos, self.grid_size[1])
        true_r, true_c = divmod(final_target_pos, self.grid_size[1])
        prediction_error = np.sqrt((pred_r - true_r)**2 + (pred_c - true_c)**2)
        
        elapsed_time = time.time() - start_time
        
        return {
            'target_found': target_found,
            'first_discovery_step': first_discovery_step,
            'discovery_count': discovery_count,
            'elapsed_time': elapsed_time,
            'final_merged_belief': final_merged,
            'final_entropy': entropy(final_merged),
            'entropy_history': entropy_history,
            'divergence_history': divergence_history,
            'merge_events': merge_events,
            'total_merges': len(merge_events),
            'prob_at_true_target': prob_at_true_target,
            'prediction_error': prediction_error,
            'agent_trajectories': agent_trajectories
        }
    
    def _run_centralized_full_communication(self, trial_config, max_steps, fast_mode=True):
        """Run true full communication - single shared belief, coordinated MPC"""
        start_time = time.time()
        
        # Single shared belief for all agents
        shared_belief = np.ones(self.grid_size[0] * self.grid_size[1]) / (self.grid_size[0] * self.grid_size[1])
        
        # Agent positions
        agent_positions = trial_config['initial_positions'].copy()
        agent_trajectories = [[pos] for pos in agent_positions]
        
        # Tracking variables
        target_found = False
        first_discovery_step = max_steps
        discovery_count = 0
        entropy_history = []
        
        # Add artificial communication overhead
        COMM_OVERHEAD_PER_STEP = 0.001 * self.n_agents * (self.n_agents - 1) / 2
        
        # Run simulation
        for step in range(max_steps):
            # Target position for this step
            target_pos = trial_config['target_trajectory'][step]
            
            # Calculate and store metrics
            entropy_val = entropy(shared_belief)
            entropy_history.append({
                'mean': entropy_val,
                'std': 0,  # No variance - single belief
                'max': entropy_val,
                'min': entropy_val
            })
            
            # Get joint action using MPC with shared belief
            joint_action = self.mpc.get_joint_action(
                shared_belief, 
                agent_positions, 
                fast_mode=fast_mode
            )
            
            # ALL agents make observations BEFORE moving
            for i, pos in enumerate(agent_positions):
                obs_rand = trial_config['observation_randoms'][step, i]
                
                if pos == target_pos:
                    observation = 1 if obs_rand > self.beta else 0
                    if observation == 1 and not target_found:
                        target_found = True
                        first_discovery_step = step
                    if observation == 1:
                        discovery_count += 1
                else:
                    observation = 1 if obs_rand < self.alpha else 0
                
                # Update SHARED belief
                shared_belief = self.mpc._update_belief_single(
                    shared_belief, pos, observation
                )
            
            # Execute joint action
            agent_positions = joint_action
            for i, new_pos in enumerate(joint_action):
                agent_trajectories[i].append(new_pos)
            
            # Add communication overhead
            time.sleep(COMM_OVERHEAD_PER_STEP)
        
        # Final metrics
        final_target_pos = trial_config['target_trajectory'][-1]
        
        # Performance metrics
        prob_at_true_target = shared_belief[final_target_pos]
        
        # Find position with highest belief
        predicted_pos = np.argmax(shared_belief)
        pred_r, pred_c = divmod(predicted_pos, self.grid_size[1])
        true_r, true_c = divmod(final_target_pos, self.grid_size[1])
        prediction_error = np.sqrt((pred_r - true_r)**2 + (pred_c - true_c)**2)
        
        elapsed_time = time.time() - start_time
        
        return {
            'target_found': target_found,
            'first_discovery_step': first_discovery_step,
            'discovery_count': discovery_count,
            'elapsed_time': elapsed_time,
            'final_merged_belief': shared_belief,
            'final_entropy': entropy(shared_belief),
            'entropy_history': entropy_history,
            'divergence_history': [{'mean': 0, 'std': 0, 'max': 0} for _ in range(max_steps)],
            'merge_events': [],  # No merge events - always together
            'total_merges': 0,
            'prob_at_true_target': prob_at_true_target,
            'prediction_error': prediction_error,
            'agent_trajectories': agent_trajectories
        }
    
    def analyze_results(self, all_results):
        """Analyze results across all trials"""
        # Aggregate results by merge interval
        aggregated = {}
        
        for trial_data in all_results:
            for strategy, result in trial_data['results'].items():
                if strategy not in aggregated:
                    aggregated[strategy] = []
                aggregated[strategy].append(result)
        
        # Calculate statistics
        summary = {}
        
        for strategy, results in aggregated.items():
            # Extract interval value for sorting
            if strategy == 'full_comm':
                interval = 0
            elif strategy == 'no_comm':
                interval = float('inf')
            else:
                interval = int(strategy.split('_')[1])
            
            # Calculate averages, handling cases where no target was found
            found_results = [r for r in results if r['target_found']]
            
            summary[strategy] = {
                'interval': interval,
                'discovery_rate': np.mean([r['target_found'] for r in results]),
                'avg_discovery_step': np.mean([r['first_discovery_step'] for r in found_results]) if found_results else max_steps,
                'avg_final_entropy': np.mean([r['final_entropy'] for r in results]),
                'avg_prediction_error': np.mean([r['prediction_error'] for r in results]),
                'avg_prob_at_target': np.mean([r['prob_at_true_target'] for r in results]),
                'avg_computation_time': np.mean([r['elapsed_time'] for r in results]),
                'total_trials': len(results)
            }
        
        return summary, aggregated
    
    def visualize_comparison(self, summary, save_path='controlled_experiment_results'):
        """Create comprehensive visualization of results"""
        os.makedirs(save_path, exist_ok=True)
        
        # Sort strategies by interval
        strategies = sorted(summary.keys(), 
                          key=lambda x: summary[x]['interval'] 
                          if summary[x]['interval'] != float('inf') else 1e10)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig)
        
        # 1. Discovery rate vs merge interval
        ax1 = fig.add_subplot(gs[0, 0])
        intervals = []
        discovery_rates = []
        
        for strategy in strategies:
            if summary[strategy]['interval'] != float('inf'):
                intervals.append(summary[strategy]['interval'])
                discovery_rates.append(summary[strategy]['discovery_rate'] * 100)
        
        ax1.plot(intervals, discovery_rates, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Merge Interval (steps)')
        ax1.set_ylabel('Discovery Rate (%)')
        ax1.set_title('Target Discovery Success Rate')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 105)
        
        # Add no_comm baseline
        no_comm_rate = summary['no_comm']['discovery_rate'] * 100
        ax1.axhline(y=no_comm_rate, color='red', linestyle='--', 
                   label=f'No Communication: {no_comm_rate:.1f}%')
        ax1.legend()
        
        # 2. Average discovery time
        ax2 = fig.add_subplot(gs[0, 1])
        avg_steps = []
        
        for strategy in strategies:
            if summary[strategy]['interval'] != float('inf') and 'avg_discovery_step' in summary[strategy]:
                avg_steps.append(summary[strategy]['avg_discovery_step'])
            else:
                avg_steps.append(np.nan)
        
        # Remove NaN values for plotting
        valid_intervals = [intervals[i] for i in range(len(avg_steps)) if not np.isnan(avg_steps[i])]
        valid_steps = [s for s in avg_steps if not np.isnan(s)]
        
        if valid_intervals and valid_steps:
            ax2.plot(valid_intervals, valid_steps, 's-', linewidth=2, markersize=8, color='orange')
            ax2.set_xlabel('Merge Interval (steps)')
            ax2.set_ylabel('Average Discovery Step')
            ax2.set_title('Speed of Target Discovery')
            ax2.grid(True, alpha=0.3)
            
            # Add no_comm baseline if available
            if 'no_comm' in summary and 'avg_discovery_step' in summary['no_comm']:
                no_comm_steps = summary['no_comm']['avg_discovery_step']
                if not np.isnan(no_comm_steps):
                    ax2.axhline(y=no_comm_steps, color='red', linestyle='--',
                               label=f'No Communication: {no_comm_steps:.1f}')
                    ax2.legend()
        
        # 3. Final belief entropy
        ax3 = fig.add_subplot(gs[0, 2])
        final_entropies = []
        
        for strategy in strategies:
            if summary[strategy]['interval'] != float('inf'):
                final_entropies.append(summary[strategy]['avg_final_entropy'])
        
        ax3.plot(intervals, final_entropies, '^-', linewidth=2, markersize=8, color='green')
        ax3.set_xlabel('Merge Interval (steps)')
        ax3.set_ylabel('Final Belief Entropy')
        ax3.set_title('Uncertainty in Final Belief')
        ax3.grid(True, alpha=0.3)
        
        # Add no_comm baseline
        no_comm_entropy = summary['no_comm']['avg_final_entropy']
        ax3.axhline(y=no_comm_entropy, color='red', linestyle='--',
                   label=f'No Communication: {no_comm_entropy:.2f}')
        ax3.legend()
        
        # 4. Prediction error
        ax4 = fig.add_subplot(gs[1, 0])
        pred_errors = []
        
        for strategy in strategies:
            if summary[strategy]['interval'] != float('inf'):
                pred_errors.append(summary[strategy]['avg_prediction_error'])
        
        ax4.plot(intervals, pred_errors, 'd-', linewidth=2, markersize=8, color='purple')
        ax4.set_xlabel('Merge Interval (steps)')
        ax4.set_ylabel('Prediction Error (grid cells)')
        ax4.set_title('Location Prediction Accuracy')
        ax4.grid(True, alpha=0.3)
        
        # Add no_comm baseline
        no_comm_error = summary['no_comm']['avg_prediction_error']
        ax4.axhline(y=no_comm_error, color='red', linestyle='--',
                   label=f'No Communication: {no_comm_error:.2f}')
        ax4.legend()
        
        # 5. Probability at true target
        ax5 = fig.add_subplot(gs[1, 1])
        prob_at_target = []
        
        for strategy in strategies:
            if summary[strategy]['interval'] != float('inf'):
                prob_at_target.append(summary[strategy]['avg_prob_at_target'])
        
        ax5.plot(intervals, prob_at_target, 'o-', linewidth=2, markersize=8, color='brown')
        ax5.set_xlabel('Merge Interval (steps)')
        ax5.set_ylabel('Probability at True Target')
        ax5.set_title('Belief Accuracy at Target Location')
        ax5.grid(True, alpha=0.3)
        
        # Add no_comm baseline
        no_comm_prob = summary['no_comm']['avg_prob_at_target']
        ax5.axhline(y=no_comm_prob, color='red', linestyle='--',
                   label=f'No Communication: {no_comm_prob:.4f}')
        ax5.legend()
        
        # 6. Computational cost
        ax6 = fig.add_subplot(gs[1, 2])
        comp_times = []
        
        for strategy in strategies:
            if summary[strategy]['interval'] != float('inf'):
                comp_times.append(summary[strategy]['avg_computation_time'])
        
        ax6.plot(intervals, comp_times, 'v-', linewidth=2, markersize=8, color='red')
        ax6.set_xlabel('Merge Interval (steps)')
        ax6.set_ylabel('Computation Time (seconds)')
        ax6.set_title('Computational Cost')
        ax6.grid(True, alpha=0.3)
        
        # 7. Performance comparison bar chart
        ax7 = fig.add_subplot(gs[2, :])
        
        # Calculate performance score
        strategies_display = []
        performance_scores = []
        
        for strategy in strategies:
            if strategy == 'full_comm':
                name = 'Full Comm'
            elif strategy == 'no_comm':
                name = 'No Comm'
            else:
                name = f'Interval {summary[strategy]["interval"]}'
            
            strategies_display.append(name)
            
            # Composite performance score
            score = (summary[strategy]['discovery_rate'] * 100 / 
                    (1 + summary[strategy]['avg_prediction_error']) *
                    (1 / (1 + summary[strategy]['avg_computation_time'])))
            performance_scores.append(score)
        
        bars = ax7.bar(strategies_display, performance_scores, alpha=0.7)
        
        # Color code bars
        for i, bar in enumerate(bars):
            if i == 0:  # Full comm
                bar.set_color('blue')
            elif i == len(bars) - 1:  # No comm
                bar.set_color('red')
            else:
                bar.set_color('green')
        
        ax7.set_xlabel('Merging Strategy')
        ax7.set_ylabel('Composite Performance Score')
        ax7.set_title('Overall Performance Comparison (Higher is Better)')
        ax7.grid(True, alpha=0.3, axis='y')
        ax7.set_xticklabels(strategies_display, rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save summary to text file
        with open(f'{save_path}/summary_report.txt', 'w') as f:
            f.write("CONTROLLED EXPERIMENT SUMMARY\n")
            f.write("="*50 + "\n\n")
            
            for strategy in strategies:
                f.write(f"\n{strategy.upper()}:\n")
                for metric, value in summary[strategy].items():
                    if isinstance(value, float):
                        f.write(f"  {metric}: {value:.4f}\n")
                    else:
                        f.write(f"  {metric}: {value}\n")
            
            # Find best strategy
            best_discovery = max(strategies, key=lambda x: summary[x]['discovery_rate'])
            best_accuracy = min(strategies, key=lambda x: summary[x]['avg_prediction_error'])
            
            # Handle case where no trials found the target
            finite_strategies = [s for s in strategies if s != 'no_comm']
            if any(summary[s].get('avg_discovery_step', float('inf')) < float('inf') for s in finite_strategies):
                best_speed = min(finite_strategies, 
                               key=lambda x: summary[x].get('avg_discovery_step', float('inf')))
            else:
                best_speed = "None (no discoveries)"
            
            f.write(f"\n\nBEST STRATEGIES:\n")
            f.write(f"  Highest Discovery Rate: {best_discovery}\n")
            f.write(f"  Best Accuracy: {best_accuracy}\n")
            f.write(f"  Fastest Discovery: {best_speed}\n")
    
    def visualize_merge_analysis(self, aggregated_results, pattern_name, save_path):
        """Visualize detailed merge event analysis"""
        # Create figure for merge analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Entropy reduction at merge events
        ax1 = axes[0, 0]
        
        for strategy, results in aggregated_results.items():
            if strategy == 'no_comm' or strategy == 'full_comm':
                continue
                
            # Extract merge interval
            interval = int(strategy.split('_')[1])
            
            # Collect entropy reductions from all trials
            all_reductions = []
            for result in results:
                if 'merge_events' in result and result['merge_events']:
                    for event in result['merge_events']:
                        if isinstance(event, dict) and 'entropy_reduction' in event:
                            all_reductions.append(event['entropy_reduction'])
            
            if all_reductions:
                ax1.scatter([interval] * len(all_reductions), all_reductions, 
                           alpha=0.5, s=20, label=f'Interval {interval}')
        
        ax1.set_xlabel('Merge Interval')
        ax1.set_ylabel('Entropy Reduction per Merge')
        ax1.set_title('Information Gain from Merging')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Entropy evolution over time
        ax2 = axes[0, 1]
        
        # Plot average entropy over time for different strategies
        for strategy, results in aggregated_results.items():
            if len(results) == 0:
                continue
                
            # Average entropy across trials
            max_steps = len(results[0]['entropy_history'])
            avg_entropy = np.zeros(max_steps)
            
            for result in results:
                for i, entropy_data in enumerate(result['entropy_history']):
                    if i < max_steps:
                        avg_entropy[i] += entropy_data['mean']
            
            avg_entropy /= len(results)
            
            # Determine label
            if strategy == 'full_comm':
                label = 'Full Comm'
            elif strategy == 'no_comm':
                label = 'No Comm'
            else:
                interval = int(strategy.split('_')[1])
                label = f'Interval {interval}'
            
            ax2.plot(avg_entropy, label=label, alpha=0.8)
        
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Average Entropy')
        ax2.set_title('Belief Uncertainty Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Divergence between agents over time
        ax3 = axes[1, 0]
        
        for strategy, results in aggregated_results.items():
            if len(results) == 0:
                continue
                
            # Average divergence across trials
            max_steps = len(results[0]['divergence_history'])
            avg_divergence = np.zeros(max_steps)
            
            for result in results:
                for i, div_data in enumerate(result['divergence_history']):
                    if i < max_steps:
                        avg_divergence[i] += div_data['mean']
            
            avg_divergence /= len(results)
            
            # Determine label
            if strategy == 'full_comm':
                label = 'Full Comm'
            elif strategy == 'no_comm':
                label = 'No Comm'
            else:
                interval = int(strategy.split('_')[1])
                label = f'Interval {interval}'
            
            ax3.plot(avg_divergence, label=label, alpha=0.8)
        
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Average JS Divergence')
        ax3.set_title('Belief Divergence Between Agents')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Merge timing visualization
        ax4 = axes[1, 1]
        
        # Show when merges occur for each strategy
        y_pos = 0
        for strategy, results in aggregated_results.items():
            if strategy == 'no_comm' or strategy == 'full_comm':
                continue
                
            interval = int(strategy.split('_')[1])
            
            # Plot merge events as vertical lines
            merge_times = list(range(interval, 1000, interval))
            for t in merge_times:
                ax4.axvline(x=t, ymin=y_pos/10, ymax=(y_pos+0.8)/10, 
                           color='blue', alpha=0.5, linewidth=1)
            
            ax4.text(-50, y_pos, f'Interval {interval}', fontsize=10)
            y_pos += 1
        
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Merge Strategy')
        ax4.set_title('Merge Event Timeline')
        ax4.set_xlim(-100, 1000)
        ax4.set_ylim(-0.5, y_pos)
        ax4.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/merge_analysis_{pattern_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_trajectories(self, trial_data, strategy='full_comm', save_path=None):
        """Visualize agent and target trajectories for a specific trial"""
        result = trial_data['results'][strategy]
        config = trial_data['config']
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw grid
        rows, cols = self.grid_size
        for i in range(rows + 1):
            ax.axhline(y=i, color='gray', linewidth=0.5, alpha=0.3)
        for j in range(cols + 1):
            ax.axvline(x=j, color='gray', linewidth=0.5, alpha=0.3)
        
        # Plot target trajectory
        target_traj = config['target_trajectory']
        target_coords = [divmod(pos, cols) for pos in target_traj]
        target_rows, target_cols = zip(*target_coords)
        
        # Plot with gradient color
        for i in range(len(target_rows) - 1):
            alpha = 0.3 + 0.7 * i / len(target_rows)
            ax.plot([target_cols[i], target_cols[i+1]], 
                   [target_rows[i], target_rows[i+1]], 
                   'r-', alpha=alpha, linewidth=2)
        
        # Mark start and end
        ax.plot(target_cols[0], target_rows[0], 'ro', markersize=10, label='Target Start')
        ax.plot(target_cols[-1], target_rows[-1], 'r*', markersize=15, label='Target End')
        
        # Plot agent trajectories
        colors = ['blue', 'green', 'orange', 'purple']
        for agent_id, trajectory in enumerate(result['agent_trajectories']):
            agent_coords = [divmod(pos, cols) for pos in trajectory]
            agent_rows, agent_cols = zip(*agent_coords)
            
            # Plot trajectory
            ax.plot(agent_cols, agent_rows, 
                   color=colors[agent_id % len(colors)], 
                   alpha=0.6, linewidth=1.5, 
                   label=f'Agent {agent_id}')
            
            # Mark start position
            ax.plot(agent_cols[0], agent_rows[0], 'o', 
                   color=colors[agent_id % len(colors)], 
                   markersize=8)
        
        # Plot final belief as heatmap
        belief = result['final_merged_belief'].reshape(rows, cols)
        im = ax.imshow(belief, cmap='YlOrRd', alpha=0.5, extent=[0, cols, rows, 0])
        plt.colorbar(im, ax=ax, label='Final Belief Probability')
        
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_title(f'Trajectories - {strategy}')
        ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f'{save_path}/trajectories_{strategy}.png', dpi=300, bbox_inches='tight')
        plt.show()


def run_comprehensive_controlled_experiment():
    """Run the complete controlled experiment"""
    print("="*80)
    print("CONTROLLED BELIEF MERGING EXPERIMENT WITH PROPER MPC")
    print("="*80)
    
    # Configuration
    config = {
        'grid_size': (20, 20),
        'n_agents': 4,
        'alpha': 0.1,  # False positive rate
        'beta': 0.2,   # False negative rate
        'horizon': 2,   # MPC horizon
        'n_trials': 30,
        'max_steps': 1000,
        'merge_intervals': [0, 10, 25, 50, 100, 200, 500, float('inf')],
        'target_patterns': ['random', 'evasive', 'patrol'],
        'fast_mode': False  # Use greedy approximation for speed
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Run experiments for each target pattern
    all_pattern_results = {}
    
    for pattern in config['target_patterns']:
        print(f"\n\nTesting Target Pattern: {pattern.upper()}")
        print("-"*50)
        
        experiment = ControlledMergingExperiment(
            grid_size=config['grid_size'],
            n_agents=config['n_agents'],
            alpha=config['alpha'],
            beta=config['beta'],
            horizon=config['horizon']
        )
        
        # Run controlled trials
        results = experiment.run_controlled_experiment(
            merge_intervals=config['merge_intervals'],
            n_trials=config['n_trials'],
            max_steps=config['max_steps'],
            target_pattern=pattern,
            verbose=True,
            fast_mode=config['fast_mode']
        )
        
        # Analyze results
        summary, aggregated = experiment.analyze_results(results)
        
        # Store results
        all_pattern_results[pattern] = {
            'raw_results': results,
            'summary': summary,
            'aggregated': aggregated
        }
        
        # Visualize results for this pattern
        save_path = f'controlled_results_{pattern}'
        experiment.visualize_comparison(summary, save_path)
        
        # Add merge analysis visualization
        experiment.visualize_merge_analysis(aggregated, pattern, save_path)
        
        # Visualize sample trajectories
        if len(results) > 0:
            experiment.visualize_trajectories(results[0], 'full_comm', save_path)
            experiment.visualize_trajectories(results[0], 'no_comm', save_path)
        
        # Print summary
        print(f"\n{pattern.upper()} PATTERN SUMMARY:")
        for strategy, stats in summary.items():
            if stats['interval'] == 0:
                interval_str = "Full Communication"
            elif stats['interval'] == float('inf'):
                interval_str = "No Communication"
            else:
                interval_str = f"Merge every {stats['interval']} steps"
            
            print(f"\n  {interval_str}:")
            print(f"    Discovery Rate: {stats['discovery_rate']*100:.1f}%")
            print(f"    Avg Discovery Step: {stats['avg_discovery_step']:.1f}")
            print(f"    Avg Prediction Error: {stats['avg_prediction_error']:.2f} cells")
            print(f"    Avg Computation Time: {stats['avg_computation_time']:.3f} seconds")
    
    # Cross-pattern analysis
    print("\n\n" + "="*80)
    print("CROSS-PATTERN ANALYSIS")
    print("="*80)
    
    # Find best interval for each pattern
    for pattern, data in all_pattern_results.items():
        summary = data['summary']
        
        # Exclude no_comm from best interval search
        finite_strategies = [s for s in summary.keys() if summary[s]['interval'] != float('inf')]
        
        best_by_discovery = max(finite_strategies, key=lambda x: summary[x]['discovery_rate'])
        best_by_accuracy = min(finite_strategies, key=lambda x: summary[x]['avg_prediction_error'])
        best_by_speed = min(finite_strategies, 
                           key=lambda x: summary[x].get('avg_discovery_step', float('inf')))
        
        print(f"\n{pattern.upper()} PATTERN:")
        print(f"  Best interval for discovery rate: {summary[best_by_discovery]['interval']}")
        print(f"  Best interval for accuracy: {summary[best_by_accuracy]['interval']}")
        print(f"  Best interval for speed: {summary[best_by_speed]['interval']}")
    
    # Save all results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'controlled_experiment_results_{timestamp}.pkl', 'wb') as f:
        pickle.dump(all_pattern_results, f)
    
    print(f"\n\nResults saved to: controlled_experiment_results_{timestamp}.pkl")
    
    # Create final summary plot comparing all patterns
    create_pattern_comparison_plot(all_pattern_results)
    
    return all_pattern_results


def create_pattern_comparison_plot(all_pattern_results):
    """Create a comparison plot across all target patterns"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    patterns = list(all_pattern_results.keys())
    colors = ['blue', 'green', 'orange']
    
    # 1. Discovery rate comparison
    ax = axes[0, 0]
    for i, pattern in enumerate(patterns):
        summary = all_pattern_results[pattern]['summary']
        intervals = []
        discovery_rates = []
        
        for strategy, stats in summary.items():
            if stats['interval'] != float('inf'):
                intervals.append(stats['interval'])
                discovery_rates.append(stats['discovery_rate'] * 100)
        
        ax.plot(intervals, discovery_rates, 'o-', color=colors[i], 
               label=f'{pattern.capitalize()}', linewidth=2, markersize=8)
    
    ax.set_xlabel('Merge Interval (steps)')
    ax.set_ylabel('Discovery Rate (%)')
    ax.set_title('Discovery Rate by Target Pattern')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Average discovery time comparison
    ax = axes[0, 1]
    for i, pattern in enumerate(patterns):
        summary = all_pattern_results[pattern]['summary']
        intervals = []
        avg_steps = []
        
        for strategy, stats in summary.items():
            if stats['interval'] != float('inf') and stats['avg_discovery_step'] < 1000:
                intervals.append(stats['interval'])
                avg_steps.append(stats['avg_discovery_step'])
        
        if intervals and avg_steps:
            ax.plot(intervals, avg_steps, 's-', color=colors[i], 
                   label=f'{pattern.capitalize()}', linewidth=2, markersize=8)
    
    ax.set_xlabel('Merge Interval (steps)')
    ax.set_ylabel('Average Discovery Step')
    ax.set_title('Discovery Speed by Target Pattern')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Final entropy comparison
    ax = axes[1, 0]
    for i, pattern in enumerate(patterns):
        summary = all_pattern_results[pattern]['summary']
        intervals = []
        entropies = []
        
        for strategy, stats in summary.items():
            if stats['interval'] != float('inf'):
                intervals.append(stats['interval'])
                entropies.append(stats['avg_final_entropy'])
        
        ax.plot(intervals, entropies, '^-', color=colors[i], 
               label=f'{pattern.capitalize()}', linewidth=2, markersize=8)
    
    ax.set_xlabel('Merge Interval (steps)')
    ax.set_ylabel('Final Entropy')
    ax.set_title('Final Belief Uncertainty by Target Pattern')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Best strategy summary
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create summary table
    table_data = []
    for pattern in patterns:
        summary = all_pattern_results[pattern]['summary']
        finite_strategies = [s for s in summary.keys() if summary[s]['interval'] != float('inf')]
        
        best_discovery = max(finite_strategies, key=lambda x: summary[x]['discovery_rate'])
        best_interval = summary[best_discovery]['interval']
        best_rate = summary[best_discovery]['discovery_rate'] * 100
        
        table_data.append([pattern.capitalize(), best_interval, f'{best_rate:.1f}%'])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Pattern', 'Best Interval', 'Discovery Rate'],
                    cellLoc='center',
                    loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    ax.text(0.5, 0.8, 'Optimal Merge Intervals by Pattern', 
           ha='center', va='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('pattern_comparison_summary.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Run the comprehensive controlled experiment
    results = run_comprehensive_controlled_experiment()
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
    
    # Print key insights
    print("\nKEY INSIGHTS:")
    print("1. Full communication may not always be optimal due to:")
    print("   - Loss of exploration diversity")
    print("   - Higher computational cost with joint MPC")
    print("   - Communication overhead")
    print("\n2. Periodic merging can balance:")
    print("   - Information sharing benefits")
    print("   - Computational efficiency")
    print("   - Exploration diversity")
    print("\n3. Optimal merge interval depends on:")
    print("   - Target movement pattern")
    print("   - Sensor accuracy (α, β)")
    print("   - Computational constraints")