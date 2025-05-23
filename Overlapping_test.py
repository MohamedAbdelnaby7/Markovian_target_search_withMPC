# overlapping_test.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

import time

class OverlappingBeliefTester:
    def __init__(self):
        """Initialize the tester class"""
        self.grid_size = (20, 20)
        self.total_states = self.grid_size[0] * self.grid_size[1]
        
    # Add this method to the OverlappingBeliefTester class

    def create_scattered_irregular_beliefs(self):
        """Create highly irregular, scattered belief distributions"""
        belief1 = np.zeros(self.total_states)
        belief2 = np.zeros(self.total_states)
        
        # Agent 1: Scattered multi-modal distribution with irregular patterns
        # Create several small, irregular clusters
        clusters1 = [
            # Format: [(row_range), (col_range), intensity_multiplier]
            [(2, 5), (3, 7), 2.0],       # Top-left cluster
            [(4, 6), (12, 16), 1.5],     # Top-right cluster
            [(8, 12), (7, 10), 3.0],     # Center cluster
            [(14, 17), (4, 8), 1.0],     # Bottom-left cluster
            [(15, 18), (14, 19), 2.5]    # Bottom-right cluster
        ]
        
        # Add random noise everywhere
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                state = r * self.grid_size[1] + c
                # Add some base randomness everywhere
                belief1[state] = 0.05 * np.random.random()
        
        # Add the clusters with varied intensity
        for (r_start, r_end), (c_start, c_end), intensity in clusters1:
            for r in range(r_start, r_end):
                for c in range(c_start, c_end):
                    if 0 <= r < self.grid_size[0] and 0 <= c < self.grid_size[1]:
                        state = r * self.grid_size[1] + c
                        # Add varied intensity with some randomness
                        belief1[state] = intensity * (0.5 + 0.5 * np.random.random())
        
        # Ensure there are some sharp peaks
        for _ in range(5):
            r = np.random.randint(0, self.grid_size[0])
            c = np.random.randint(0, self.grid_size[1])
            state = r * self.grid_size[1] + c
            belief1[state] = 5.0 + 2.0 * np.random.random()
        
        belief1 = belief1 / np.sum(belief1)
        
        # Agent 2: Different scattered pattern with partial overlap
        # Create several small, irregular clusters
        clusters2 = [
            # Some clusters in different locations
            [(1, 4), (12, 16), 2.0],     # Top-right cluster
            [(6, 9), (2, 6), 1.5],       # Mid-left cluster
            [(9, 13), (8, 11), 2.0],     # Center cluster (partial overlap with Agent 1)
            [(10, 14), (15, 19), 3.0],   # Mid-right cluster
            [(16, 19), (7, 12), 1.0]     # Bottom-mid cluster
        ]
        
        # Add random noise everywhere
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                state = r * self.grid_size[1] + c
                # Add some base randomness everywhere (different pattern from Agent 1)
                belief2[state] = 0.08 * np.random.random()
        
        # Add the clusters with varied intensity
        for (r_start, r_end), (c_start, c_end), intensity in clusters2:
            for r in range(r_start, r_end):
                for c in range(c_start, c_end):
                    if 0 <= r < self.grid_size[0] and 0 <= c < self.grid_size[1]:
                        state = r * self.grid_size[1] + c
                        # Add varied intensity with some randomness
                        belief2[state] = intensity * (0.4 + 0.6 * np.random.random())
        
        # Ensure there are some sharp peaks (in different locations)
        for _ in range(5):
            r = np.random.randint(0, self.grid_size[0])
            c = np.random.randint(0, self.grid_size[1])
            state = r * self.grid_size[1] + c
            belief2[state] = 4.0 + 3.0 * np.random.random()
        
        belief2 = belief2 / np.sum(belief2)
        
        return [belief1, belief2]

    def create_overlapping_plateaus(self):
        """Create overlapping plateau beliefs"""
        belief1 = np.zeros(self.total_states)
        belief2 = np.zeros(self.total_states)
        
        # Agent 1: Left-biased plateau with gradient
        for r in range(5, 15):
            for c in range(3, 13):
                state = r * self.grid_size[1] + c
                # Create gradient from left to right (higher on left)
                gradient = 1.0 - (c - 3) / 15.0
                belief1[state] = max(0.1, gradient)
        belief1 = belief1 / np.sum(belief1)
        
        # Agent 2: Right-biased plateau with gradient
        for r in range(5, 15):
            for c in range(7, 17):
                state = r * self.grid_size[1] + c
                # Create gradient from right to left (higher on right)
                gradient = 1.0 - (17 - c) / 15.0
                belief2[state] = max(0.1, gradient)
        belief2 = belief2 / np.sum(belief2)
        
        return [belief1, belief2]
        
    def create_diffuse_overlapping(self):
        """Create diffuse overlapping beliefs"""
        belief1 = np.zeros(self.total_states)
        belief2 = np.zeros(self.total_states)
        
        # Agent 1: Diffuse distribution centered left
        center1_r, center1_c = 10, 7
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                dist = np.sqrt((r - center1_r)**2 + (c - center1_c)**2)
                if dist < 12:  # Large radius
                    state = r * self.grid_size[1] + c
                    belief1[state] = np.exp(-dist/5)  # Very gradual falloff
        belief1 = belief1 / np.sum(belief1)
        
        # Agent 2: Diffuse distribution centered right
        center2_r, center2_c = 10, 13
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                dist = np.sqrt((r - center2_r)**2 + (c - center2_c)**2)
                if dist < 12:  # Large radius
                    state = r * self.grid_size[1] + c
                    belief2[state] = np.exp(-dist/5)  # Very gradual falloff
        belief2 = belief2 / np.sum(belief2)
        
        return [belief1, belief2]
    
    def create_conflicting_regions(self):
        """Create conflicting beliefs in overlapping regions"""
        belief1 = np.zeros(self.total_states)
        belief2 = np.zeros(self.total_states)
        
        # Create a shared region in the middle
        middle_region = []
        for r in range(7, 13):
            for c in range(7, 13):
                middle_region.append(r * self.grid_size[1] + c)
        
        # Agent 1: Believes target is in the left part of the shared region
        for r in range(7, 13):
            for c in range(7, 10):
                state = r * self.grid_size[1] + c
                # Linear gradient from left to right
                gradient = 1.0 - (c - 7) / 3.0
                belief1[state] = max(0.1, gradient)
        
        # Add some probability outside the shared region
        for r in range(7, 13):
            for c in range(3, 7):
                state = r * self.grid_size[1] + c
                dist = c - 3
                belief1[state] = 0.5 * np.exp(-dist/2)
        
        belief1 = belief1 / np.sum(belief1)
        
        # Agent 2: Believes target is in the right part of the shared region
        for r in range(7, 13):
            for c in range(10, 13):
                state = r * self.grid_size[1] + c
                # Linear gradient from right to left
                gradient = 1.0 - (13 - c) / 3.0
                belief2[state] = max(0.1, gradient)
        
        # Add some probability outside the shared region
        for r in range(7, 13):
            for c in range(13, 17):
                state = r * self.grid_size[1] + c
                dist = c - 13
                belief2[state] = 0.5 * np.exp(-dist/2)
        
        belief2 = belief2 / np.sum(belief2)
        
        return [belief1, belief2]
    
    def create_uncertain_vs_certain(self):
        """Create one certain belief and one uncertain belief that overlap"""
        belief1 = np.zeros(self.total_states)
        belief2 = np.zeros(self.total_states)
        
        # Agent 1: Very diffuse, uncertain belief (high entropy)
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                state = r * self.grid_size[1] + c
                # Add small random noise
                belief1[state] = 0.1 + 0.9 * np.random.random() * np.exp(-(r-10)**2/100 - (c-10)**2/100)
        
        # Ensure higher probability in center region
        for r in range(7, 13):
            for c in range(7, 13):
                state = r * self.grid_size[1] + c
                belief1[state] *= 2.0
                
        belief1 = belief1 / np.sum(belief1)
        
        # Agent 2: More certain belief with a clear structure (low entropy)
        center_r, center_c = 10, 10
        for r in range(5, 15):
            for c in range(5, 15):
                state = r * self.grid_size[1] + c
                dist = np.sqrt((r - center_r)**2 + (c - center_c)**2)
                if dist < 6:
                    # Create a ring structure with peak at radius 3
                    ring_factor = np.abs(dist - 3)
                    belief2[state] = np.exp(-ring_factor**2)
        
        belief2 = belief2 / np.sum(belief2)
        
        return [belief1, belief2]
    
    def merge_beliefs_average(self, beliefs):
        """Simple averaging of beliefs"""
        merged = np.mean(beliefs, axis=0)
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
                
                # Normalize
                belief_i = belief_i / np.sum(belief_i)
                new_beliefs.append(belief_i)
            
            all_beliefs = new_beliefs
        
        merged = np.mean(all_beliefs, axis=0)
        return merged / np.sum(merged)
    
    def merge_beliefs_kl(self, beliefs):
        """KL divergence-based merging with multiple initializations"""
        def kl_divergence(p, q):
            p = np.clip(p, 1e-10, 1)
            q = np.clip(q, 1e-10, 1)
            return np.sum(p * np.log(p / q))
        
        def objective(merged_flat):
            merged = merged_flat.reshape(beliefs[0].shape)
            merged = merged / np.sum(merged)
            
            total_divergence = 0
            for belief in beliefs:
                total_divergence += kl_divergence(belief, merged)
            
            return total_divergence
        
        # Try multiple initial guesses
        initial_guesses = []
        
        # Average of beliefs
        avg_guess = np.mean(beliefs, axis=0)
        avg_guess = avg_guess / np.sum(avg_guess)
        initial_guesses.append(avg_guess)
        
        # Each individual belief
        for belief in beliefs:
            initial_guesses.append(belief.copy())
        
        # Random perturbation of average
        perturbed = avg_guess + np.random.normal(0, 0.01, size=avg_guess.shape)
        perturbed = np.clip(perturbed, 0, None)
        perturbed = perturbed / np.sum(perturbed)
        initial_guesses.append(perturbed)
        
        # Constraints and bounds
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = [(0, 1) for _ in range(len(avg_guess))]
        
        best_result = None
        best_value = np.inf
        
        for init_guess in initial_guesses:
            result = minimize(
                objective,
                init_guess.flatten(),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            
            if result.success and result.fun < best_value:
                best_result = result
                best_value = result.fun
        
        if best_result is not None:
            merged = best_result.x.reshape(avg_guess.shape)
            return merged / np.sum(merged)
        else:
            return avg_guess
        
    def merge_beliefs_reverse_kl(self, beliefs, agent_weights=None):
        """KL divergence-based merging with multiple initializations and optional agent weighting"""
        # Default to equal weights if none provided
        if agent_weights is None:
            agent_weights = np.ones(len(beliefs)) / len(beliefs)
        
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
        
        # Try multiple initial guesses
        initial_guesses = []
        
        # Average of beliefs
        avg_guess = np.mean(beliefs, axis=0)
        avg_guess = avg_guess / np.sum(avg_guess)
        initial_guesses.append(avg_guess)
        
        # Each individual belief
        # for belief in beliefs:
        #     initial_guesses.append(belief.copy())
        
        # Random perturbation of average
        # perturbed = avg_guess + np.random.normal(0, 0.01, size=avg_guess.shape)
        # perturbed = np.clip(perturbed, 0, None)
        # perturbed = perturbed / np.sum(perturbed)
        # initial_guesses.append(perturbed)
        
        # Constraints and bounds
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = [(0, 1) for _ in range(len(avg_guess))]
        
        best_result = None
        best_value = np.inf
        
        for init_guess in initial_guesses:
            result = minimize(
                objective,
                init_guess.flatten(),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            
            if result.success and result.fun < best_value:
                best_result = result
                best_value = result.fun
                print("got best values")
        
        if best_result is not None:            
            merged = best_result.x.reshape(avg_guess.shape)
            return merged / np.sum(merged)
        else:
            return avg_guess
            print("Did not work out")
    
    def merge_beliefs_geometric_mean(self, beliefs):
        """Geometric mean merging"""
        # Small epsilon to avoid zeros
        epsilon = 1e-10
        
        # Element-wise product of all beliefs
        product = np.ones_like(beliefs[0])
        for belief in beliefs:
            product *= np.clip(belief, epsilon, 1)
        
        # Take nth root (geometric mean)
        geo_mean = product ** (1.0 / len(beliefs))
        
        # Normalize
        return geo_mean / np.sum(geo_mean)
    
    def merge_beliefs_log_opinion_pool(self, beliefs):
        """Log opinion pool with adaptively tuned agent weights based on belief quality"""
        epsilon = 1e-10
        
        # Calculate uncertainty (entropy) for each belief
        entropies = []
        for belief in beliefs:
            p = np.clip(belief, epsilon, 1)
            entropy = -np.sum(p * np.log(p))
            entropies.append(entropy)
        
        # Lower entropy = more certain = higher weight
        max_entropy = max(entropies)
        weights = [1.0 - e/max_entropy for e in entropies]
        weights = [w / sum(weights) for w in weights]
        
        # Weighted product of beliefs
        merged = np.ones_like(beliefs[0])
        for i, belief in enumerate(beliefs):
            merged *= np.power(np.clip(belief, epsilon, 1), weights[i])
        
        # Normalize
        merged = merged / np.sum(merged)
        return merged
    
    def merge_beliefs_covariance_intersection(self, beliefs):
        """Covariance Intersection method for belief fusion"""
        def objective(omega):
            # Clip omega to be between 0 and 1
            omega = np.clip(omega, 0, 1)
            
            # Fusion with current omega
            ci_belief = omega * beliefs[0] + (1 - omega) * beliefs[1]
            ci_belief = ci_belief / np.sum(ci_belief)
            
            # Calculate determinant of covariance (use entropy as proxy)
            p = np.clip(ci_belief, 1e-10, 1)
            entropy = -np.sum(p * np.log(p))
            
            # We want to minimize entropy (maximize information)
            return entropy
        
        # Only works for 2 beliefs
        if len(beliefs) != 2:
            return self.merge_beliefs_average(beliefs)
        
        # Find optimal omega between 0 and 1
        result = minimize_scalar(objective, bounds=(0, 1), method='bounded')
        omega_opt = result.x
        
        # Compute final fusion
        merged = omega_opt * beliefs[0] + (1 - omega_opt) * beliefs[1]
        return merged / np.sum(merged)

    def merge_beliefs_jensen_renyi(self, beliefs, alpha=0.5):
        """Minimize Jensen-Rényi divergence for belief fusion"""
        def renyi_entropy(p, alpha):
            p = np.clip(p, 1e-10, 1)
            if alpha == 1:
                return -np.sum(p * np.log(p))
            else:
                return np.log(np.sum(p**alpha)) / (1 - alpha)
        
        def objective(merged_flat):
            merged = merged_flat.reshape(beliefs[0].shape)
            merged = merged / np.sum(merged)
            
            # Calculate Jensen-Rényi divergence
            avg_belief = np.zeros_like(merged)
            for belief in beliefs:
                avg_belief += belief / len(beliefs)
            
            # Jensen-Rényi divergence
            jr_div = renyi_entropy(avg_belief, alpha)
            for belief in beliefs:
                jr_div -= renyi_entropy(belief, alpha) / len(beliefs)
            
            return jr_div
        
        # Multiple initial guesses
        initial_guesses = [np.mean(beliefs, axis=0)]
        for belief in beliefs:
            initial_guesses.append(belief.copy())
        
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = [(0, 1) for _ in range(len(beliefs[0]))]
        
        best_result = None
        best_value = np.inf
        
        for init_guess in initial_guesses:
            result = minimize(
                objective,
                init_guess / np.sum(init_guess),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            
            if result.success and result.fun < best_value:
                best_result = result
                best_value = result.fun
        
        if best_result is not None:
            merged = best_result.x.reshape(beliefs[0].shape)
            return merged / np.sum(merged)
        else:
            return initial_guesses[0] / np.sum(initial_guesses[0])
        
    def merge_beliefs_max_confidence(self, beliefs):
        """Maximum confidence method - takes the max probability for each state"""
        # Find the agent with highest confidence for each state
        merged = np.zeros_like(beliefs[0])
        
        for state in range(len(merged)):
            # Get beliefs for this state from all agents
            state_beliefs = [belief[state] for belief in beliefs]
            # Take the maximum value
            merged[state] = max(state_beliefs)
        
        # Normalize
        return merged / np.sum(merged)
    
    def merge_beliefs_weighted_avg(self, beliefs):
        """Weighted average based on belief entropy"""
        # Calculate entropy for each belief
        entropies = []
        for belief in beliefs:
            p = np.clip(belief, 1e-10, 1)
            entropy = -np.sum(p * np.log(p))
            entropies.append(entropy)
        
        # Convert to confidence (lower entropy = higher confidence)
        confidences = [1.0 / (e + 0.1) for e in entropies]  # Avoid division by zero
        total_confidence = sum(confidences)
        weights = [c / total_confidence for c in confidences]
        
        print(f"Belief entropies: {entropies}")
        print(f"Weights for merging: {weights}")
        
        # Weighted average
        weighted_avg = np.zeros_like(beliefs[0])
        for i, belief in enumerate(beliefs):
            weighted_avg += weights[i] * belief
        
        return weighted_avg / np.sum(weighted_avg)
    
    def compare_methods(self, beliefs, test_name):
        
        """Compare different belief merging methods"""
        methods = {
            "Simple Average": self.merge_beliefs_average,
            "Consensus": self.merge_beliefs_consensus,
            "KL Divergence": self.merge_beliefs_kl,
            "Reverse KL": self.merge_beliefs_reverse_kl,  # New method
            "Log Opinion Pool": self.merge_beliefs_log_opinion_pool,  # New method
            "Max Confidence": self.merge_beliefs_max_confidence,  # New method
            "Geometric Mean": self.merge_beliefs_geometric_mean
        }
        
        # Add covariance intersection only for 2 beliefs
        if len(beliefs) == 2:
            methods["Covariance Intersection"] = self.merge_beliefs_covariance_intersection
        
        results = {}
        
        print(f"\nComparing belief merging methods for test: {test_name}")
        
        for name, method in methods.items():
            start_time = time.time()
            merged = method(beliefs)
            elapsed = time.time() - start_time
            results[name] = merged
            print(f"  - {name}: {elapsed:.4f} seconds")
        
        # Calculate pairwise distances
        print("\nPairwise Jensen-Shannon Divergence:")
        method_names = list(results.keys())
        for i in range(len(method_names)):
            for j in range(i+1, len(method_names)):
                name1 = method_names[i]
                name2 = method_names[j]
                js_div = self.jensen_shannon_divergence(results[name1], results[name2])
                print(f"  - {name1} vs {name2}: {js_div:.6f}")
        
        # Check objective values
        print("\nKL Divergence Objective Values:")
        for name, merged in results.items():
            obj_value = self.kl_objective(beliefs, merged)
            print(f"  - {name}: {obj_value:.6f}")
            
        # Calculate entropy of each merged belief
        print("\nEntropy of Merged Beliefs:")
        for name, merged in results.items():
            p = np.clip(merged, 1e-10, 1)
            entropy = -np.sum(p * np.log(p))
            print(f"  - {name}: {entropy:.6f}")
        
        self.visualize_results(beliefs, results, test_name)
        
        return results
    
    def jensen_shannon_divergence(self, p, q):
        """Calculate Jensen-Shannon divergence between two distributions"""
        p = np.clip(p, 1e-10, 1)
        q = np.clip(q, 1e-10, 1)
        m = 0.5 * (p + q)
        return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))
    
    def kl_objective(self, beliefs, merged):
        """Calculate the KL divergence objective value"""
        total = 0
        merged = np.clip(merged, 1e-10, 1)
        for belief in beliefs:
            belief = np.clip(belief, 1e-10, 1)
            total += np.sum(belief * np.log(belief / merged))
        return total
    
    def visualize_results(self, beliefs, merged_beliefs, test_name):
        """Visualize agent beliefs and merged results"""
        rows, cols = self.grid_size
        n_methods = len(merged_beliefs)
        
        # Create figure for agent beliefs
        n_agents = len(beliefs)
        fig_agents = plt.figure(figsize=(10, 5 * n_agents))
        
        # Plot agent beliefs
        for i, belief in enumerate(beliefs):
            ax = fig_agents.add_subplot(n_agents, 1, i+1)
            belief_grid = belief.reshape(rows, cols)
            im = ax.imshow(belief_grid, cmap='hot', interpolation='nearest')
            ax.set_title(f'Agent {i+1} Belief')
            fig_agents.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig(f'agent_beliefs_{test_name}.png', dpi=300)
        
        # Create figure for merged beliefs
        fig_merged = plt.figure(figsize=(15, 5 * ((n_methods+1) // 2)))
        
        # Plot each merged belief
        for i, (method_name, merged_belief) in enumerate(merged_beliefs.items()):
            ax = fig_merged.add_subplot((n_methods+1) // 2, 2, i+1)
            merged_grid = merged_belief.reshape(rows, cols)
            im = ax.imshow(merged_grid, cmap='hot', interpolation='nearest')
            ax.set_title(f'Merged Belief: {method_name}')
            fig_merged.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig(f'merged_beliefs_{test_name}.png', dpi=300)
        plt.show()

def run_overlapping_tests():
    tester = OverlappingBeliefTester()

    print("=" * 80)
    print("TEST 1: OVERLAPPING PLATEAUS")
    print("=" * 80)
    beliefs = tester.create_overlapping_plateaus()
    tester.compare_methods(beliefs, "plateaus")
    
    print("\n" + "=" * 80)
    print("TEST 2: DIFFUSE OVERLAPPING BELIEFS")
    print("=" * 80)
    beliefs = tester.create_diffuse_overlapping()
    tester.compare_methods(beliefs, "diffuse")
    
    print("\n" + "=" * 80)
    print("TEST 3: CONFLICTING REGIONS")
    print("=" * 80)
    beliefs = tester.create_conflicting_regions()
    tester.compare_methods(beliefs, "conflict")
    
    print("\n" + "=" * 80)
    print("TEST 4: UNCERTAIN VS CERTAIN")
    print("=" * 80)
    beliefs = tester.create_uncertain_vs_certain()
    tester.compare_methods(beliefs, "uncertain")

       
    print("\n" + "=" * 80)
    print("TEST 5: SCATTERED IRREGULAR BELIEFS")
    print("=" * 80)
    beliefs = tester.create_scattered_irregular_beliefs()
    tester.compare_methods(beliefs, "scattered")

if __name__ == "__main__":
    run_overlapping_tests()