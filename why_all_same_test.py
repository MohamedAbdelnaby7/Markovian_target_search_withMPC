# belief_merging_test.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import minimize
import time
import random

class BeliefMergingTester:
    def __init__(self):
        """Initialize the tester class"""
        self.grid_size = (20, 20)
        self.total_states = self.grid_size[0] * self.grid_size[1]
        self.target_position = self.grid_size[0] * self.grid_size[1] // 2  # Center position
    
    def create_conflicting_beliefs(self):
        """Create agents with explicitly conflicting beliefs"""
        belief1 = np.zeros(self.total_states)
        belief2 = np.zeros(self.total_states)
        
        # Agent 1 believes target is on the left
        for r in range(8, 12):
            for c in range(5, 9):
                state = r * self.grid_size[1] + c
                dist = np.sqrt((r - 10)**2 + (c - 7)**2)
                belief1[state] = np.exp(-dist)
        belief1 = belief1 / np.sum(belief1)
        
        # Agent 2 believes target is on the right
        for r in range(8, 12):
            for c in range(11, 15):
                state = r * self.grid_size[1] + c
                dist = np.sqrt((r - 10)**2 + (c - 13)**2)
                belief2[state] = np.exp(-dist)
        belief2 = belief2 / np.sum(belief2)
        
        return [belief1, belief2]
    
    def create_different_confidence_beliefs(self):
        """Create agents with different confidence levels"""
        belief1 = np.zeros(self.total_states)
        belief2 = np.zeros(self.total_states)
        
        # Agent 1: Very high confidence but wrong location
        for r in range(5, 8):
            for c in range(5, 8):
                state = r * self.grid_size[1] + c
                dist = np.sqrt((r - 6.5)**2 + (c - 6.5)**2)
                belief1[state] = np.exp(-dist/0.5)  # Very concentrated (high confidence)
        belief1 = belief1 / np.sum(belief1)
        
        # Agent 2: Lower confidence but correct location
        for r in range(8, 13):
            for c in range(8, 13):
                state = r * self.grid_size[1] + c
                dist = np.sqrt((r - 10)**2 + (c - 10)**2)
                belief2[state] = np.exp(-dist/2.0)  # More diffuse (lower confidence)
        belief2 = belief2 / np.sum(belief2)
        
        return [belief1, belief2]
    
    def create_different_shape_beliefs(self):
        """Create agents with different belief shapes"""
        belief1 = np.zeros(self.total_states)
        belief2 = np.zeros(self.total_states)
        
        # Agent 1: Gaussian shape
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                state = r * self.grid_size[1] + c
                dist = np.sqrt((r - 10)**2 + (c - 10)**2)
                belief1[state] = np.exp(-dist**2/20)
        belief1 = belief1 / np.sum(belief1)
        
        # Agent 2: Uniform band
        for r in range(8, 13):
            for c in range(self.grid_size[1]):
                state = r * self.grid_size[1] + c
                belief2[state] = 1.0
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
    
    def merge_beliefs_kl_original(self, beliefs):
        """Original KL divergence-based merging"""
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
        
        # Initial guess: average of beliefs
        initial_guess = np.mean(beliefs, axis=0)
        initial_guess = initial_guess / np.sum(initial_guess)
        
        # Constraints and bounds
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = [(0, 1) for _ in range(len(initial_guess))]
        
        # Perform optimization
        result = minimize(
            objective,
            initial_guess.flatten(),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-8}
        )
        
        if result.success:
            merged = result.x.reshape(initial_guess.shape)
            return merged / np.sum(merged)
        else:
            return initial_guess
    
    def merge_beliefs_kl_improved(self, beliefs):
        """Improved KL divergence merging with better optimization"""
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
        
        # Try multiple initial guesses to avoid local minima
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
    
    def merge_beliefs_geometric_mean(self, beliefs):
        """Geometric mean merging (another approach)"""
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
        
        # Weighted average
        weighted_avg = np.zeros_like(beliefs[0])
        for i, belief in enumerate(beliefs):
            weighted_avg += weights[i] * belief
        
        return weighted_avg / np.sum(weighted_avg)
    
    def compare_methods(self, beliefs, methods=None, show_plots=True):
        """Compare different belief merging methods"""
        if methods is None:
            methods = {
                "Simple Average": self.merge_beliefs_average,
                "Consensus": self.merge_beliefs_consensus,
                "KL Divergence (Original)": self.merge_beliefs_kl_original,
                "KL Divergence (Improved)": self.merge_beliefs_kl_improved,
                "Geometric Mean": self.merge_beliefs_geometric_mean,
                "Weighted Average": self.merge_beliefs_weighted_avg
            }
        
        results = {}
        
        print(f"\nComparing {len(methods)} belief merging methods:")
        
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
                if js_div < 1e-6:
                    print(f"    WARNING: Methods are producing nearly identical results!")
        
        # Check objective values
        print("\nKL Divergence Objective Values:")
        for name, merged in results.items():
            obj_value = self.kl_objective(beliefs, merged)
            print(f"  - {name}: {obj_value:.6f}")
        
        if show_plots:
            self.visualize_results(beliefs, results)
        
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
    
    def visualize_results(self, beliefs, merged_beliefs):
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
        plt.savefig('agent_beliefs_test.png', dpi=300)
        
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
        plt.savefig('merged_beliefs_test.png', dpi=300)
        plt.show()

def run_tests():
    tester = BeliefMergingTester()
    
    print("=" * 80)
    print("TEST 1: CONFLICTING BELIEFS")
    print("=" * 80)
    beliefs = tester.create_conflicting_beliefs()
    tester.compare_methods(beliefs)
    
    print("\n" + "=" * 80)
    print("TEST 2: DIFFERENT CONFIDENCE LEVELS")
    print("=" * 80)
    beliefs = tester.create_different_confidence_beliefs()
    tester.compare_methods(beliefs)
    
    print("\n" + "=" * 80)
    print("TEST 3: DIFFERENT BELIEF SHAPES")
    print("=" * 80)
    beliefs = tester.create_different_shape_beliefs()
    tester.compare_methods(beliefs)
    
    print("\n" + "=" * 80)
    print("TEST 4: TESTING OPTIMIZATION PARAMETERS")
    print("=" * 80)
    # Create a test case where KL optimization should make a difference
    print("Creating test case with intentionally conflicting beliefs...")
    
    # Two agents with concentrated peaks at different locations
    belief1 = np.zeros(tester.total_states)
    belief2 = np.zeros(tester.total_states)
    
    # First agent: very confident but wrong
    pos1 = 5 * tester.grid_size[1] + 5
    belief1[pos1] = 0.95
    # Add some small probability elsewhere
    for i in range(10):
        idx = random.randint(0, tester.total_states - 1)
        if idx != pos1:
            belief1[idx] = 0.005
    belief1 = belief1 / np.sum(belief1)
    
    # Second agent: less confident but correct
    pos2 = 10 * tester.grid_size[1] + 10
    belief2[pos2] = 0.6
    # Add probability nearby
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            r, c = 10 + dr, 10 + dc
            if 0 <= r < tester.grid_size[0] and 0 <= c < tester.grid_size[1]:
                idx = r * tester.grid_size[1] + c
                if idx != pos2:
                    belief2[idx] = 0.01
    belief2 = belief2 / np.sum(belief2)
    
    beliefs = [belief1, belief2]
    
    # Try different optimization methods
    methods = {
        "Simple Average": tester.merge_beliefs_average,
        "Consensus": tester.merge_beliefs_consensus,
        "KL (SLSQP)": tester.merge_beliefs_kl_original,
        "KL (Multiple Starts)": tester.merge_beliefs_kl_improved,
        "Geometric Mean": tester.merge_beliefs_geometric_mean
    }
    
    tester.compare_methods(beliefs, methods)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("If all methods are still producing identical results, consider:")
    print("1. The KL divergence objective may have a very flat landscape")
    print("2. For certain belief patterns, average might be mathematically optimal")
    print("3. Geometric mean provides a fundamentally different approach")
    print("\nCheck the objective values to see if the methods are truly equivalent.")

if __name__ == "__main__":
    run_tests()