from TargetSearchMPC import *
from TargetSearchMCTS import *
from Environment import *
from utils import *

# Example usage
if __name__ == "__main__":
    # Initialize Environment
    GRID_SIZE = (50, 50)  # Grid dimensions (rows, cols)
    N_AGENTS = 3  # Number of searching agents
    HORIZON = 3  # Rollout depth for MCTS and MPC
    DETECTION_THRESHOLD = 0.8  # Confidence threshold for stopping
    STEPS = 9000  # Number of time steps for the search

    env_genterative = SearchEnvironment(grid_size=GRID_SIZE, n_agents=N_AGENTS,
                 target_mdp = True)
    for step in range(STEPS):
        env_genterative.move_target()
    generated_target = env_genterative.trajectories["target"]
    env_mpc = SearchEnvironment(grid_size=GRID_SIZE, n_agents=N_AGENTS, target_mdp=True)
    simulator = TargetSearchMPC(env_mpc, horizon=HORIZON, fast_mode=False)
    # Choose objective: "entropy", "greedy_map", or "variance_reducing"
    trajectories = simulator.run_simulation(steps=3900, objective_type="entropy")
    simulator.visualize_trajectories()
    print("Simulation complete. Check plots for trajectories.")
    
    # Print final results
    print("Final belief distribution:")
    print(simulator.belief.reshape(50,50))
    print(f"True target position: {divmod(simulator.true_position, 50)}")

    
    SIMULATIONS = 30  # Number of MCTS simulations per decision
    gaussian_center = (25,25)

    # Create the environment
    env_mcts = SearchEnvironment(grid_size=GRID_SIZE, n_agents=N_AGENTS)
    #env_mcts = SearchEnvironment(grid_size=GRID_SIZE, n_agents=N_AGENTS, gaussian_bias=True,heatmap_center=gaussian_center)
    
    # Initialize MCTS Planner
    mcts = MCTSPlanner(env_mcts, horizon=HORIZON, simulations=SIMULATIONS)
    #mcts = MCTSPlanner(env_mcts, horizon=HORIZON, simulations=SIMULATIONS, gaussian_bias=True, heatmap_center=gaussian_center)

    # Store the initial state for visualization
    initial_agent_positions = env_mcts.agent_positions
    initial_target_position = env_mcts.true_position

    print(f" Starting MCTS Search with {N_AGENTS} agents on a {GRID_SIZE} grid.")
    print(f" Initial agent positions: {initial_agent_positions}")
    print(f" Initial target position: {initial_target_position}")

    # Run the MCTS search loop
    for step in range(STEPS):
        print(f"\n Step {step + 1}")

        # Run MCTS planning to get the best actions for all agents
        best_action = mcts(env_mcts.agent_positions, num_simulations=SIMULATIONS)
        env_mcts.update_agents(best_action)  # Move agents based on MCTS decisions

        # Move the target randomly
        #env_mcts.move_target()
        env_mcts.move_target(step)
        # Check if the target is found
        if env_mcts.is_terminal(env_mcts.agent_positions):
            print(f" Target detected at step {step + 1}!")
            break

    print("\n Search complete!")

    # Visualize the final search results
    #visualize_trajectories_MCTS(env.grid_size, env.trajectories)
    animate_trajectories(GRID_SIZE, env_mcts.trajectories, interval=600)