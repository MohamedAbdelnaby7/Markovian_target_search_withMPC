
from TargetSearchMPC import *
from TargetSearchMCTS import *
from EnvironmentMCTS import *
    
# Example usage
if __name__ == "__main__":
    # simulator = TargetSearchMPC(grid_size=(50,50), n_agents=3, horizon=2, fast_mode=False)
    # # Choose objective: "entropy", "greedy_map", or "variance_reducing"
    # trajectories = simulator.run_simulation(steps=3900, objective_type="entropy")
    # simulator.visualize_trajectories()
    # print("Simulation complete. Check plots for trajectories.")
    
    # # Print final results
    # print("Final belief distribution:")
    # print(simulator.belief.reshape(50,50))
    # print(f"True target position: {divmod(simulator.true_position, 50)}")

    # Initialize Environment
    GRID_SIZE = (50, 50)  # Grid dimensions (rows, cols)
    N_AGENTS = 3  # Number of searching agents
    SIMULATIONS = 20  # Number of MCTS simulations per decision
    HORIZON = 3  # Rollout depth for MCTS
    STEPS = 100  # Number of time steps for the search
    DETECTION_THRESHOLD = 0.8  # Confidence threshold for stopping

    # Create the environment
    env = SearchEnvironment(grid_size=GRID_SIZE, n_agents=N_AGENTS)

    # Initialize MCTS Planner
    mcts = MCTSPlanner(env, horizon=HORIZON, simulations=SIMULATIONS)

    # Store the initial state for visualization
    initial_agent_positions = env.agent_positions
    initial_target_position = env.true_position

    print(f" Starting MCTS Search with {N_AGENTS} agents on a {GRID_SIZE} grid.")
    print(f" Initial agent positions: {initial_agent_positions}")
    print(f" Initial target position: {initial_target_position}")

    # Run the MCTS search loop
    for step in range(STEPS):
        print(f"\n Step {step + 1}")

        # Run MCTS planning to get the best actions for all agents
        best_action = mcts(env.agent_positions, num_simulations=SIMULATIONS)
        env.update_agents(best_action)  # Move agents based on MCTS decisions

        # Move the target randomly
        env.move_target()

        # Simulate sensor observations
        obs = env.simulate_observation(env.agent_positions, HORIZON)

        # Check if the target is found
        if env.is_terminal(env.agent_positions):
            print(f" Target detected at step {step + 1}!")
            break

    print("\n Search complete!")

    # Visualize the final search results
    visualize_trajectories(env.grid_size, None, None, env.trajectories)