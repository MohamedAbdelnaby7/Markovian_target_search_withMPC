import numpy as np
from scipy.optimize import minimize

# Define KL divergence function
def kl_divergence(p, q):
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    return np.sum(p * np.log(p / q))

# Define the belief merging optimization function
def merge_beliefs(agent_beliefs, agent_weights):
    num_states = agent_beliefs.shape[1]

    def objective(merged_belief):
        merged_belief = np.clip(merged_belief, 1e-10, 1)
        merged_belief /= np.sum(merged_belief)
        divergence = 0
        for belief, weight in zip(agent_beliefs, agent_weights):
            divergence += weight * kl_divergence(belief, merged_belief)
        return divergence

    # Initial guess: average beliefs
    initial_guess = np.mean(agent_beliefs, axis=0)

    # Constraints: merged belief must sum to 1
    constraints = ({'type': 'eq', 'fun': lambda b: np.sum(b) - 1})
    bounds = [(0, 1) for _ in range(num_states)]

    result = minimize(objective, initial_guess, bounds=bounds,
                      constraints=constraints, method='SLSQP')

    if result.success:
        return result.x / np.sum(result.x)
    else:
        raise ValueError("Optimization failed: " + result.message)


# Agent class definition
class Agent:
    def __init__(self, id, num_states, initial_belief, position):
        self.id = id
        self.belief = initial_belief
        self.position = position

    def update_belief(self, observation_model):
        self.belief *= observation_model
        self.belief /= np.sum(self.belief)


# Simulation environment class
class Simulation:
    def __init__(self, num_agents, num_states, proximity_distance):
        self.agents = []
        self.num_states = num_states
        self.proximity_distance = proximity_distance

        for i in range(num_agents):
            belief = np.random.dirichlet(np.ones(num_states))
            position = np.random.uniform(0, 10, size=2)
            agent = Agent(i, num_states, belief, position)
            self.agents.append(agent)

    def distance(self, agent1, agent2):
        return np.linalg.norm(agent1.position - agent2.position)

    def communicate_and_merge(self):
        # Find groups of agents within proximity
        communication_groups = []
        visited = set()
        for agent in self.agents:
            if agent.id in visited:
                continue

            group = [agent]
            visited.add(agent.id)

            for other_agent in self.agents:
                if other_agent.id not in visited:
                    if self.distance(agent, other_agent) <= self.proximity_distance:
                        group.append(other_agent)
                        visited.add(other_agent.id)

            if len(group) > 1:
                communication_groups.append(group)

        # Merge beliefs within each communication group
        for group in communication_groups:
            agent_beliefs = np.array([agent.belief for agent in group])
            agent_weights = np.ones(len(group))  # Can be modified to represent confidence

            merged_belief = merge_beliefs(agent_beliefs, agent_weights)

            for agent in group:
                agent.belief = merged_belief

    def step(self):
        # Agents update beliefs independently with local observation
        for agent in self.agents:
            observation_model = np.random.dirichlet(np.ones(self.num_states))
            agent.update_belief(observation_model)

        # Communication and merging step
        self.communicate_and_merge()


# Testing the framework
def test_simulation():
    np.random.seed(42)

    num_agents = 10
    num_states = 5
    proximity_distance = 3.0

    sim = Simulation(num_agents, num_states, proximity_distance)

    print("Initial agent beliefs:")
    for agent in sim.agents:
        print(f"Agent {agent.id} belief: {agent.belief}")

    sim.step()

    print("\nAgent beliefs after one step:")
    for agent in sim.agents:
        print(f"Agent {agent.id} belief: {agent.belief}")


if __name__ == "__main__":
    test_simulation()
