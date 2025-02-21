import matplotlib.pyplot as plt

def visualize_trajectories(self, initial_belief):
        """Visualize agent and target trajectories with start/end beliefs"""
        plt.figure(figsize=(12, 6))
        
        # Plot initial belief
        plt.subplot(1, 2, 1)
        plt.imshow(initial_belief.reshape(self.grid_size), cmap='Reds')
        plt.title("Initial Belief")
        plt.colorbar()
        
        # Plot final belief
        plt.subplot(1, 2, 2)
        plt.imshow(self.belief.reshape(self.grid_size), cmap='Reds')
        plt.title("Final Belief")
        plt.colorbar()
        
        # Create trajectory plot
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        
        # Convert positions to coordinates
        target_path = [divmod(s, self.grid_size[1]) for s in self.trajectories['target']]
        agent_paths = [[divmod(s, self.grid_size[1]) for s in path] 
                      for path in self.trajectories['agents']]
        
        # Plot target trajectory
        t_x, t_y = zip(*target_path)
        plt.plot(t_y, t_x, 'b-', marker='o', label='Target Path')
        
        # Plot agent trajectories
        colors = ['green', 'purple', 'orange', 'red']
        for i, path in enumerate(agent_paths):
            a_x, a_y = zip(*path)
            plt.plot(a_y, a_x, colors[i], 
                    label=f'Agent {i+1} Path')
        
        plt.xticks(range(self.grid_size[1]))
        plt.yticks(range(self.grid_size[0]))
        plt.grid(True)
        plt.title("Agent and Target Trajectories")
        plt.legend()
        plt.show()