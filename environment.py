import random
import math
import torch

# random.seed(2000)

def env_function(state, control):
    return (state**(1/2)+control)**2
    # return 0.22*state - 0.3*control

class SimulationEnv():
    """Basic simulation environment with user control and random control implemented"""
    def __init__(self, obs_dim):
        self.state = []
        self.current_state = None
        self.control = []
        self.obs_dim = obs_dim
    
    def env_initialize(self, state):
        self.state = [state]
        self.control = []
        self.current_state = state
        return self.state
    
    def env_random_control(self, device):
        new_state = torch.tensor([201.]*self.obs_dim)
        
        while new_state.mean() > 200. or new_state.mean() < -200.:
            control = torch.tensor([random.uniform(-50,50) for _ in range(self.obs_dim)]).to(device)
            
            new_state = env_function(self.current_state, control)
            
        self.control.append(control)
        self.current_state = new_state
        self.state.append(self.current_state)
        return self.current_state, self.control[-1]
    
    def env_control(self, control):
        self.control.append(control)
        self.current_state = env_function(self.current_state, control)
        self.state.append(self.current_state)
        return self.current_state