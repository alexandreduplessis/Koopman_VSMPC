import gym
import numpy as np
from vs_env import VsEnv

my_env = VsEnv(length=100, width=100, max_steps=10)
my_env.reset()
max_steps = 10
done = False
while done == False:
    _, _, done, _ = my_env.step({"abscisse": np.array([1.], dtype=np.float32), "ordonnee": np.array([1.], dtype=np.float32), "depth": np.array([.2], dtype=np.float32)})
    my_env.render()