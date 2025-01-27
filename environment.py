import os
import gym
import gym_donkeycar
import numpy as np
from const import *
from Actor_Critic.actor_critic import ActorCritic
import torch


# Define the configuration
config = {
    'actions': 2,
    'lr': 1e-4,
    'action_low': [-1, 0],
    'action_high': [1, 1], 
}

model = ActorCritic(config)

exe_path = f"{PATH}/donkey_sim.exe"
port = 9091

conf = { "exe_path" : exe_path, "port" : port }

env = gym.make("donkey-generated-track-v0", conf=conf)

# PLAY
obs = env.reset()
for t in range(100):
  obs_tensor = torch.tensor(np.transpose(obs, (2, 0, 1)), dtype=torch.float32).unsqueeze(0).to(model.device)

  action_logits, state_value = model.act(obs_tensor)
  #action = np.array([0.0, 0.5]) # drive straight with small speed
  # execute the action
  action = np.array(action_logits.cpu().tolist()[0])
  obs, reward, done, info = env.step(action)

# Exit the scene
env.close()