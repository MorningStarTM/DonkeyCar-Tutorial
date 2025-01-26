import os
from actor_critic import ActorCritic
import gym
import gym_donkeycar
import numpy as np
import torch




class Trainer:
    def __init__(self, config, env) -> None:
        self.config = config
        self.agent = ActorCritic(self.config)
        self.env = env

    
    def train(self, epochs):
        random_seed = 543
        
        torch.manual_seed(random_seed)

        running_reward = 0
        
        for i_episode in range(0, epochs):
            state = self.env.reset()
            for t in range(epochs):
                action = self.agent(state)
                state, reward, done, _ = self.env.step(action)
                self.agent.rewards.append(reward)
                running_reward += reward

                if done:
                    break

            self.agent.optimizer.zero_grad()
            loss = self.agent.calculate_loss()
            loss.backward()
            self.agent.optimizer.step()
            self.agent.clearMemory()

            






"""

gamma = 0.99
lr = 0.02
betas = (0.9, 0.999)

"""