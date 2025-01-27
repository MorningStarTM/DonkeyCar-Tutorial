import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal




class ActorCritic(nn.Module):
    def __init__(self, config) -> None:
        super(ActorCritic, self).__init__()

        self.config = config

        self.logprobs = []
        self.state_values = []
        self.rewards = []

        # Shared convolutional base
        self.shared_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        self.flatten_size = 256 * 3 * 5

        self.actor = nn.Sequential(
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2* self.config['actions'])
        )

        self.critic = nn.Sequential(
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.Adam(self.parameters(), lr=self.config['lr'])
        self.to(self.device)

    
    def forward(self, observation):
        x = self.shared_conv(observation) 
        print(x.shape)
        x = x.view(x.size(0), -1) 

        action_output = self.actor(x)  
        state_value = self.critic(x)   

        # Split actor output into mean and log std for Gaussian distribution
        mean, log_std = action_output[:, :self.config['actions']], action_output[:, self.config['actions']:]
        std = torch.exp(log_std)  # Convert log std to std

        self.state_values.append(state_value)

        return mean, std, state_value
    

    def act(self, observation):
        mean, std, state_value = self.forward(observation)
        dist = Normal(mean, std)

        # Sample action and calculate log probability
        action = dist.rsample()  # Reparameterization trick
        log_prob = dist.log_prob(action).sum(-1)  # Sum over action dimensions

        # Scale actions to match the environment's action space
        low = torch.tensor(self.config['action_low'], dtype=torch.float32, device=self.device)
        high = torch.tensor(self.config['action_high'], dtype=torch.float32, device=self.device)
        scaled_action = low + (0.5 * (action + 1.0) * (high - low))  # Scale to [low, high]

        # Clip actions to be within the valid range
        clipped_action = torch.clamp(scaled_action, low, high)

        self.logprobs.append(log_prob)
        self.state_values.append(state_value)

        return clipped_action, log_prob


    

    def calculate_loss(self, gamma=0.9):
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8) # add small value to prevent from zero

        loss = 0

        for logprob, value, reward in zip(self.logprobs, self.state_values, rewards):
            advantage = reward - value.item()
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value, reward)    
            loss += (action_loss + value_loss)
        
        return loss
    

    def clearMemory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]


    def save_model(self, model_path):
        torch.save(self.state_dict(), model_path)
        print(f"model saved at {model_path}")

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))
        print(f"model loaded from {model_path}")

