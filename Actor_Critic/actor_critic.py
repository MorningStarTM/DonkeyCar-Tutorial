import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical




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
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.flatten_size = 512 * 8 * 8


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
            nn.Linear(32, self.config['actions'])
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
        x = x.view(x.size(0), -1) 

        action_logits = self.actor(x)  
        state_value = self.critic(x)   

        action_probs = F.softmax(action_logits)
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        
        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)
        
        return action.item()
    

    def calculate_loss(self, gamma=0.9):
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)

        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std())

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

