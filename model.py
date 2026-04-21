import torch
from torch import nn
import numpy as np
from torch.distributions import Normal

LOG_STD_MIN, LOG_STD_MAX = -5, 2


class Critic(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = int(np.prod(env.single_observation_space.shape))
        act_dim = int(np.prod(env.single_action_space.shape))
        self.network = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs, action):
        return self.network(torch.cat([obs, action], dim=-1))


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = int(np.prod(env.single_observation_space.shape))
        action_dim = int(np.prod(env.single_action_space.shape))
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.mean_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

        high = env.single_action_space.high
        low = env.single_action_space.low
        self.register_buffer('scale', torch.tensor((high - low) / 2.0, dtype=torch.float32))
        self.register_buffer('bias', torch.tensor((high + low) / 2.0, dtype=torch.float32))

    def forward(self, obs):
        x = self.network(obs)
        mean = self.mean_head(x)
        log_std = torch.clamp(self.log_std_head(x), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        u = normal.rsample()
        y = torch.tanh(u)
        action = self.scale * y + self.bias

        # stable: log(1 - tanh(u)^2) = 2*(log(2) - u - softplus(-2u))
        log_prob = normal.log_prob(u) - 2 * (np.log(2) - u - nn.functional.softplus(-2 * u))
        log_prob = log_prob.sum(dim=-1, keepdim=True) - torch.log(self.scale).sum()
        return action, log_prob
