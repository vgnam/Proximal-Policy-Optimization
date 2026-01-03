import torch
import torch.nn as nn
import torch.distributions as D


class ActorCritic(nn.Module):
    def __init__(self, obs_shape, action_dim, discrete=True):
        super().__init__()
        self.discrete = discrete
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # For MLP, we expect the observation shape to be a flattened 1D vector
        input_dim = obs_shape  # The input dimension is the size of the flattened observation

        # Shared MLP Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )

        # Actor Head
        if discrete:
            self.actor = nn.Sequential(
                nn.Linear(8, 8),
                nn.ReLU(),
                nn.Linear(8, action_dim)
            )
        else:
            self.actor_mean = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, action_dim)
            )
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic Head
        self.critic = nn.Sequential(
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

        self.to(self.device)

    def forward(self, obs):
        # Normalize image or input vector if needed
        obs = obs / 255.0 if obs.max() > 1 else obs  # Normalize to [0, 1] if required

        # Pass the observation through the encoder (MLP)
        feat = self.encoder(obs)

        # Actor: Discrete or Continuous
        if self.discrete:
            logits = self.actor(feat)
            dist = D.Categorical(logits=logits)
        else:
            mean = self.actor_mean(feat)
            std = self.actor_log_std.exp()
            dist = D.Normal(mean, std)

        # Critic: Value function
        value = self.critic(feat).squeeze(-1)
        return dist, value
