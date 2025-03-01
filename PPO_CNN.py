import gymnasium as gym
import os
from itertools import count
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm
from torchvision import transforms

class GAE:
    def __init__(self, gamma, lambda_):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def compute(self, rewards, values, masks):
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        values = torch.tensor(values, dtype=torch.float32, device=self.device)
        masks = torch.tensor(masks, dtype=torch.float32, device=self.device)

        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards) - 1)):
            delta = rewards[t] + self.gamma * values[t + 1] * masks[t] - values[t]
            gae = delta + self.gamma * self.lambda_ * masks[t] * gae
            advantages[t] = gae

        return advantages

class PPO_CNN:
    def __init__(self, env, actor, critic, transform, discrete=True, lr_actor=3e-4, lr_critic=3e-4, gamma=0.9, lambda_=0.95,
                 epsilon=0.2, c=0.01):
        self.env = env
        self.discrete = discrete  # Flag to handle discrete vs. continuous action spaces
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.optimizerA = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizerC = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.advantages = GAE(gamma, lambda_)
        self.c = c
        self.transform = transform
        self.records_history = []

    def compute_returns(self, next_value, rewards, masks):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * masks[step]
            returns.insert(0, R)
        return torch.cat(returns).detach()

    def clip_advantage(self, advantage):
        return torch.where(
            advantage >= 0,
            (1 + self.epsilon) * advantage,
            (1 - self.epsilon) * advantage
        )

    def train(self, n_iters, max_timesteps=1000, n_ppo_updates=500):
        for iter in range(n_iters):
            states, actions, rewards, masks, old_log_probs, values = [], [], [], [], [], []
            _, _ = self.env.reset()
            state = torch.tensor(self.env.render()).permute(2, 0, 1)  # Get initial state (RGB image)
            state = self.transform(state)  # Apply transform to the state (image processing)
            state = torch.FloatTensor(state).to(self.device)  # Add batch dimension and move to device

            total_reward = 0

            for i in count():
                state_tensor = state.unsqueeze(0).to(self.device)

                # Sample action based on action space type
                if self.discrete:
                    dist = self.actor(state_tensor)  # Categorical for discrete actions
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                else:
                    dist = self.actor(state_tensor)  # Normal for continuous actions
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)  # Sum over dimensions

                # Step in the environment

                _, reward, done, _, _ = self.env.step(action.item() if self.discrete else action.cpu().numpy())

                # Get next state (RGB image)
                next_state = torch.tensor(self.env.render()).permute(2, 0, 1)
                next_state = self.transform(next_state)  # Apply transform
                next_state = torch.FloatTensor(next_state).to(self.device)  # Add batch dimension and move to device

                total_reward += reward

                # Store values for training
                states.append(state_tensor)
                actions.append(action)
                old_log_probs.append(log_prob.detach())
                values.append(self.critic(state_tensor))
                rewards.append(torch.tensor([reward], dtype=torch.float, device=self.device))
                masks.append(torch.tensor([1 - done], dtype=torch.float, device=self.device))

                state = next_state
                if done or i >= max_timesteps:
                    print(f'Iteration: {iter}, Timesteps: {i}, Total Reward: {total_reward}')
                    self.records_history.append(total_reward)
                    break

            states_tensor = torch.cat(states)
            actions_tensor = torch.stack(actions)
            next_state_tensor = next_state.unsqueeze(0).to(self.device)
            next_value = self.critic(next_state_tensor)
            returns = self.compute_returns(next_value, rewards, masks)
            old_log_probs_tensor = torch.cat(old_log_probs)
            values_tensor = torch.cat(values)
            advantage = self.advantages.compute(rewards, values, masks)
            advantage_detach = advantage.clone().detach()

            # PPO Updates
            for _ in tqdm(range(n_ppo_updates)):
                dist = self.actor(states_tensor)

                # Compute log probability based on action space
                if self.discrete:
                    log_probs = dist.log_prob(actions_tensor)
                else:
                    log_probs = dist.log_prob(actions_tensor).sum(dim=-1, keepdim=True)

                ratio = torch.exp(log_probs - old_log_probs_tensor)
                clipped_advantage = self.clip_advantage(advantage)
                actor_loss = -torch.min(ratio * advantage_detach,
                                        clipped_advantage).mean() + self.c * dist.entropy().mean()

                # Optimize Actor
                self.optimizerA.zero_grad()
                actor_loss.backward()
                self.optimizerA.step()

                # Optimize Critic
                critic_loss = (returns - self.critic(states_tensor)).pow(2).mean()
                self.optimizerC.zero_grad()
                critic_loss.backward()
                self.optimizerC.step()

        self.env.close()

    def save(self):
        torch.save(self.actor.state_dict(), 'actor_ppo.pth')
        torch.save(self.critic.state_dict(), "critic_ppo.pth")

    def history(self):
        return self.records_history()