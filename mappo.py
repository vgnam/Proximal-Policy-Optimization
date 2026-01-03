import torch
import numpy as np
from torch.optim import Adam
import logging
from tqdm import tqdm
from rollout_buffer import RolloutBuffer
from returns import gae, GAE


class MAPPO:
    """
    Multi-Agent Proximal Policy Optimization (MAPPO) agent.
    Supports independent agents with shared or separate networks.
    """

    def __init__(self, env, actors, critics, discrete=True,
                 lr_actor=3e-4, lr_critic=3e-4,
                 gamma=0.99, gae_lambda=0.95,
                 epsilon=0.2, c_ent=0.01, value_coef=0.5,
                 max_grad_norm=0.5, target_kl=0.015,
                 early_stopping=True, buffer_size=2048, mini_batch_size=256,
                 share_params=True):
        """
        Initializes the MAPPO agent.

        Args:
            env: The multi-agent Gym-like environment (must have agent_ids attribute).
            actors: List of actor networks (one per agent) or single shared network.
            critics: List of critic networks (one per agent) or single shared network.
            discrete (bool): Whether the action space is discrete.
            lr_actor (float): Learning rate for the actor.
            lr_critic (float): Learning rate for the critic.
            gamma (float): Discount factor.
            gae_lambda (float): Lambda for Generalized Advantage Estimation.
            epsilon (float): Clipping parameter for PPO.
            c_ent (float): Entropy coefficient.
            value_coef (float): Value loss coefficient.
            max_grad_norm (float): Maximum norm for gradient clipping.
            target_kl (float): Target KL divergence for early stopping.
            early_stopping (bool): Whether to use early stopping based on KL.
            buffer_size (int): Size of the rollout buffer.
            mini_batch_size (int): Mini-batch size for training updates.
            share_params (bool): Whether agents share the same network parameters.
        """
        self.env = env
        self.discrete = discrete
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon
        self.c_ent = c_ent
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.early_stopping = early_stopping
        self.buffer_size = buffer_size
        self.mini_batch_size = mini_batch_size
        self.share_params = share_params

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Handle agents
        if isinstance(actors, list):
            self.actors = [actor.to(self.device) for actor in actors]
            self.n_agents = len(actors)
        else:
            self.actors = [actors.to(self.device)]  # Shared network
            self.n_agents = env.n_agents  # Assuming env has n_agents attribute

        if isinstance(critics, list):
            self.critics = [critic.to(self.device) for critic in critics]
        else:
            self.critics = [critics.to(self.device)]  # Shared network

        # Optimizers
        if share_params:
            # Single optimizer for shared networks
            self.optimizerA = Adam(self.actors[0].parameters(), lr=lr_actor)
            self.optimizerC = Adam(self.critics[0].parameters(), lr=lr_critic)
        else:
            # Separate optimizers for each agent
            self.optimizerA = [Adam(actor.parameters(), lr=lr_actor) for actor in self.actors]
            self.optimizerC = [Adam(critic.parameters(), lr=lr_critic) for critic in self.critics]

        # Multi-agent buffer: one buffer per agent (or shared)
        obs_dim = env.observation_space.shape
        if discrete:
            act_dim = ()  # Scalar for discrete actions
        else:
            act_dim = env.action_space.shape

        # Create separate buffers for each agent
        self.buffers = [
            RolloutBuffer(obs_dim, act_dim, buffer_size, gamma, gae_lambda, self.device)
            for _ in range(self.n_agents)
        ]

        self.gae_calculator = GAE(gamma, gae_lambda)
        self.returns_history = [0.0] * self.n_agents  # Track returns per agent
        self.best_rewards = [float('-inf')] * self.n_agents

    def collect_rollouts(self, max_timesteps=None):
        """
        Collects trajectories for all agents simultaneously.
        """
        obs, _ = self.env.reset()
        total_rewards = [0.0] * self.n_agents
        timesteps = 0

        if max_timesteps is None:
            max_timesteps = self.buffer_size

        for _ in tqdm(range(max_timesteps)):
            # Check if any buffer is full
            if any(buf.ptr >= buf.max_size for buf in self.buffers):
                break

            # Convert observations to tensors
            if isinstance(obs, dict):
                obs_tensors = {aid: torch.as_tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0)
                               for aid, o in obs.items()}
            else:
                obs_tensors = [torch.as_tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0)
                               for o in obs]

            # Sample actions for all agents
            actions = []
            log_probs = []
            values = []

            for i in range(self.n_agents):
                # Select the correct actor/critic (shared or individual)
                actor = self.actors[0] if self.share_params else self.actors[i]
                critic = self.critics[0] if self.share_params else self.critics[i]

                obs_tensor = obs_tensors[i] if isinstance(obs_tensors, list) else obs_tensors[i]

                with torch.no_grad():
                    value = critic(obs_tensor).squeeze()
                    dist = actor(obs_tensor)

                    if self.discrete:
                        action = dist.sample()
                        log_prob = dist.log_prob(action).squeeze()
                    else:
                        action = dist.sample()
                        log_prob = dist.log_prob(action).sum(dim=-1).squeeze()

                    values.append(value.cpu().numpy())
                    actions.append(action.cpu().numpy())
                    log_probs.append(log_prob.cpu().numpy())

            # Take step in environment
            next_obs, rewards, terminated, truncated, _ = self.env.step(actions)
            done = terminated or truncated

            # Store in each agent's buffer
            for i in range(self.n_agents):
                action_np = actions[i]
                if self.discrete and hasattr(action_np, 'item'):
                    action_np = action_np.item()

                # Store data in agent's buffer
                self.buffers[i].store(
                    obs=obs[i] if isinstance(obs, (list, tuple)) else list(obs.values())[i],
                    act=action_np,
                    rew=rewards[i],  # Each agent gets its own reward
                    val=values[i],
                    logp=log_probs[i]
                )
                total_rewards[i] += rewards[i]

            obs = next_obs
            timesteps += 1

            if done:
                obs, _ = self.env.reset()

        # Finish paths for all buffers
        last_vals = [0.0] * self.n_agents
        if not done:
            # Bootstrap last values
            for i in range(self.n_agents):
                actor = self.actors[0] if self.share_params else self.actors[i]
                critic = self.critics[0] if self.share_params else self.critics[i]

                obs_tensor = torch.as_tensor(
                    obs[i] if isinstance(obs, (list, tuple)) else list(obs.values())[i],
                    dtype=torch.float32, device=self.device
                ).unsqueeze(0)

                with torch.no_grad():
                    last_val = critic(obs_tensor).squeeze().cpu().numpy()
                    last_vals[i] = last_val

        for i, buf in enumerate(self.buffers):
            buf.finish_path(last_val=last_vals[i])

        return total_rewards, timesteps

    def train(self, n_iters, max_timesteps_per_iter=None):
        """
        Main training loop for MAPPO.
        """
        for iter in range(n_iters):
            # Clear all buffers
            for buf in self.buffers:
                buf.clear()

            # Collect rollouts
            total_rewards, timesteps = self.collect_rollouts(max_timesteps_per_iter)

            # Update returns history and save best models
            for i in range(self.n_agents):
                self.returns_history[i] = total_rewards[i]
                if total_rewards[i] > self.best_rewards[i]:
                    self.best_rewards[i] = total_rewards[i]
                    self.save(suffix=f'_agent_{i}_best')

            print(f"Iteration {iter + 1}/{n_iters}: Timesteps={timesteps}, Total Rewards={total_rewards}")

            # Train each agent separately
            for agent_id in range(self.n_agents):
                self._train_agent(agent_id)

            # Optional: Save periodically
            if iter % 20 == 0:
                self.save()

    def _train_agent(self, agent_id):
        """
        Train a single agent using PPO update rules.
        """
        # Get data from this agent's buffer
        data = self.buffers[agent_id].get()
        states = data['obs']
        actions = data['act']
        returns = data['ret']
        advantages = data['adv']
        logp_old = data['logp']

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Select optimizer and networks for this agent
        if self.share_params:
            optimizerA = self.optimizerA
            optimizerC = self.optimizerC
            actor = self.actors[0]
            critic = self.critics[0]
        else:
            optimizerA = self.optimizerA[agent_id]
            optimizerC = self.optimizerC[agent_id]
            actor = self.actors[agent_id]
            critic = self.critics[agent_id]

        # PPO update
        approx_kl_divs = []
        for _ in range(20):  # Number of PPO epochs per agent
            indices = np.random.permutation(self.buffer_size)

            for start in range(0, self.buffer_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_indices = indices[start:end]

                # Get mini-batch data
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_logp_old = logp_old[mb_indices]

                # Forward pass
                values = critic(mb_states).squeeze()
                dist = actor(mb_states)

                # Compute new log probabilities
                if self.discrete:
                    logp = dist.log_prob(mb_actions).squeeze()
                else:
                    mb_actions_tensor = torch.as_tensor(mb_actions, dtype=torch.float32, device=self.device)
                    logp = dist.log_prob(mb_actions_tensor).sum(dim=-1).squeeze()

                # Compute ratio
                ratio = torch.exp(logp - mb_logp_old.detach())

                # Compute surrogate losses
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Compute value loss
                value_loss = self.value_coef * (mb_returns - values).pow(2).mean()

                # Compute entropy bonus
                entropy_loss = -self.c_ent * dist.entropy().mean()

                # Total loss
                total_loss = policy_loss + value_loss + entropy_loss

                # Optimize
                optimizerA.zero_grad()
                optimizerC.zero_grad()
                total_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), self.max_grad_norm)

                optimizerA.step()
                optimizerC.step()

                # Calculate approximate KL divergence
                with torch.no_grad():
                    kl_div = ((mb_logp_old - logp) * ratio).mean().item()
                    approx_kl_divs.append(kl_div)

    def save(self, suffix=''):
        """Saves the actor and critic models for all agents."""
        if self.share_params:
            torch.save(self.actors[0].state_dict(), f'mappo_actor_shared{suffix}.pth')
            torch.save(self.critics[0].state_dict(), f'mappo_critic_shared{suffix}.pth')
        else:
            for i in range(self.n_agents):
                torch.save(self.actors[i].state_dict(), f'mappo_actor_agent_{i}{suffix}.pth')
                torch.save(self.critics[i].state_dict(), f'mappo_critic_agent_{i}{suffix}.pth')
        print(f"Models saved with suffix '{suffix}'.")

    def load(self, suffix=''):
        """Loads the actor and critic models for all agents."""
        if self.share_params:
            self.actors[0].load_state_dict(torch.load(f'mappo_actor_shared{suffix}.pth', map_location=self.device))
            self.critics[0].load_state_dict(torch.load(f'mappo_critic_shared{suffix}.pth', map_location=self.device))
        else:
            for i in range(self.n_agents):
                self.actors[i].load_state_dict(
                    torch.load(f'mappo_actor_agent_{i}{suffix}.pth', map_location=self.device))
                self.critics[i].load_state_dict(
                    torch.load(f'mappo_critic_agent_{i}{suffix}.pth', map_location=self.device))
        print(f"Models loaded with suffix '{suffix}'.")

    def evaluate(self, n_episodes=10, render=False):
        """Evaluates the current policy for all agents."""
        total_rewards = [[] for _ in range(self.n_agents)]

        for ep in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            ep_rewards = [0.0] * self.n_agents

            while not done:
                actions = []
                for i in range(self.n_agents):
                    # Select the correct actor
                    actor = self.actors[0] if self.share_params else self.actors[i]
                    obs_tensor = torch.as_tensor(
                        obs[i] if isinstance(obs, (list, tuple)) else list(obs.values())[i],
                        dtype=torch.float32, device=self.device
                    ).unsqueeze(0)

                    with torch.no_grad():
                        dist = actor(obs_tensor)
                        if self.discrete:
                            action = torch.argmax(dist.probs).item()  # Greedy for discrete
                        else:
                            action = dist.mean.cpu().numpy().flatten()

                    actions.append(action)

                obs, rewards, terminated, truncated, _ = self.env.step(actions)
                done = terminated or truncated

                for i in range(self.n_agents):
                    ep_rewards[i] += rewards[i]

                if render:
                    self.env.render()

            for i in range(self.n_agents):
                total_rewards[i].append(ep_rewards[i])
            print(f"Evaluation Episode {ep + 1}/{n_episodes}, Rewards: {ep_rewards}")

        avg_rewards = [np.mean(agent_rewards) for agent_rewards in total_rewards]
        print(f"Average Evaluation Rewards over {n_episodes} episodes: {avg_rewards}")
        return avg_rewards, total_rewards

