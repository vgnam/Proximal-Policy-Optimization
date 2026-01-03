import torch
import numpy as np
from torch.optim import Adam
from collections import deque
from rollout_buffer import RolloutBuffer


class PPO:
    """
    Proximal Policy Optimization (PPO) agent.
    This implementation uses a RolloutBuffer for data management and GAE for advantage calculation.
    """

    def __init__(self, env, actor, critic, discrete=True,
                 lr_actor=3e-4, lr_critic=3e-4,
                 gamma=0.99, gae_lambda=0.95,
                 epsilon=0.2, c_ent=0.01, value_coef=0.5,
                 max_grad_norm=0.5, target_kl=0.015,
                 early_stopping=True, buffer_size=512, mini_batch_size=64,
                 rolling_window=100, suffix=''):
        """
        Initializes the PPO agent.

        Args:
            env: The Gym-like environment.
            actor: The policy network (PyTorch module).
            critic: The value network (PyTorch module).
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
            rolling_window (int): Window size for computing rolling statistics.
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
        self.suffix = suffix
        self.rolling_window = rolling_window

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        self.optimizerA = Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizerC = Adam(self.critic.parameters(), lr=lr_critic)

        # Setup buffer dimensions
        obs_dim = env.observation_space.shape
        if discrete:
            act_dim = ()  # Scalar for discrete actions
        else:
            act_dim = env.action_space.shape

        self.buffer = RolloutBuffer(obs_dim, act_dim, buffer_size, gamma, gae_lambda, self.device)

        # Return history with rolling window tracking
        self.episode_returns = []  # All episode returns
        self.rolling_returns = deque(maxlen=rolling_window)  # Rolling window of returns
        self.iteration_stats = []  # Per-iteration stats: (iter, mean, std, min, max, rolling_mean)
        self.best_reward = float('-inf')
        self.best_rolling_avg = float('-inf')

    def get_rolling_stats(self):
        """Compute statistics from the rolling window."""
        if len(self.rolling_returns) == 0:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0}

        returns_array = np.array(self.rolling_returns)
        return {
            'mean': np.mean(returns_array),
            'std': np.std(returns_array),
            'min': np.min(returns_array),
            'max': np.max(returns_array),
            'count': len(self.rolling_returns)
        }

    def collect_rollouts(self, max_timesteps=None):
        """
        Collects trajectories using the current policy and stores them in the buffer.
        Returns episode-level statistics.
        """
        state, _ = self.env.reset()
        episode_rewards = []  # Completed episodes in this rollout
        current_episode_reward = 0
        timesteps = 0

        if max_timesteps is None:
            max_timesteps = self.buffer_size

        for _ in range(max_timesteps):
            if self.buffer.ptr >= self.buffer.max_size:
                break

            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            with torch.no_grad():
                value = self.critic(state_tensor).squeeze(0)
                dist = self.actor(state_tensor)

                if self.discrete:
                    action = dist.sample()
                    log_prob = dist.log_prob(action).squeeze()
                    action_np = action.cpu().numpy().item()
                else:
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(dim=-1).squeeze(0)
                    action_np = action.squeeze(0).cpu().numpy()

            next_state, reward, terminated, truncated, _ = self.env.step(action_np)
            done = terminated or truncated

            # Store in buffer
            self.buffer.store(
                obs=state,
                act=action_np.copy() if not self.discrete else action_np,
                rew=reward,
                val=value.cpu().numpy(),
                logp=log_prob.cpu().numpy()
            )

            current_episode_reward += reward
            state = next_state
            timesteps += 1

            # Episode finished - log the complete episode return
            if done:
                self.buffer.finish_path(last_val=0.0)
                episode_rewards.append(current_episode_reward)
                self.episode_returns.append(current_episode_reward)
                self.rolling_returns.append(current_episode_reward)  # Add to rolling window
                current_episode_reward = 0
                state, _ = self.env.reset()

        # Handle case where collection ended mid-episode
        if not done and self.buffer.ptr > self.buffer.path_start_idx:
            with torch.no_grad():
                last_val_tensor = self.critic(
                    torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                ).squeeze()
                last_val = last_val_tensor.cpu().numpy()
            self.buffer.finish_path(last_val=last_val)

        return episode_rewards, timesteps

    def train(self, n_iters, max_timesteps_per_iter=None, n_eps_per_iter=10):
        """
        Main training loop for PPO with rolling window statistics.
        """
        for iter in range(n_iters):
            self.buffer.clear()

            # Collect rollouts
            episode_rewards, timesteps = self.collect_rollouts(max_timesteps_per_iter)

            # Compute rolling statistics
            rolling_stats = self.get_rolling_stats()

            # Compute iteration statistics
            if len(episode_rewards) > 0:
                iter_mean = np.mean(episode_rewards)
                iter_std = np.std(episode_rewards)
                iter_min = np.min(episode_rewards)
                iter_max = np.max(episode_rewards)
            else:
                iter_mean = iter_std = iter_min = iter_max = 0.0

            # Store iteration stats
            self.iteration_stats.append({
                'iteration': iter + 1,
                'episodes': len(episode_rewards),
                'timesteps': timesteps,
                'mean_return': iter_mean,
                'std_return': iter_std,
                'min_return': iter_min,
                'max_return': iter_max,
                'rolling_mean': rolling_stats['mean'],
                'rolling_std': rolling_stats['std'],
                'rolling_min': rolling_stats['min'],
                'rolling_max': rolling_stats['max'],
                'rolling_count': rolling_stats['count']
            })

            # Print iteration summary with rolling window stats (single line)
            print(f"Iter {iter + 1}/{n_iters} | Eps: {len(episode_rewards)} | Steps: {timesteps} | "
                  f"Mean: {iter_mean:.2f} | Rolling({rolling_stats['count']}): {rolling_stats['mean']:.2f}±{rolling_stats['std']:.2f}")

            # Save best model based on rolling average
            if rolling_stats['mean'] > self.best_rolling_avg and rolling_stats['count'] >= 10:
                self.best_rolling_avg = rolling_stats['mean']
                self.save(suffix=self.suffix + '_best')
                print(f"  ✓ New best rolling avg: {self.best_rolling_avg:.2f}")

            # Training logic
            actual_size = self.buffer.ptr
            if actual_size == 0:
                print("  Warning: No data collected, skipping training.")
                continue

            data = self.buffer.get()

            # Prepare data
            states = data['obs']
            actions = data['act']
            returns = data['ret']
            advantages = data['adv']
            logp_old = data['logp']

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO update
            early_stopped = False
            for epoch in range(n_eps_per_iter):
                if early_stopped:
                    break

                indices = np.random.permutation(actual_size)

                for start in range(0, actual_size, self.mini_batch_size):
                    end = min(start + self.mini_batch_size, actual_size)
                    mb_indices = indices[start:end]

                    # Get mini-batch data
                    mb_states = states[mb_indices]
                    mb_actions = actions[mb_indices]
                    mb_returns = returns[mb_indices]
                    mb_advantages = advantages[mb_indices]
                    mb_logp_old = logp_old[mb_indices]

                    # Forward pass
                    values = self.critic(mb_states).squeeze(-1)
                    dist = self.actor(mb_states)

                    # Compute new log probabilities
                    if self.discrete:
                        logp = dist.log_prob(mb_actions.long())
                    else:
                        logp = dist.log_prob(mb_actions).sum(dim=-1)

                    # Calculate KL divergence before optimization
                    with torch.no_grad():
                        ratio = torch.exp(logp - mb_logp_old)
                        kl_div = ((mb_logp_old - logp) * ratio).mean().item()

                    # Check for early stopping
                    if self.early_stopping and kl_div > self.target_kl:
                        print(f"  Early stop at epoch {epoch + 1} (KL: {kl_div:.4f})")
                        early_stopped = True
                        break

                    # Compute ratio (recalculate for gradient flow)
                    ratio = torch.exp(logp - mb_logp_old)

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
                    self.optimizerA.zero_grad()
                    self.optimizerC.zero_grad()
                    total_loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                    self.optimizerA.step()
                    self.optimizerC.step()

            # Periodic evaluation
            if (iter + 1) % 50 == 0:
                self.save(suffix=self.suffix + f'_iter{iter + 1}')
                avg_reward, _ = self.evaluate(n_episodes=5)
                print(f"  Eval (5 eps): {avg_reward:.2f}")

    def save(self, suffix=''):
        """Saves the actor and critic models."""
        torch.save(self.actor.state_dict(), f'ppo_actor{suffix}.pth')
        torch.save(self.critic.state_dict(), f'ppo_critic{suffix}.pth')

    def load(self, suffix=''):
        """Loads the actor and critic models."""
        self.actor.load_state_dict(torch.load(f'ppo_actor{suffix}.pth', map_location=self.device))
        self.critic.load_state_dict(torch.load(f'ppo_critic{suffix}.pth', map_location=self.device))

    def evaluate(self, n_episodes=10, render=False):
        """Evaluates the current policy."""
        total_rewards = []

        for ep in range(n_episodes):
            state, _ = self.env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

                with torch.no_grad():
                    dist = self.actor(state_tensor)
                    if self.discrete:
                        action = torch.argmax(dist.probs).item()
                    else:
                        action = dist.mean.squeeze(0).cpu().numpy()

                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                ep_reward += reward

                if render:
                    self.env.render()

            total_rewards.append(ep_reward)

        avg_reward = np.mean(total_rewards)
        return avg_reward, total_rewards

    def get_returns_history(self):
        """Returns all episode returns and iteration statistics."""
        return {
            'episode_returns': self.episode_returns,
            'iteration_stats': self.iteration_stats,
            'rolling_window_size': self.rolling_window
        }