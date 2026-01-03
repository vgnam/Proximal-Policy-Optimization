import numpy as np
import torch
from returns import gae


class RolloutBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.obs_buf = np.zeros((size, *obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, *act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, act, rew, val, logp):
        """Append one timestep of agent-environment interaction"""
        assert self.ptr < self.max_size, "Buffer overflow! Call get() or clear() before adding more data."
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Calculate GAE and returns for the current trajectory using the gae() function.
        Call this at the end of each episode or when truncating a trajectory.
        """
        path_slice = slice(self.path_start_idx, self.ptr)

        # Convert to tensors for gae() function
        rewards = torch.FloatTensor(self.rew_buf[path_slice])
        values = torch.FloatTensor(np.append(self.val_buf[path_slice], last_val))
        dones = torch.zeros_like(rewards)  # No dones within a single trajectory

        # Calculate advantages and returns
        advantages, returns = gae(rewards, values, dones, self.gamma, self.lam)

        # Store results
        self.adv_buf[path_slice] = advantages.numpy()
        self.ret_buf[path_slice] = returns.numpy()

        # Update path start for next trajectory
        self.path_start_idx = self.ptr

    def get(self):
        """
        Return all data in buffer up to current pointer.
        CRITICAL FIX: Removed assertion that buffer must be full.
        """
        # Use actual data size, not max_size
        actual_size = self.ptr

        if actual_size == 0:
            raise ValueError("Buffer is empty! Call collect_rollouts() before get().")

        # Reset pointers for next collection
        self.ptr, self.path_start_idx = 0, 0

        # Return only the filled portion of the buffer
        return {
            'obs': torch.FloatTensor(self.obs_buf[:actual_size]).to(self.device),
            'act': torch.FloatTensor(self.act_buf[:actual_size]).to(self.device),
            'ret': torch.FloatTensor(self.ret_buf[:actual_size]).to(self.device),
            'adv': torch.FloatTensor(self.adv_buf[:actual_size]).to(self.device),
            'logp': torch.FloatTensor(self.logp_buf[:actual_size]).to(self.device)
        }

    def clear(self):
        """Clear the buffer to reset it for the next set of data."""
        self.ptr, self.path_start_idx = 0, 0
        self.obs_buf.fill(0)
        self.act_buf.fill(0)
        self.rew_buf.fill(0)
        self.val_buf.fill(0)
        self.logp_buf.fill(0)
        self.adv_buf.fill(0)
        self.ret_buf.fill(0)