import torch

def gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    gae = 0
    ret = values[-1]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae

        ret = rewards[t] + gamma * ret * (1 - dones[t])
        returns[t] = ret

    return advantages, returns


def reward_to_go(rewards, dones, gamma):
    returns = torch.zeros_like(rewards)
    ret = 0

    for t in reversed(range(len(rewards))):
        ret = rewards[t] + gamma * ret * (1 - dones[t])
        returns[t] = ret

    return returns

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