from Network import Actor, Critic
from PPO_CNN import PPO_CNN, GAE
import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="rgb_array")

actor = Actor(input_channels=3)
critic = Critic(input_channels=3)

from torchvision import transforms

Transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert the state to PIL image
    transforms.Resize((48, 48)),  # Resize image to 224x224
    transforms.ToTensor(),  # Convert the PIL image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image
])

gae = GAE(gamma=0.9, lambda_=0.9)
ppo_trainer = PPO_CNN(env, actor, critic, transform=Transform, gamma=0.9, epsilon=0.001)
ppo_trainer.train(max_timesteps=10000, n_iters=1000, n_ppo_updates=200)
