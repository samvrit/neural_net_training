import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch_directml
import matplotlib.pyplot as plt
import numpy as np

GAMMA = 0.99
SIM_STEPS = 2048
LEARNING_RATE_ACTOR = 3e-4
LEARNING_RATE_CRITIC = 1e-3
NUM_EPISODES = 256

device = torch_directml.device()
criterion_critic = torch.nn.MSELoss()

class ActorNN(nn.Module):
    def __init__(self):
        super(ActorNN, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        mean = torch.tanh(x[..., 0]) * 3.0
        sigma = F.softplus(x[..., 1]) + 1e-5
        return mean, sigma

class CriticNN(nn.Module):
    def __init__(self):
        super(CriticNN, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 2. Initialize Model and Move to DirectML
actor_nn = ActorNN().to(device)
critic_nn = CriticNN().to(device)
optimizer_actor = torch.optim.Adam(actor_nn.parameters(), lr=LEARNING_RATE_ACTOR)
optimizer_critic = torch.optim.Adam(critic_nn.parameters(), lr=LEARNING_RATE_CRITIC)

env = gym.make("InvertedPendulum-v5", render_mode="rgb_array", reset_noise_scale=0.1)

# Add video recording for every episode
env = RecordVideo(
    env,
    video_folder="cartpole-agent",    # Folder to save videos
    name_prefix="eval",               # Prefix for video filenames
    episode_trigger=lambda x: True    # Record every episode
)

# Reset environment
state, info = env.reset()

actor_losses = []
critic_losses = []
iterations = []

for episode in range(NUM_EPISODES):
    done = False
    state = torch.from_numpy(state).float()
    total_reward = 0
    state, info = env.reset()
    while not done:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        value = critic_nn(state_tensor)
    
        # Get action from actor network
        mu, sigma = actor_nn(state_tensor)
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample()
        action_to_step = action.detach().cpu().numpy().flatten()

        next_state, reward, terminated, truncated, info = env.step(action_to_step)
        done = terminated or truncated

        next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
        value_next = critic_nn(next_state_tensor)

        with torch.no_grad():
            target = reward + (GAMMA * value_next if not done else 0)

        # 2. Convert target to a tensor matching the device of 'value'
        target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device)
        target_tensor = target_tensor.view_as(value)
        advantage = target - value
        
        # Actor and critic losses
        log_prob = dist.log_prob(action)
        actor_loss = -(log_prob * advantage.detach())
        critic_loss = F.mse_loss(value, target_tensor)

        actor_losses.append(actor_loss.item())
        critic_losses.append(critic_loss.item())

        optimizer_actor.zero_grad()
        actor_loss.backward()
        optimizer_actor.step()

        optimizer_critic.zero_grad()
        critic_loss.backward()
        optimizer_critic.step()

        state = next_state
        total_reward += reward
        iterations.append(len(iterations))

    if episode % 10 == 0:
        print(f"Episode {episode} | Reward: {total_reward:.2f}")

env.close()

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(iterations, actor_losses, color='blue')
ax[1].plot(iterations, critic_losses, color='red')
plt.savefig('cartpole-agent/1-losses.png')
