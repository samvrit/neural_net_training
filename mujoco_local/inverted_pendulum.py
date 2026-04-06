import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import numpy as np

GAMMA = 0.99
SIM_STEPS = 1000
LEARNING_RATE_ACTOR = 3e-4
LEARNING_RATE_CRITIC = 1e-3
NUM_EPISODES = 100
NUM_TRAINING_EPOCHS = 10
EPS_CLIP = 0.2
NUM_INFERENCE_EPISODES = 5
ENV_NAME = "InvertedPendulum-v5"

device = torch.device("cpu")

# 1. Define Actor and Critic Networks
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

env = gym.make(ENV_NAME, reset_noise_scale=0.1)

# Reset environment
state, info = env.reset()

actor_losses = []
critic_losses = []
iterations = []
rewards_per_episode = []
sigma_values = []

# 3. Training Loop
for episode in range(NUM_EPISODES):
    done = False
    state = torch.from_numpy(state).float()
    total_reward = 0
    state, info = env.reset()

    states_list = []
    actions_list = []
    log_probs_list = []
    rewards_list = []
    dones_list = []
    steps_per_episode = 0

    while (steps_per_episode < SIM_STEPS):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
        # Get action from actor network
        mu, sigma = actor_nn(state_tensor)
        dist = Normal(mu, sigma)
        action = dist.sample()
        action_to_step = action.detach().cpu().numpy().flatten()

        next_state, reward, terminated, truncated, info = env.step(action_to_step)
        done = terminated or truncated

        log_prob = dist.log_prob(action)

        states_list.append(state)
        actions_list.append(action.detach().cpu().numpy())
        log_probs_list.append(log_prob.detach())
        rewards_list.append(reward)
        dones_list.append(done)
        sigma_values.append(sigma.detach().cpu().numpy())

        total_reward += reward
        iterations.append(len(iterations))
        steps_per_episode += 1

        if done:
            state, info = env.reset()
        else:
            state = next_state

    rewards_per_episode.append(total_reward)

    returns = []
    discounted_reward = 0
    for reward, done in zip(reversed(rewards_list), reversed(dones_list)):
        if done:
            discounted_reward = reward
        else:
            discounted_reward = reward + (GAMMA * discounted_reward)
        returns.insert(0, discounted_reward)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)

    # Convert lists to Tensors
    states = torch.from_numpy(np.array(states_list)).float().to(device)
    actions = torch.from_numpy(np.array(actions_list)).float().to(device).squeeze(-1)  # [T]
    old_log_probs = torch.stack(log_probs_list).detach().squeeze(-1)  # [T]

    # Update Actor and Critic Networks
    for _ in range(NUM_TRAINING_EPOCHS):
        mu, sigma = actor_nn(states)
        dist = Normal(mu, sigma)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        values = critic_nn(states).squeeze()
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-EPS_CLIP, 1+EPS_CLIP) * advantages
        actor_loss = -torch.min(surr1, surr2).mean() - (0.01 * entropy)

        critic_loss = F.mse_loss(values, returns)

        optimizer_actor.zero_grad()
        actor_loss.backward()
        optimizer_actor.step()

        optimizer_critic.zero_grad()
        critic_loss.backward()
        optimizer_critic.step()

    print(f"Episode {episode} | Reward: {total_reward:.2f}")

env.close()

# 4. Inference and Evaluation
eval_env = RecordVideo(
    gym.make(ENV_NAME, render_mode="rgb_array", reset_noise_scale=0.1),
    video_folder="cartpole-agent",
    name_prefix="eval",
    episode_trigger=lambda episode_id: True
)

for eval_episode in range(NUM_INFERENCE_EPISODES):
    state, info = eval_env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done and steps < SIM_STEPS:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            mu, _ = actor_nn(state_tensor)

        action_to_step = mu.detach().cpu().numpy().flatten()
        state, reward, terminated, truncated, info = eval_env.step(action_to_step)
        done = terminated or truncated
        total_reward += reward
        steps += 1

    print(f"Inference Episode {eval_episode + 1} | Reward: {total_reward:.2f}")

eval_env.close()

# 5. Plotting Rewards and Sigma Values
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(range(NUM_EPISODES), rewards_per_episode, color='blue')
ax[1].plot(iterations, sigma_values, color='red')
plt.savefig('cartpole-agent/1-losses.png')
