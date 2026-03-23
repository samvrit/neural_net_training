import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch_directml

GAMMA = 0.99

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
        return x

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

def actor_loss(mu, sigma, action, advantage):
    """
    Compute the log probability of the action under a Gaussian distribution with mean mu and std sigma.
    This can be used as the loss function for the actor network in policy gradient methods.
    """
    dist = Normal(mu, sigma)
    log_prob = dist.log_prob(action).sum(dim=-1)  # Sum over action dimensions if multi-dimensional
    return -(log_prob.mean() * advantage)  # Negative for minimization (policy gradient)

def critic_mse_loss(predicted_value, target_value):
    """
    Compute the mean squared error loss for the critic network.
    Used to train the value function to predict state values.
    """
    return criterion_critic(predicted_value, target_value)

def advantage_estimation(reward, current_value, future_value, gamma=GAMMA):
    """
    Compute the advantage estimate using the reward, current value, and future value.
    This can be used to compute the advantage for policy gradient updates.
    """
    return reward + (gamma * future_value) - current_value

# 2. Initialize Model and Move to DirectML
actor_nn = ActorNN().to(device)
critic_nn = CriticNN().to(device)
optimizer_actor = torch.optim.Adam(actor_nn.parameters(), lr=3e-4)
optimizer_critic = torch.optim.Adam(critic_nn.parameters(), lr=3e-4)

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

for i in range(2048):
    # Get action from actor network
    output = actor_nn(torch.tensor(state, dtype=torch.float32, device=device))
    mu, log_sigma = output[0], output[1]  # Assuming output is [mu, log_sigma]
    sigma = torch.exp(log_sigma)  # Convert log_sigma to sigma
    dist = torch.distributions.Normal(mu, sigma)
    action = dist.sample() # This is falling back to CPU and breaking the graph
    log_prob = dist.log_prob(action)

    value = critic_nn(torch.tensor(state, dtype=torch.float32, device=device))

    next_state, reward, terminated, truncated, info = env.step(action.unsqueeze(0).cpu().numpy())
    done = terminated or truncated

    value_next = critic_nn(torch.tensor(next_state, dtype=torch.float32, device=device))
    advantage = advantage_estimation(reward, value.item(), value_next.item(), gamma=GAMMA)

    actor_loss_value = actor_loss(mu, sigma, action, advantage)
    critic_loss_value = critic_mse_loss(value, reward + (GAMMA * value_next))

    optimizer_actor.zero_grad()
    actor_loss_value.backward()
    optimizer_actor.step()

    optimizer_critic.zero_grad()
    critic_loss_value.backward()
    optimizer_critic.step()

    print(f"Step {i+1}: Reward={reward:.2f}, Advantage={advantage:.2f}, Actor Loss={actor_loss_value.item():.4f}, Critic Loss={critic_loss_value.item():.4f}")

    if not done:
        state = next_state
    else:
        state, info = env.reset()

env.close()
# print(f"Ran {episodes} episodes with {steps} total steps")
