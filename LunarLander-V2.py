"""
LunarLander-v3 – DQN / DDQN / PER
Author: Filipe Ramos (CIS2719 Coursework 2)

This Python script trains three deep RL agents on the Gymnasium LunarLander-v3 task using:
  - DQN  (Deep Q-Network)
  - DDQN (Double DQN)
  - PER  (Prioritised Experience Replay + Double DQN targets)

For clarity of the marker:
  - learning_curves.png - shows episodic return against number of episodes
  - dqn_agent.gif       - Displays DQN post training
  - ddqn_agent.gif      - Displays DDQN post training
  - per_agent.gif       - Displays PER post training
"""

import os
import math
import random
import collections
from datetime import datetime

import gymnasium as gym
import imageio
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

def make_env(env_name: str, seed: int = 42, render_mode=None):
    """Create and seed environment"""
    env = gym.make(env_name, render_mode=render_mode)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return env

class QNetwork(nn.Module):
    """2-layer MLP for Q(s,a) approximation"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ReplayBuffer:
    """Standard replay buffer"""

    def __init__(self, capacity: int = 100_000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """Proportional priority replay buffer"""

    def __init__(self, capacity: int = 100_000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha

        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, indices, new_priorities):
        for idx, prio in zip(indices, new_priorities):
            self.priorities[idx] = float(prio)


class DQNAgent:
    """Agent supporting DQN, DDQN, and PER"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        algo_type: str = "dqn",
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 64,
        buffer_capacity: int = 100_000,
        min_buffer_size: int = 10_000,
        target_update_freq: int = 1_000,
        device: str | None = None,
        eps_decay_frames: int = 250_000,  # Allow configuring epsilon decay
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.target_update_freq = target_update_freq
        self.algo_type = algo_type.lower()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50000, gamma=0.9)

        self.loss_fn = nn.SmoothL1Loss(
            reduction="none" if self.algo_type == "per" else "mean"
        )

        if self.algo_type == "per":
            self.buffer = PrioritizedReplayBuffer(capacity=buffer_capacity)
            self.beta_start = 0.4
            self.beta_frames = 200_000
        else:
            self.buffer = ReplayBuffer(capacity=buffer_capacity)

        self.eps_start = 1.0
        self.eps_end = 0.05  # Minimum epsilon to maintain some exploration
        self.eps_decay = eps_decay_frames
        self.frame_idx = 0

        self.training_steps = 0

    def epsilon(self) -> float:
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.frame_idx / self.eps_decay
        )

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection"""
        self.frame_idx += 1
        if random.random() < self.epsilon():
            return random.randrange(self.action_dim)

        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def greedy_action(self, state: np.ndarray) -> int:
        """Greedy action selection"""
        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def push(self, *transition):
        self.buffer.push(*transition)

    def can_update(self) -> bool:
        return len(self.buffer) >= self.min_buffer_size

    def compute_td_target(
        self, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        """Compute TD targets for DQN/DDQN/PER"""
        with torch.no_grad():
            next_q_target = self.target_net(next_states)

            if self.algo_type in ("ddqn", "per"):
                next_q_online = self.q_net(next_states)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)
                next_q = next_q_target.gather(1, next_actions).squeeze(1)
            else:
                next_q, _ = next_q_target.max(dim=1)

            td_target = rewards + self.gamma * (1.0 - dones) * next_q
        return td_target

    def update(self):
        if not self.can_update():
            return None

        self.training_steps += 1

        if self.algo_type == "per":
            beta = min(
                1.0,
                self.beta_start
                + self.training_steps * (1.0 - self.beta_start) / self.beta_frames,
            )
            (
                states,
                actions,
                rewards,
                next_states,
                dones,
                indices,
                weights,
            ) = self.buffer.sample(self.batch_size, beta)
            weights = torch.tensor(
                weights, dtype=torch.float32, device=self.device
            )
        else:
            states, actions, rewards, next_states, dones = self.buffer.sample(
                self.batch_size
            )
            weights = torch.ones(self.batch_size, device=self.device)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(
            1
        )
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(
            next_states, dtype=torch.float32, device=self.device
        )
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.q_net(states).gather(1, actions).squeeze(1)

        td_target = self.compute_td_target(rewards, next_states, dones)

        loss_tensor = self.loss_fn(q_values, td_target)
        loss = (loss_tensor * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.scheduler.step()

        if self.algo_type == "per":
            new_priorities = loss_tensor.detach().cpu().numpy() + 1e-6
            self.buffer.update_priorities(indices, new_priorities)

        if self.training_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())


def train_agent(
    env_name: str,
    algo_type: str,
    num_episodes: int = 600,
    seed: int = 42,
) -> tuple[DQNAgent, list[float]]:
    """Train agent and return rewards"""

    env = make_env(env_name, seed=seed, render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    eps_decay_frames = max(250_000, num_episodes * 400)

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        algo_type=algo_type,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        buffer_capacity=100_000,
        min_buffer_size=10_000,
        target_update_freq=1_000,
        eps_decay_frames=eps_decay_frames,
    )

    episode_rewards: list[float] = []
    best_avg_reward = -float('inf')
    patience_counter = 0
    patience_limit = 500

    print(f"\n=== Training {algo_type.upper()} on {env_name} for {num_episodes} episodes ===")
    print(f"Epsilon decay frames: {eps_decay_frames:,}")

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.push(state, action, reward, next_state, float(done))

            state = next_state
            total_reward += reward

            if agent.can_update():
                agent.update()

        episode_rewards.append(total_reward)

        if episode >= 100:
            avg_reward_100 = np.mean(episode_rewards[-100:])
            if avg_reward_100 > best_avg_reward:
                best_avg_reward = avg_reward_100
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience_limit and num_episodes > 2000:
                print(f"\nEarly stopping at episode {episode}: Performance plateaued")
                print(f"Best 100-ep avg: {best_avg_reward:.2f}, Current: {avg_reward_100:.2f}")
                break

        if episode % 20 == 0:
            last_mean = np.mean(episode_rewards[-20:])
            current_lr = agent.optimizer.param_groups[0]['lr']
            print(
                f"Episode {episode:4d} | "
                f"avg reward (last 20): {last_mean:7.2f} | "
                f"epsilon: {agent.epsilon():.3f} | "
                f"lr: {current_lr:.6f}"
            )

    env.close()
    return agent, episode_rewards


def plot_learning_curves(results: dict, save_path: str = "learning_curves.png"):
    """Plot reward curves"""
    plt.figure(figsize=(10, 6))
    for label, rewards in results.items():
        rewards = np.array(rewards)
        window = 20
        if len(rewards) >= window:
            smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
            plt.plot(
                range(window, len(rewards) + 1),
                smooth,
                label=f"{label} (moving avg)",
            )
        plt.plot(np.arange(1, len(rewards) + 1), rewards, alpha=0.25, linestyle="--")

    plt.axhline(200, color="grey", linestyle=":", label="Solved threshold (≈200)")
    plt.xlabel("Episode")
    plt.ylabel("Episodic Return")
    plt.title("LunarLander-v3: DQN vs DDQN vs PER")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved learning curves to {save_path}")


def generate_gif(
    agent: DQNAgent,
    env_name: str,
    filename: str,
    seed: int = 0,
    episodes: int = 3,
):
    """Generate GIF of trained agent"""
    env = make_env(env_name, seed=seed, render_mode="rgb_array")
    frames = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        step = 0
        while not done:
            frame = env.render()
            frames.append(frame)

            action = agent.greedy_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            step += 1

    env.close()
    imageio.mimsave(filename, frames, fps=30)
    print(f"Saved GIF for policy to {filename}")


if __name__ == "__main__":
    ENV_NAME = "LunarLander-v3"

    NUM_EPISODES = 10000
    TEST_NAME = "Test 10000eps (Post changes)"

    output_dir = f"results/{TEST_NAME}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Starting Training Run: {TEST_NAME}")
    print(f"Episodes per algorithm: {NUM_EPISODES}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    results: dict[str, list[float]] = {}
    trained_agents: dict[str, DQNAgent] = {}

    for algo in ["dqn", "ddqn", "per"]:
        agent, rewards = train_agent(ENV_NAME, algo_type=algo, num_episodes=NUM_EPISODES)
        results[algo.upper()] = rewards
        trained_agents[algo.upper()] = agent

        print(
            f"{algo.upper()} final mean reward over last 50 episodes: "
            f"{np.mean(rewards[-50:]):.2f}"
        )

    plot_learning_curves(results, save_path=f"{output_dir}/learning_curves.png")

    generate_gif(trained_agents["DQN"], ENV_NAME, f"{output_dir}/dqn_agent.gif")
    generate_gif(trained_agents["DDQN"], ENV_NAME, f"{output_dir}/ddqn_agent.gif")
    generate_gif(trained_agents["PER"], ENV_NAME, f"{output_dir}/per_agent.gif")

    print(f"\n{'='*60}")
    print(f"Training Complete: {TEST_NAME}")
    print(f"All results saved to: {output_dir}/")
    print(f"{'='*60}")
