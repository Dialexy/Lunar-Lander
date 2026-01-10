# LunarLander-v2 Deep Reinforcement Learning

A showcase of deep reinforcement learning algorithms on the Gymnasium LunarLander-v3 environment.

## Overview

This project implements and compares three Q-learning algorithms:

**Algorithm overview found in the LunarLander-V3.py header comment**

## Features

- Automated training with configurable episodes (line: 552)
- Learning curve visualisation via outputted graph
- GIF generation for post training visualisation of each agent.

## Requirements

```bash
gymnasium
imageio
numpy
matplotlib
torch
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Dialexy/Lunar-Lander.git
cd Lunar-Lander
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate
# This if for Unix Systems on bash or zsh, see offical documentation for windows specific information
# Offical Doc: https://gymnasium.farama.org/environments/box2d/
```

3. Install dependencies:
```bash
pip install gymnasium imageio numpy matplotlib torch
```

## Usage

Activiate the virtual environment if step 2 was skipped:
```bash
source venv/bin/activate
```

Run the training script:
```bash
python 'Local Scripts'/LunarLander-V3.py
# Run via terminal (Extensions such as "code runner" will not work)
```

Configure training parameters in the main block:
```python
NUM_EPISODES = 10000  # Adjust episode count (line: 525)
TEST_NAME = "Test 10000eps"  # Name your test run
```

## Project Structure

```
├── Collab Scripts
│   └── LunarLander_Colab.ipynb
├── Gifs & Graphs
│   └── Full-Test-Runs
│   │   ├── Test 10000eps
│   │   ├── Test 1000eps
│   │   ├── Test 2500eps
│   │   ├── Test 5000eps
│   │   └── Test 500eps
├── Local Scripts
│   └── LunarLander-V2.py
└── README.md
```

## Architecture

### Q-Network
- 2-layer MLP with 128 hidden units
- ReLU activation
- Output: Q-values for each action

### Key Components

**ReplayBuffer** - Standard uniform sampling replay buffer

**PrioritisedReplayBuffer** - Ensures agent is learning from its largest mistakes

**DQNAgent** - sinuglar agent class supporting all three algorithms with:
- Epsilon-greedy exploration
- Target network updates
- Gradient clipping
- Learning rate scheduling

## Training Details

- **Optimizer**: Adam (lr=2.5e-4)
- **Batch Size**: 32
- **Gamma**: 0.99
- **Buffer Capacity**: 500,000
- **Target Update Frequency**: 10,000 steps
- **Epsilon Decay**: Linear over first 10% of timesteps
- **Loss Function**: Smooth L1 (Huber)

## Output

Each training run generates:
- `learning_curves.png` - Episodic return vs episodes for all algorithms
- `dqn_agent.gif` - DQN policy visualization
- `ddqn_agent.gif` - DDQN policy visualization
- `per_agent.gif` - PER policy visualization

## Results

The environment is considered solved when the agent achieves an average reward of ≥200 over 100 consecutive episodes.

### Output Structure

Each training run generates performance data in the `Gifs & Graphs/{TEST_NAME}/` directory:

**Performance Metrics** (`results_table.txt`):
- **Training Time**: Total time for all training episodes (seconds and minutes)
- **Inference Time**: Average time per evaluation episode using greedy policy
- **Final Train Avg**: Mean reward over last 100 training episodes
- **Eval Avg**: Mean reward ± standard deviation over 10 evaluation episodes

**Visualizations**:
- `learning_curves.png`:
  - Episodic returns plotted against episode number for all three algorithms
  - Moving average (window=20) showing smoothed learning progression
  - Raw episodic rewards for granular performance view
  - Solved threshold line at 200 reward

- `{algorithm}_agent.gif`:
  - Visual demonstration of trained agent's policy (3 episodes per GIF)
  - Generated for DQN, DDQN, and PER algorithms

**Console Output**:
During training, the script prints:
- Episode progress updates every 20 episodes
- Moving average reward (last 20 episodes)
- Current epsilon value (exploration rate)
- Current learning rate (with scheduler)
- Elapsed training time

After training, evaluation metrics include:
- Per-episode greedy policy performance
- Average inference time per episode
- Final performance summary with statistics

Test results are organized by episode count (500, 1000, 2500, 5000, 10000) in separate directories for comparative analysis.

## Author

Filipe Ramos (CIS2719 Coursework 2)

## License

This project is for educational purposes.
