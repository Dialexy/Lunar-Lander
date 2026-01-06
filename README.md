# LunarLander-v3 Deep Reinforcement Learning

A showcase of deep reinforcement learning algorithms on the Gymnasium LunarLander-v3 environment.

## Overview

This project implements and compares three Q-learning algorithms:

**Algorithm overview found in the LunarLander-V2.py header comment**

## Features

- Automated training with configurable episodes (line: 452)
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
# This if for Unix Systems, see offical documentation for windows specific information
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

Run the training script for full episode tests:
```bash
python LunarLander-V2.py
# Run via terminal (Extensions such as "code runner" will not work)
```

Run the training script for early-stoppage tests:
```bash
python LunarLander-V2-Optimised.py
```

Configure training parameters in the main block:
```python
NUM_EPISODES = 10000  # Adjust episode count (line: 452)
TEST_NAME = "Test 10000eps (Post changes)"  # Name your test run
```

## Project Structure

```
.
├── LunarLander-V2.py                     # Optimised training script
├── LunarLander-V2-Optimised.py           # Main training script
├── README.md
├── Gifs/
│   ├── Full-Test-Runs/         # Pre-optimisation test results (Runs episodes to completion)
│   │   ├── Test 500eps/
│   │   ├── Test 1000eps/
│   │   ├── Test 2500eps/
│   │   ├── Test 5000eps/
│   │   └── Test 10000eps/
│   └── Optimised-Test-Runs/    # Post-optimisation test results (Runs until episodic return stagnates)
│       ├── Test 500eps/
│       ├── Test 1000eps/
│       ├── Test 2500eps/
│       ├── Test 5000eps/
│       └── Test 10000eps/
└── results/                     # Generated during training
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
- Early stopping (Only for LunarLander-V2-Optimised.py)

## Training Details

- **Optimizer**: Adam (lr=1e-3)
- **Batch Size**: 64
- **Gamma**: 0.99
- **Buffer Capacity**: 100,000
- **Target Update Frequency**: 1,000 steps
- **Epsilon Decay**: Adaptive based on episode count
- **Loss Function**: Smooth L1 (Huber)

## Output

Each training run generates:
- `learning_curves.png` - Episodic return vs episodes for all algorithms
- `dqn_agent.gif` - DQN policy visualization
- `ddqn_agent.gif` - DDQN policy visualization
- `per_agent.gif` - PER policy visualization

## Results

The environment is considered solved when the agent achieves an average reward of ≥200 over 100 consecutive episodes.

Test results are organized by episode count (500, 1000, 2500, 5000, 10000) and include both pre-optimisation and post-optimisation runs for performance comparison.

## Author

Filipe Ramos (CIS2719 Coursework 2)

## License

This project is for educational purposes.
