# LunarLander-v3 Reinforcement Learning Comparison

## Overview

This project implements and compares three value-based deep reinforcement learning algorithms for the LunarLander-v3 environment from OpenAI Gymnasium. The implementation is part of coursework for Foundations of Robotics & AI (CIS2719 Coursework 2).

**Author:** Ethan Hulme

## Algorithms Implemented

The project compares the following algorithms:

1. **DQN (Deep Q-Network)** - Standard value-based RL using experience replay and target networks
2. **DDQN (Double DQN)** - Addresses overestimation bias by decoupling action selection from evaluation
3. **PER (Prioritized Experience Replay)** - Samples transitions based on TD-error magnitude combined with DDQN targets

## Environment

- **Task:** LunarLander-v3 (discrete action space)
- **State Space:** 8-dimensional continuous observations
- **Action Space:** 4 discrete actions (do nothing, fire left engine, fire main engine, fire right engine)
- **Solved Threshold:** Average reward ≥ 200 over 100 consecutive episodes

## Architecture

### Q-Network
- 2-layer fully connected neural network
- Hidden dimension: 128 units per layer
- Activation: ReLU
- Output: Q-values for each action

### Hyperparameters
- Learning rate: 1e-3
- Discount factor (γ): 0.99
- Batch size: 64
- Replay buffer capacity: 100,000
- Minimum buffer size before training: 10,000
- Target network update frequency: 1,000 steps
- Epsilon decay: 250,000 frames (1.0 → 0.05)
- Loss function: Smooth L1 (Huber loss)
- Optimizer: Adam

### PER-Specific Parameters
- Alpha (priority exponent): 0.6
- Beta (importance sampling): 0.4 → 1.0 (annealed over 200,000 frames)

## Project Structure

```
LunarLander-V2/
├── LunarLander-V2.py       # Main implementation
├── README.md               # This file
└── Test Results/           # Experimental results
    ├── Test 1 (600 Eps)/
    ├── Test 2 (600 Eps)/
    ├── Test 3 (600 Eps)/
    ├── Test 4 (600 Eps)/
    ├── Test 5 (600 Eps)/
    ├── Test 6 (2000 Eps)/
    ├── Test 7 (2000 Eps)/
    ├── Test 8 (2000 Eps)/
    ├── Test 9 (5000 Eps)/
    └── Test 10 (10000 Eps)/
```

Each test directory contains:
- `learning_curves.png` - Training curves for all three algorithms
- `dqn_agent.gif` - Visualization of DQN policy
- `ddqn_agent.gif` - Visualization of DDQN policy
- `per_agent.gif` - Visualization of PER policy

## Requirements

```bash
gymnasium[box2d]
numpy
torch
matplotlib
imageio
```

## Installation

```bash
pip install gymnasium[box2d] numpy torch matplotlib imageio
```

## Usage

Run the main script to train all three agents and generate results:

```bash
python LunarLander-V2.py
```

The script will:
1. Train DQN, DDQN, and PER agents sequentially
2. Output progress every 20 episodes
3. Generate a comparative learning curve plot (`learning_curves.png`)
4. Create policy visualization GIFs for each algorithm

### Configuring Training Duration

Modify the `NUM_EPISODES` variable in the main block:

```python
NUM_EPISODES = 600  # Adjust as needed
```

## Implementation Details

### DQN
- Uses standard experience replay with uniform sampling
- TD target: `r + γ * max_a' Q_target(s', a')`

### DDQN
- Decouples action selection (online network) from action evaluation (target network)
- TD target: `r + γ * Q_target(s', argmax_a' Q_online(s', a'))`
- Reduces overestimation bias present in DQN

### PER
- Samples transitions proportionally to TD-error magnitude
- Uses importance sampling weights to correct for bias
- Combines prioritized sampling with DDQN-style targets
- Priorities updated after each gradient step

## Results

The project includes 10 experimental runs with varying episode counts:
- Tests 1-5: 600 episodes (initial experiments)
- Tests 6-8: 2,000 episodes (extended training)
- Test 9: 5,000 episodes
- Test 10: 10,000 episodes (maximum training duration)

Results demonstrate the relative performance and sample efficiency of each algorithm across different training horizons.

## Key Features

- Epsilon-greedy exploration with exponential decay
- Target network for stability
- Gradient clipping (max norm: 10.0) for training stability
- Reproducible results via seeding
- Smooth L1 loss for robustness to outliers
- Moving average smoothing for learning curves

## License

This project is submitted as coursework for academic evaluation.

## References

- ABELSON, H., SUSSMAN, G.J. and SUSSMAN, J., 1996. *Structure and interpretation of classical mechanics*. Cambridge, MA: MIT Press.

- BELLMAN, R., 1957. *Dynamic programming*. Princeton, NJ: Princeton University Press.

- BROCKMAN, G. et al., 2016. OpenAI Gym. arXiv preprint [online]. Available from: [https://arxiv.org/abs/1606.01540](https://arxiv.org/abs/1606.01540?-utm_source=chatgpt.com) [Accessed 6 December 2025].

- FUJIMOTO, S., VAN HOOF, H. and MEGER, D., 2018. Addressing function approximation error in actor–critic methods. In: *Proceedings of the 35th International Conference on Machine Learning (ICML 2018)*. Stockholm: PMLR, pp. 1587–1596.

- GAUDET, B., CROSSLEY, W. and GUSTAFSON, D., 2021. Autonomous precision landing using reinforcement learning. *Journal of Guidance, Control, and Dynamics*, 44(3), pp. 523–537.

- HESSEL, M. et al., 2018. Rainbow: combining improvements in deep reinforcement learning. In: *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI-18)*. New Orleans, LA: AAAI Press, pp. 3215–3222.

- KAELBLING, L.P., LITTMAN, M.L. and MOORE, A.W., 1996. Reinforcement learning: a survey. *Journal of Artificial Intelligence Research*, 4, pp. 237–285.

- KOBER, J., BAGNELL, J.A. and PETERS, J., 2013. Reinforcement learning in robotics: a survey. *The International Journal of Robotics Research*, 32(11), pp. 1238–1274.

- KOCH, W., MANCUSO, R. and WEST, R., 2019. Reinforcement learning for spacecraft landing control. In: *AIAA Scitech 2019 Forum*. Reston, VA: American Institute of Aeronautics and Astronautics. Available from: https://doi.org/10.2514/6.2019-0478.

- MNIH, V. et al., 2013. Playing Atari with deep reinforcement learning. arXiv preprint [online]. Available from: [https://arxiv.org/abs/1312.5602](https://arxiv.org/abs/1312.5602?utm_source=chatgpt.com) [Accessed 6 December 2025].

- MNIH, V. et al., 2015. Human-level control through deep reinforcement learning. *Nature*, 518(7540), pp. 529–533.

- SCHAUL, T. et al., 2016. Prioritized experience replay. In: *Proceedings of the International Conference on Learning Representations (ICLR 2016)*. San Juan, Puerto Rico.

- SUTTON, R.S. and BARTO, A.G., 2018. *Reinforcement learning: an introduction*. 2nd ed. Cambridge, MA: MIT Press.

- TOWERS, M. et al., 2023. Gymnasium: a standard API for reinforcement learning environments. *Journal of Machine Learning Research*, 24(229), pp. 1–8.

- VAN HASSELT, H., 2010. Double Q-learning. In: *Advances in Neural Information Processing Systems 23 (NeurIPS 2010)*. Vancouver, Canada, pp. 2613–2621.

- VAN HASSELT, H., GUEZ, A. and SILVER, D., 2016. Deep reinforcement learning with double Q-learning. In: *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI-30)*. Phoenix, AZ: AAAI Press, pp. 2094–2100.

- WANG, Z. et al., 2016. Dueling network architectures for deep reinforcement learning. In: *Proceedings of the 33rd International Conference on Machine Learning (ICML 2016)*. New York, NY: PMLR, pp. 1995–2003.

- WILLIAMS, R.J., 1992. Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8, pp. 229–256.

# Lunar-Lander
# Lunar-Lander
