# Flappy Bird DQN

[![Python](https://img.shields.io/badge/Python-3.10+-yellow)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Pygame](https://img.shields.io/badge/Pygame-2.0+-blue)](https://www.pygame.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)

A Flappy Bird clone with a Double DQN agent that learns to play automatically.

## Overview

This project includes both a playable Flappy Bird game and a reinforcement learning agent that learns to play the game using Double Deep Q-Network (Double DQN).

## Features

- Playable Flappy Bird game with keyboard controls
- Double DQN agent that learns to play autonomously
- Experience replay buffer for stable training
- Soft target network updates
- Gradient clipping for training stability
- Best model checkpoint saving

## Installation

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/dev-prashanna/flappy-bird-dqn.git
cd flappy-bird-dqn

# Install dependencies
pip install pygame torch numpy
```

## Usage

### Play the Game

```bash
python main.py
```

| Key | Action |
|-----|--------|
| SPACE | Jump |

### Train the AI Agent

```bash
python train_net.py
```

Training runs for 2000 episodes. The best model is saved to `best_model.pth`.

## Project Structure

```
flappy-bird-dqn/
├── main.py            # Playable game
├── flappyenv.py       # Gym-style environment wrapper
├── neural_net.py      # DQN neural network architecture
├── double_dqn.py      # Double DQN agent
├── replay_buffer.py   # Experience replay buffer
├── train_net.py       # Training script
├── LICENSE            # MIT License
└── README.md          # This file
```

## How It Works

### State Representation (5 features)

| Feature | Description |
|---------|-------------|
| Bird Y position | Normalized by screen height |
| Bird velocity | Normalized by 10 |
| Horizontal distance to next pipe | Normalized by screen width |
| Top pipe height | Normalized by screen height |
| Bottom pipe Y position | Normalized by screen height |

### Actions

- `0`: Do nothing (fall)
- `1`: Jump

### Training Details

- **Algorithm**: Double DQN
- **Optimizer**: Adam (lr=0.0005)
- **Loss**: Smooth L1 (Huber)
- **Batch size**: 64
- **Replay buffer**: 100,000 experiences
- **Epsilon decay**: 0.995 per episode (1.0 → 0.01)
- **Target network update**: Every 10 episodes
- **Gradient clipping**: max_norm=1.0

## Technical Details

- **Window**: 800x600 pixels
- **FPS**: 60 frames per frame
- **Physics**: Gravity 0.5, jump velocity -8
- **Pipe Gap**: 150 pixels
- **Pipe Speed**: 3 pixels per frame

## License

This project is licensed under the MIT License -- see the [LICENSE](LICENSE) file for details.

## Author

**Prashanna Tiwari**
- GitHub: [@dev-prashanna](https://github.com/dev-prashanna)
