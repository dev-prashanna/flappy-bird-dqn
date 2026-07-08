# Flappy Bird Game

[![Python](https://img.shields.io/badge/Python-3.10+-yellow)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Pygame](https://img.shields.io/badge/Pygame-2.0+-blue)](https://www.pygame.org/)

A simple Flappy Bird clone built with Python and Pygame.

## Overview

This is a classic Flappy Bird game implementation using Pygame. Control a bird to fly through pipes without hitting them.

## Features

- Smooth bird physics with gravity and jump mechanics
- Procedurally generated pipes
- Collision detection
- Score tracking
- Retro-style graphics

## Installation

### Prerequisites

- Python 3.10+
- Pygame

### Setup

```bash
# Clone the repository
git clone https://github.com/dev-prashanna/Fun_Projects.git
cd Fun_Projects

# Install dependencies
pip install pygame
```

## Usage

```bash
# Run the game
python main.py
```

### Controls

| Key | Action |
|-----|--------|
| SPACE | Jump |
| ESC | Quit |

## Game Mechanics

- **Bird**: Yellow circle with gravity physics
- **Pipes**: Green rectangles with fixed gap
- **Collision**: Game ends when bird hits pipe or screen boundary
- **Speed**: Pipes move left at constant speed

## Project Structure

```
Fun_Projects/
├── main.py        # Game implementation
├── LICENSE        # MIT License
└── README.md      # This file
```

## Technical Details

- **Window**: 800x600 pixels
- **FPS**: 60 frames per second
- **Physics**: Simple gravity (0.5) with jump velocity (-8)
- **Pipe Gap**: 150 pixels
- **Pipe Speed**: 3 pixels per frame

## Future Improvements

- [ ] Add score display
- [ ] Implement pipe recycling for endless gameplay
- [ ] Add sound effects
- [ ] Create menu screen
- [ ] Add difficulty progression
- [ ] Implement bird animation

## References

- [Pygame Documentation](https://www.pygame.org/docs/)
- [Flappy Bird Game Design](https://en.wikipedia.org/wiki/Flappy_Bird)

## License

This project is licensed under the MIT License -- see the [LICENSE](LICENSE) file for details.

## Author

**Prashanna Tiwari**
- GitHub: [@dev-prashanna](https://github.com/dev-prashanna)
- LinkedIn: [Prashanna Tiwari](https://www.linkedin.com/in/prashanna-tiwari-1b9a01398/)
