# Distributed Reinforcement Learning Algorithms 🤖

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A collection of PyTorch implementations of distributed reinforcement learning algorithms, designed with clarity and educational value in mind.

## 🚀 Features

- **Clear, documented implementations** of state-of-the-art distributed RL algorithms
- **Educational focus** with detailed comments and explanations
- **Modular design** for easy extension and modification
- **Comprehensive logging** with Weights & Biases integration
- **Configurable experiments** using Hydra

## 📦 Currently Implemented

### Asynchronous DQN (A3C-DQN variant)
An implementation based on ["Asynchronous Methods for Deep Reinforcement Learning"](https://arxiv.org/abs/1602.01783) (Mnih et al., 2016).

**Key Features:**
- ✨ Parallel training across multiple workers
- 🎯 Shared model architecture with target network
- 🔍 Epsilon-greedy exploration with annealing
- 📉 Learning rate annealing
- 📊 Wandb integration for experiment tracking

## 🏗️ Project Structure
```
distributed_rl_algos/
├── algorithms/           # Algorithm implementations
│   └── adqn.py          # Asynchronous DQN implementation
├── common/              # Shared utilities
│   ├── network_factory.py
│   └── utils.py
└── config/             # Configuration files
    └── adqn.yaml       # ADQN hyperparameters
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/distributed_rl_algos.git
cd distributed_rl_algos
```

2. Install dependencies with Poetry:
```bash
poetry install
poetry shell
```

## 📋 Requirements

- Python 3.8+
- PyTorch
- Gymnasium
- Weights & Biases
- Poetry
- Hydra

## 🚦 Quick Start

Run the algorithm with default configurations:

```bash
python examples/train.py algorithm=adqn
```

### Configuration Options

- `num_workers`: Number of parallel workers
- `network.architecture`: Neural network architecture
- `training.lr`: Learning rate
- `training.epsilon`: Exploration parameters
- `training.duration`: Training duration

## 💻 Development

```bash
# Format code
poetry run black .
poetry run isort .

# Run linting
poetry run pylint distributed_rl_algos

# Run tests
poetry run pytest
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@misc{distributed_rl_algos,
  author = {Your Name},
  title = {Distributed Reinforcement Learning Algorithms},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/distributed_rl_algos}
}
```

## 🙏 Acknowledgments

- The A3C-DQN implementation is based on the work by Mnih et al.
- Thanks to all contributors who have helped improve this project