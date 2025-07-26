# DIAL (Differentiable Inter-Agent Learning)

This project implements DIAL, a framework for learning to communicate between agents using differentiable communication channels. The implementation focuses on a "Guess My Number" game where agents must learn to communicate effectively to succeed.

## Project Structure

```
├── checkpoints/           # Saved model checkpoints
├── games/                # Environment implementations
│   └── ColorDigitGuessEnv.py
├── guess_my_number_example/
│   ├── config.yaml       # Configuration parameters
│   ├── dial_algorithm.py # Core DIAL implementation
│   ├── guess_my_number.py # Game environment
│   ├── main.py          # Training script
│   └── playfull_episode.py # Episode execution logic
├── networks/             # Neural network architectures
│   ├── simple_network.py # AgentNet and AgentNet2 implementations
│   └── C_Net.py         # Complex network implementation
└── utils/               # Utility functions
```

## Network Architectures

### 1. AgentNet
- Simple RNN-based architecture using GRU cells
- Direct concatenation of observation and message
- Two output heads: Q-values and messages

### 2. AgentNet2
- Enhanced version with embedding layers
- Separate processing for observations and messages
- Uses GRU for temporal dependencies

### 3. C_Net
- Complex architecture with multiple GRU layers
- Includes embeddings for actions, messages, and agent IDs
- Designed for more sophisticated communication patterns

## Core Components

### Discrete Regularization Unit (DRU)
```python
def dru(m, sigma=2.0, training=True):
    if training:
        noise = torch.randn_like(m) * sigma
        return torch.sigmoid(m + noise)
    else:
        return (m > 0).float()
```
Converts continuous messages to discrete values during training while maintaining differentiability.

### DIAL Algorithm
The core algorithm implements:
- Experience collection through agent interaction
- Message passing between agents
- Gradient computation for both direct rewards and message impacts
- Batch processing for stable training

## Training Process

1. Agents interact in episodes of the Guess My Number game
2. Messages are passed between agents using DRU
3. Gradients are computed considering:
   - Direct reward optimization
   - Impact of messages on other agents
4. Parameters are updated using batched gradients
5. Models are periodically saved based on performance

## Usage

1. Configure parameters in `config.yaml`
2. Run training:
```bash
python guess_my_number_example/main.py
```

## Results Visualization

The training process generates:
- `average_r.png`: Plot of average rewards
- `loss_vec.png`: Plot of training loss
- Checkpoints of best-performing models

## Requirements

- PyTorch
- NumPy
- Matplotlib
