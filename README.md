# PINN Observer

A Physics-Informed Neural Network (PINN) based observer implementation for nonlinear dynamical systems state estimation.

## Overview

This package implements PINN observers that can simultaneously estimate system states and learn observer gain matrices for various nonlinear dynamical systems. The approach combines the power of neural networks with physics-based constraints to achieve accurate state estimation.

## Supported Systems

- **Reverse Duffing**: Reverse Duffing oscillator system
- **Satellite**: Satellite attitude dynamics
- **Harmonic**: Harmonic oscillator system  
- **Induction Motor**: Induction motor dynamic model
- **Academic**: Academic example system
- **Modified Academic**: Modified academic example system

## Installation

### From Source

1. Clone the repository:
```bash
git clone https://github.com/FARKANE/pinn_observer.git
cd pinn_observer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

### Training a PINN Observer

```python
from pinn_observer import train

# Train on the modified academic system
model, best_loss = train(
    system='modified_academic',
    epochs=200000,
    lr=1e-3,
    Nc=10000,      # Number of collocation points
    Nbc=10,        # Number of boundary condition points
    t0=0.0,
    t_end=20.0,
    seed=9,
    log_interval=1000,
    weight_ic=1.0,    # Initial condition loss weight
    weight_pde=1.0,   # PDE residual loss weight
    weight_y=1.0,     # Measurement loss weight
)
```

### Visualizing Results

```python
from pinn_observer import plot_pinn_observer_separate_figures

# Generate publication-quality plots
plot_pinn_observer_separate_figures(
    system='modified_academic',
    ckpt_path='checkpoints/modified_academic.pth',
    t0=0.0,
    t_end=20.0,
    N=1000,
    out_dir='figures',
    use_log_error=False
)
```

### Using Pre-trained Models

```python
from pinn_observer.train import load_trained_model

# Load a pre-trained model
model = load_trained_model('modified_academic')

# Use for inference
import torch
t = torch.linspace(0, 20, 1000).unsqueeze(1)
states, gain_matrix = model.get_states_and_gain(t)
```

## Examples

The `examples/` directory contains ready-to-run scripts:

- `train_modified_academic.py`: Train a PINN observer
- `plot_modified_academic.py`: Visualize trained model results

Run examples:
```bash
cd examples
python train_modified_academic.py
python plot_modified_academic.py
```

## Package Structure

```
pinn_observer/
├── __init__.py           # Package initialization
├── data_generation.py    # System definitions and data generation
├── models.py            # PINNObserver neural network model
├── residuals.py         # Physics-informed residual functions
├── train.py             # Training utilities
└── plotting.py          # Visualization utilities
```

## Key Features

- **Multiple System Support**: Six different nonlinear dynamical systems
- **Physics-Informed Training**: Enforces system dynamics through residual loss terms
- **Flexible Architecture**: Configurable neural network architecture
- **Publication-Quality Plots**: Professional visualization tools
- **Checkpoint Management**: Automatic model saving and loading
- **GPU Support**: Automatic CUDA detection and usage

## Configuration

Key training parameters:

- `epochs`: Number of training iterations
- `lr`: Learning rate for Adam optimizer
- `Nc`: Number of collocation points for physics loss
- `Nbc`: Number of boundary condition points
- `weight_ic`: Weight for initial condition loss
- `weight_pde`: Weight for physics residual loss  
- `weight_y`: Weight for measurement loss

## Performance

The PINN observer achieves excellent state estimation accuracy across all supported systems. Training typically converges within 200,000 epochs, with final losses on the order of 1e-6 to 1e-4.

## Requirements

- Python ≥ 3.7
- PyTorch ≥ 1.9.0
- NumPy ≥ 1.19.0
- SciPy ≥ 1.5.0
- Matplotlib ≥ 3.3.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{pinn_observer,
  title={PINN Observer: Physics-Informed Neural Network Observer for Nonlinear Systems},
  author={Your Name},
  year={2024},
  url={https://github.com/FARKANE/pinn_observer}
}
```

## Acknowledgments

This implementation is based on physics-informed neural network methodologies for observer design in nonlinear dynamical systems.
