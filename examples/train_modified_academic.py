#!/usr/bin/env python3
"""
Example script for training a PINN observer on the modified academic system.

This script demonstrates how to use the pinn_observer package to train
a Physics-Informed Neural Network observer for state estimation.
"""

import os
import sys

# Add the parent directory to the path to import pinn_observer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinn_observer import train


def main():
    """Train PINN observer for modified academic system."""
    
    # Training configuration
    config = {
        'system': 'modified_academic',
        'epochs': 200000,
        'lr': 1e-3,
        'Nc': 10000,              # Number of collocation points
        'Nbc': 10,                # Number of boundary condition points
        't0': 0.0,
        't_end': 20.0,
        'seed': 9,
        'log_interval': 1000,
        'weight_ic': 1.0,         # Weight for initial condition loss
        'weight_pde': 1.0,        # Weight for PDE residual loss
        'weight_y': 1.0,          # Weight for measurement loss
    }
    
    print("="*60)
    print("PINN Observer Training - Modified Academic System")
    print("="*60)
    print(f"System: {config['system']}")
    print(f"Epochs: {config['epochs']:,}")
    print(f"Learning Rate: {config['lr']}")
    print(f"Collocation Points: {config['Nc']:,}")
    print(f"Time Interval: [{config['t0']}, {config['t_end']}]")
    print("-"*60)
    
    # Train the model
    model, best_loss = train(
        system=config['system'],
        epochs=config['epochs'],
        lr=config['lr'],
        Nc=config['Nc'],
        Nbc=config['Nbc'],
        t0=config['t0'],
        t_end=config['t_end'],
        seed=config['seed'],
        log_interval=config['log_interval'],
        weight_ic=config['weight_ic'],
        weight_pde=config['weight_pde'],
        weight_y=config['weight_y']
    )
    
    print("="*60)
    print("Training Completed Successfully!")
    print(f"Final Best Loss: {best_loss:.6e}")
    print("="*60)


if __name__ == "__main__":
    main()