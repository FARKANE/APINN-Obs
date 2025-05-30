"""
Neural network models for PINN observer implementation.
"""

import torch
import torch.nn as nn


class PINNObserver(nn.Module):
    """
    Physics-Informed Neural Network Observer.
    
    A neural network that simultaneously estimates system states and
    observer gain matrix for nonlinear dynamical systems.
    
    Parameters:
    -----------
    state_dim : int
        Dimension of the state vector
    output_dim : int
        Dimension of the measurement vector
    hidden : tuple, optional
        Hidden layer sizes (default: (20,)*9)
    """
    
    def __init__(self, state_dim, output_dim, hidden=(20,)*9):
        super().__init__()
        
        self.state_dim = state_dim
        self.output_dim = output_dim
        
        # Build the main network
        layers = []
        inp = 1  # Input is time (scalar)
        
        for h in hidden:
            layers += [nn.Linear(inp, h), nn.Tanh()]
            inp = h
            
        self.net = nn.Sequential(*layers)
        
        # Output layer: state_dim (for states) + state_dim*output_dim (for observer gain L)
        self.out_layer = nn.Linear(inp, state_dim + state_dim*output_dim)

    def forward(self, t):
        """
        Forward pass of the PINN observer.
        
        Parameters:
        -----------
        t : torch.Tensor
            Time points, shape (N, 1) or (N,)
            
        Returns:
        --------
        torch.Tensor
            Combined output containing estimated states and observer gain matrix.
            Shape: (N, state_dim + state_dim*output_dim)
            First state_dim columns: estimated states
            Remaining columns: flattened observer gain matrix L
        """
        h = self.net(t)
        return self.out_layer(h)
    
    def get_states_and_gain(self, t):
        """
        Extract states and observer gain matrix separately.
        
        Parameters:
        -----------
        t : torch.Tensor
            Time points
            
        Returns:
        --------
        states : torch.Tensor
            Estimated states, shape (N, state_dim)
        gain_matrix : torch.Tensor
            Observer gain matrix, shape (N, state_dim, output_dim)
        """
        out = self.forward(t)
        states = out[:, :self.state_dim]
        gain_flat = out[:, self.state_dim:]
        gain_matrix = gain_flat.view(-1, self.state_dim, self.output_dim)
        return states, gain_matrix
