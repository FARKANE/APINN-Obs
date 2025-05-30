"""
Training utilities for PINN observer.
"""

import os
import time
import torch
import torch.nn.functional as F

from .data_generation import generate_dataset
from .models import PINNObserver
from .residuals import RESIDUAL_MAP, MEAS_IDX


def train(system, epochs, lr, Nc, Nbc, t0, t_end, seed, log_interval, 
          weight_ic, weight_pde, weight_y, checkpoint_dir="checkpoints"):
    """
    Train a PINN observer for a specified dynamical system.
    
    Parameters:
    -----------
    system : str
        Name of the dynamical system to train on
    epochs : int
        Number of training epochs
    lr : float
        Learning rate for Adam optimizer
    Nc : int
        Number of collocation points
    Nbc : int
        Number of boundary condition points
    t0 : float
        Initial time
    t_end : float
        Final time
    seed : int
        Random seed for reproducibility
    log_interval : int
        Frequency of logging (epochs)
    weight_ic : float
        Weight for initial condition loss
    weight_pde : float
        Weight for PDE residual loss
    weight_y : float
        Weight for measurement loss
    checkpoint_dir : str, optional
        Directory to save model checkpoints (default: "checkpoints")
    
    Returns:
    --------
    model : PINNObserver
        Trained model
    best_loss : float
        Best achieved total loss
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate training data
    print(f"Generating dataset for '{system}'...")
    t_col, x_col, y_col, t_bc, x_bc0 = generate_dataset(
        system, t0, t_end, Nc, Nbc, seed
    )
    
    # Convert to tensors
    t_col_t = torch.tensor(t_col, dtype=torch.float32, device=device).unsqueeze(1)
    x_col_t = torch.tensor(x_col, dtype=torch.float32, device=device)
    t_bc_t = torch.tensor(t_bc, dtype=torch.float32, device=device).unsqueeze(1)
    x_bc0_t = torch.tensor(x_bc0, dtype=torch.float32, device=device)

    # Initialize model
    nx = x_col.shape[1]
    meas_idxs = MEAS_IDX[system]
    m = len(meas_idxs)
    model = PINNObserver(nx, m).to(device)
    
    print(f"Model architecture:")
    print(f"  State dimension: {nx}")
    print(f"  Measurement dimension: {m}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Setup checkpoints
    best_loss = float("inf")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{system}.pth")

    # Get residual function
    resid_fn = RESIDUAL_MAP[system]
    
    print(f"\nStarting training for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        # Initial condition loss
        xc0_hat = model(t_bc_t)[:, :nx]
        loss_ic = F.mse_loss(xc0_hat, x_bc0_t)

        # Physics residual loss
        t_col_t.requires_grad_(True)
        res = resid_fn(t_col_t, x_col_t, model, device)
        loss_phys = F.mse_loss(res, torch.zeros_like(res))

        # Measurement loss
        xc_pred = model(t_col_t)[:, :nx]
        pred_meas = xc_pred[:, meas_idxs]
        true_meas = x_col_t[:, meas_idxs]
        loss_out = F.mse_loss(pred_meas, true_meas)

        # Total loss
        loss = weight_ic * loss_ic + weight_pde * loss_phys + weight_y * loss_out
        loss.backward()
        optimizer.step()

        # Save best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), checkpoint_path)

        # Logging
        if epoch == 1 or epoch % log_interval == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:6d}/{epochs} | "
                  f"Total: {loss.item():.3e} | "
                  f"IC: {loss_ic.item():.3e} | "
                  f"PDE: {loss_phys.item():.3e} | "
                  f"Meas: {loss_out.item():.3e} | "
                  f"Time: {elapsed:.1f}s")

    total_time = time.time() - start_time
    print(f"\n✓ Training completed in {total_time:.1f}s ({total_time/60:.2f} min)")
    print(f"✓ Best loss: {best_loss:.3e}")
    print(f"✓ Model saved to: {checkpoint_path}")
    
    # Load best model
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    return model, best_loss


def load_trained_model(system, checkpoint_dir="checkpoints", device=None):
    """
    Load a pre-trained PINN observer model.
    
    Parameters:
    -----------
    system : str
        Name of the system
    checkpoint_dir : str, optional
        Directory containing checkpoints
    device : torch.device, optional
        Device to load model on (auto-detected if None)
        
    Returns:
    --------
    model : PINNObserver
        Loaded model in evaluation mode
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine model dimensions from system
    state_dims = {
        "reverse": 2, "satellite": 3, "harmonic": 3,
        "induction_motor": 5, "academic": 2, "modified_academic": 2
    }
    
    nx = state_dims[system]
    meas_idxs = MEAS_IDX[system]
    m = len(meas_idxs)
    
    # Load model
    model = PINNObserver(nx, m).to(device)
    checkpoint_path = os.path.join(checkpoint_dir, f"{system}.pth")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found for system '{system}' at {checkpoint_path}")
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    print(f"✓ Loaded trained model for '{system}' from {checkpoint_path}")
    return model