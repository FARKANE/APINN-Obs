
"""
Plotting utilities for PINN observer results visualization.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter
from scipy.integrate import odeint

from .data_generation import SYSTEM_MAP
from .models import PINNObserver
from .residuals import MEAS_IDX


# Color scheme
COLORS = {
    "true":      "red",
    "predicted": "green", 
    "error":     "blue",
}

# Configure matplotlib for publication-quality plots
plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         9,
    'axes.labelsize':    10,
    'axes.labelweight':  'bold',
    'axes.titlesize':    11,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   11,
    'lines.linewidth':   2.0,
    'figure.dpi':        300,
    'savefig.format':    'pdf',
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.02,
    'axes.linewidth':    1.0,
    'grid.linewidth':    0.6,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.minor.width': 0.8,
    'ytick.minor.width': 0.8,
})


def get_system_initial_conditions():
    """Get initial conditions for all systems."""
    return {
        "reverse":          np.array([2, -1]),
        "satellite":        np.array([2, -1, 3]),
        "harmonic":         np.array([2, -1, 3]),
        "induction_motor":  np.array([1, 0, 2, 3, 0]),
        "academic":         np.array([2, -1]),
        "modified_academic": np.array([2, -1]),
    }


def plot_pinn_observer_separate_figures(system, ckpt_path, t0, t_end, N=1000, 
                                       out_dir="figures", use_log_error=True):
    """
    Generate separate publication-quality figures for trajectories and errors.
    
    Parameters:
    -----------
    system : str
        Name of the dynamical system
    ckpt_path : str
        Path to the trained model checkpoint
    t0 : float
        Initial time
    t_end : float
        Final time
    N : int, optional
        Number of evaluation points (default: 1000)
    out_dir : str, optional
        Output directory for figures (default: "figures")
    use_log_error : bool, optional
        Whether to use logarithmic scale for error plots (default: True)
    
    Returns:
    --------
    None
        Saves PDF figures to the specified output directory
    """
    # Setup
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load system and initial conditions
    sys_fn = SYSTEM_MAP[system]
    x0_map = get_system_initial_conditions()
    x0 = x0_map[system]
    nx = x0.size
    
    # Load model
    meas_idxs = MEAS_IDX[system]
    m_val = len(meas_idxs)
    model = PINNObserver(nx, m_val).to(device)
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Generate evaluation data
    t = np.linspace(t0, t_end, N)
    
    # Compute true trajectory
    if system == "induction_motor":
        x_true = odeint(sys_fn, x0, t, args=(0.0,))
    else:
        x_true = odeint(sys_fn, x0, t)

    # Compute predicted trajectory
    t_tensor = torch.tensor(t, dtype=torch.float32, device=device).unsqueeze(1)
    with torch.no_grad():
        out = model(t_tensor)
    x_pred = out[:, :nx].cpu().numpy()
    
    # Compute errors
    err = np.abs(x_true - x_pred)

    # Setup figure layout
    ncols_layout = 1
    nrows_layout = nx
    fig_width = 6.0
    fig_height = 1.5 * nrows_layout + 0.7

    # --- Trajectories Figure ---
    fig_traj, axs_traj = plt.subplots(nrows_layout, ncols_layout,
                                      figsize=(fig_width, fig_height),
                                      sharex=True, squeeze=False)
    
    lines_for_legend = []
    labels_for_legend = []

    for i in range(nx):
        ax = axs_traj[i, 0]

        line_true, = ax.plot(t, x_true[:, i], label="True", 
                           color=COLORS["true"], linewidth=2.0)
        line_pred, = ax.plot(t, x_pred[:, i], '--', label="Predicted", 
                           color=COLORS["predicted"], linewidth=2.0)
        
        ax.set_ylabel(f"$x_{{{i+1}}}(t)$")
        ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.3)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.tick_params(direction='in', top=True, right=True)

        if i == 0:
            lines_for_legend.extend([line_true, line_pred])
            labels_for_legend.extend(["True", "Predicted"])

    # Set x-axis label only on bottom subplot
    axs_traj[nrows_layout - 1, 0].set_xlabel("Time (s)")

    # Add legend
    fig_traj.legend(lines_for_legend, labels_for_legend,
                    loc='lower center', ncol=2, bbox_to_anchor=(0.5, 1.0),
                    frameon=True, fontsize=plt.rcParams['legend.fontsize'])
    fig_traj.tight_layout(rect=[0, 0.02, 1, 1.0], h_pad=1.2)

    # Save trajectories figure
    out_path_traj = os.path.join(out_dir, f"{system}_trajectories.pdf")
    fig_traj.savefig(out_path_traj)
    print(f"✓ Saved trajectories to: {out_path_traj}")

    # --- Errors Figure ---
    fig_err, axs_err = plt.subplots(nrows_layout, ncols_layout,
                                    figsize=(fig_width, fig_height),
                                    sharex=True, squeeze=False)
    
    for i in range(nx):
        ax = axs_err[i, 0]
        current_err_values = err[:, i]

        if use_log_error:
            error_floor = 1e-8
            plot_err_vals = np.maximum(current_err_values, error_floor)
            ax.plot(t, plot_err_vals, color=COLORS["error"], linewidth=2.0)
            ax.set_yscale('log')
            ax.set_ylabel(f"$|e_{{{i+1}}}(t)|$")
            
            min_val_to_show = error_floor
            max_val_observed = np.max(plot_err_vals)
            if max_val_observed > min_val_to_show * 1.1:
                ax.set_ylim(bottom=min_val_to_show * 0.5, top=max_val_observed * 2)
            else:
                ax.set_ylim(bottom=min_val_to_show * 0.5, top=min_val_to_show * 100)
        else:
            ax.plot(t, current_err_values, color=COLORS["error"], linewidth=2.0)
            ax.set_ylabel(f"$|e_{{{i+1}}}(t)|$")
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.3)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        if not use_log_error:
            ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.tick_params(direction='in', top=True, right=True)

    # Set x-axis label only on bottom subplot
    axs_err[nrows_layout - 1, 0].set_xlabel("Time (s)")

    fig_err.tight_layout(rect=[0, 0.02, 1, 0.98], h_pad=1.2)

    # Save errors figure
    out_path_err = os.path.join(out_dir, f"{system}_errors.pdf")
    fig_err.savefig(out_path_err)
    print(f"✓ Saved errors to: {out_path_err}")
    
    return fig_traj, fig_err


def plot_training_history(loss_history, system, out_dir="figures"):
    """
    Plot training loss history.
    
    Parameters:
    -----------
    loss_history : dict
        Dictionary containing loss components over epochs
    system : str
        System name for filename
    out_dir : str, optional
        Output directory
    """
    os.makedirs(out_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    epochs = loss_history['epochs']
    ax.semilogy(epochs, loss_history['total'], 'k-', label='Total Loss', linewidth=2)
    ax.semilogy(epochs, loss_history['ic'], 'r--', label='IC Loss', linewidth=1.5)
    ax.semilogy(epochs, loss_history['pde'], 'b--', label='PDE Loss', linewidth=1.5)
    ax.semilogy(epochs, loss_history['measurement'], 'g--', label='Measurement Loss', linewidth=1.5)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    out_path = os.path.join(out_dir, f"{system}_training_history.pdf")
    fig.savefig(out_path)
    print(f"✓ Saved training history to: {out_path}")
    
    return fig