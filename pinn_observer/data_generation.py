

"""
Data generation utilities for PINN observer training.

Contains dynamical system definitions and dataset generation functions.
"""

import numpy as np
from scipy.integrate import odeint


def reverse_duffing(x, t, u=0.0):
    """Reverse Duffing oscillator system."""
    x1, x2 = x
    dx1 = x2**3
    dx2 = -x1
    B = np.array([0.0, 1.0])
    f = np.array([dx1, dx2])
    return B*u + f


def satellite_system(x, t, u=0.0):
    """Satellite attitude dynamics system."""
    a1, a2, a3 = 1.0, -1.0, 1.0
    B = np.array([0.0, 0.0, 1.0])
    x1, x2, x3 = x
    f = np.array([a1*x2*x3, a2*x1*x3, a3*x1*x2])
    return B*u + f


def harmonic_system(x, t, u=0.0):
    """Harmonic oscillator system."""
    a1, a2 = 1.0, -1.0
    B = np.array([0.0, 0.0, 1.0])
    x1, x2, x3 = x
    f = np.array([a1*x2, a2*x1*x3, 0.0])
    return B*u + f


def induction_motor_system(x, t, u=0.0):
    """Induction motor dynamic model."""
    Lr, Rr = 0.0699, 0.15
    M, Ls = 0.068, 0.0699
    Rs, J, p = 0.18, 0.0586, 1.0
    Tr = Lr/Rr
    sigma = 1 - (M**2)/(Ls*Lr)
    K = M/(sigma*Ls*Lr)
    gamma = Rs/(sigma*Ls) + Rr*M**2/(sigma*Ls*Lr**2)
    B = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    x1, x2, x3, x4, x5 = x
    f = np.array([
        -gamma*x1 + (K/Tr)*x3 + K*p*x5*x4,
        -gamma*x2 - K*p*x5*x3 + (K/Tr)*x4,
        (M/Tr)*x1 - (1/Tr)*x3 - p*x5*x4,
        (M/Tr)*x2 + p*x5*x3 - (1/Tr)*x4,
        (p*M)/(J*Lr)*(x3*x2 - x4*x1)
    ])
    return B*u + f


def academic_system(x, t, u=0.0):
    """Academic example system."""
    a1, a2 = 1.0, -1.0
    B = np.array([0.0, 1.0])
    x1, x2 = x
    f = np.array([
        a1 * x2 * np.sqrt(1 + x1**2),
        a2 * (x1/np.sqrt(1 + x1**2)) * x2**2
    ])
    return B*u + f


def modified_academic_system(x, t, u=0.0):
    """Modified academic example system."""
    a1, a2 = 1.0, -1.0
    B = np.array([0.0, 1.0])
    x1, x2 = x
    f = np.array([
        a1 * x2 * np.sqrt(1 + x2**2),
        a2 * (x1/np.sqrt(1 + x2**2)) * x2**2
    ])
    return B*u + f


# System mapping
SYSTEM_MAP = {
    "reverse":           reverse_duffing,
    "satellite":         satellite_system,
    "harmonic":          harmonic_system,
    "induction_motor":   induction_motor_system,
    "academic":          academic_system,
    "modified_academic": modified_academic_system,
}


def generate_dataset(name, t0, t_end, N_colloc, N_bc, seed):
    """
    Generate training dataset for PINN observer.
    
    Parameters:
    -----------
    name : str
        System name (key in SYSTEM_MAP)
    t0 : float
        Initial time
    t_end : float
        Final time
    N_colloc : int
        Number of collocation points
    N_bc : int
        Number of boundary condition points
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    t_colloc : ndarray
        Collocation time points
    x_colloc : ndarray
        True states at collocation points
    y_colloc : ndarray
        Measurements at collocation points
    t_bc : ndarray
        Boundary condition time points
    x_bc0 : ndarray
        Initial conditions for boundary points
    """
    np.random.seed(seed)
    sys_fn = SYSTEM_MAP[name]

    # Initial state per system
    x0_map = {
        "reverse":           np.array([2, -1]),
        "satellite":         np.array([2, -1, 3]),
        "harmonic":          np.array([2, -1, 3]),
        "induction_motor":   np.array([1, 0, 2, 3, 0]),
        "academic":          np.array([2, -1]),
        "modified_academic": np.array([2, -1]),
    }
    x0 = x0_map[name]
    
    # Boundary condition initial states
    xc0_map = {
        "reverse":           np.array([1, 1]),
        "satellite":         np.array([1, 1, 2]),
        "harmonic":          np.array([1, 1, 2]),
        "induction_motor":   np.array([-1, 1, 1, 0, 1]),
        "academic":          np.array([1, 2]),
        "modified_academic": np.array([3, 1]),
    }
    xc0 = xc0_map[name]

    # Generate collocation times & compute true states
    t_colloc = np.sort(np.random.uniform(t0, t_end, size=(N_colloc,)))
    x_colloc = odeint(sys_fn, x0, t_colloc)
    
    # Extract measurements based on system
    if name == "satellite":
        meas_idxs = [0, 2]
        y_colloc = x_colloc[:, meas_idxs]  
    else:
        m_map = {
            "reverse":           1,
            "harmonic":          1,
            "induction_motor":   2,
            "academic":          1,
            "modified_academic": 1,
        }
        m = m_map[name]
        y_colloc = x_colloc[:, :m]         

    # Boundary conditions
    t_bc = np.full((N_bc,), t0)
    x_bc0 = np.tile(xc0, (N_bc, 1))

    return t_colloc, x_colloc, y_colloc, t_bc, x_bc0