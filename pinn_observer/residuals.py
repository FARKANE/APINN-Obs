"""
Residual functions for physics-informed neural network observer training.

Each residual function enforces the observer dynamics for a specific system.
"""

import torch


def reverse_residual(t, xa, model, device):
    """Residual function for reverse Duffing system."""
    t = t.detach().requires_grad_(True)
    out = model(t)
    xc = out[:, :2]
    L = out[:, 2:].view(-1, 2, 1)
    
    dxcdt = torch.stack([
        torch.autograd.grad(xc[:, i].sum(), t, create_graph=True)[0]
        for i in range(2)
    ], dim=1).squeeze(-1)
    
    x1, x2 = xc[:, 0:1], xc[:, 1:2]
    f_phys = torch.cat([x2**3, -x1], dim=1)
    
    y_true = xa[:, 0:1]
    y_est = xc[:, 0:1]
    innov = (y_true - y_est).unsqueeze(-1)
    corr = (L @ innov).squeeze(-1)
    Bu = torch.zeros_like(f_phys, device=device)
    
    return dxcdt - f_phys - corr - Bu


def satellite_residual(t, xa, model, device):
    """Residual function for satellite system."""
    t = t.detach().requires_grad_(True)
    
    out = model(t)           
    xc = out[:, :3]        
    L_flat = out[:, 3:]         
    L = L_flat.view(-1, 3, 2) 
    
    dxcdt = torch.stack([
        torch.autograd.grad(xc[:, i].sum(), t, create_graph=True)[0]
        for i in range(3)
    ], dim=1).squeeze(-1)  

    x1, x2, x3 = xc[:, 0:1], xc[:, 1:2], xc[:, 2:3]
    a1, a2, a3 = 1.0, -1.0, 1.0
    f_phys = torch.cat([
        a1 * x2 * x3,
        a2 * x1 * x3,
        a3 * x1 * x2
    ], dim=1)
    
    y_true = xa[:, [0, 2]]           
    y_est = torch.cat([x1, x3], 1)  
    innov = (y_true - y_est).unsqueeze(-1)
    corr = (L @ innov).squeeze(-1)
    Bu = torch.zeros_like(f_phys, device=device)
    
    return dxcdt - f_phys - corr - Bu


def harmonic_residual(t, xa, model, device):
    """Residual function for harmonic system."""
    t = t.detach().requires_grad_(True)
    out = model(t)
    xc = out[:, :3]
    L = out[:, 3:].view(-1, 3, 1)
    
    dxcdt = torch.stack([
        torch.autograd.grad(xc[:, i].sum(), t, create_graph=True)[0]
        for i in range(3)
    ], dim=1).squeeze(-1)
    
    a1, a2 = 1.0, -1.0
    x1, x2, x3 = xc[:, 0:1], xc[:, 1:2], xc[:, 2:3]
    f_vec = torch.cat([a1*x2, a2*x1*x3, torch.zeros_like(x3)], 1)
    
    y_t, y_h = xa[:, 0:1], xc[:, 0:1]
    corr = (L * (y_t - y_h).unsqueeze(-1)).squeeze(-1)
    Bu = torch.zeros_like(f_vec, device=device)
    
    return dxcdt - f_vec - corr - Bu


def induction_motor_residual(t, xa, model, device):
    """Residual function for induction motor system."""
    # System parameters
    Lr, Rr, M, Ls, Rs, J, p = 0.0699, 0.15, 0.068, 0.0699, 0.18, 0.0586, 1.0
    Tr = Lr/Rr
    sigma = 1 - (M**2)/(Ls*Lr)
    K = M/(sigma*Ls*Lr)
    gamma = Rs/(sigma*Ls) + Rr*M**2/(sigma*Ls*Lr**2)

    t = t.detach().requires_grad_(True)
    out = model(t)
    xc = out[:, :5]             
    L = out[:, 5:].view(-1, 5, 2)
    
    dxcdt = torch.stack([
        torch.autograd.grad(xc[:, i].sum(), t, create_graph=True)[0]
        for i in range(5)
    ], dim=1).squeeze(-1)

    x1, x2, x3, x4, x5 = [xc[:, i:i+1] for i in range(5)]
    f_vec = torch.cat([
        -gamma*x1 + (K/Tr)*x3 + K*p*x5*x4,
        -gamma*x2 - K*p*x5*x3 + (K/Tr)*x4,
        (M/Tr)*x1 - (1/Tr)*x3 - p*x5*x4,
        (M/Tr)*x2 + p*x5*x3 - (1/Tr)*x4,
        (p*M)/(J*Lr)*(x3*x2 - x4*x1)
    ], dim=1)

    y_t = xa[:, :2]
    y_h = xc[:, :2]
    innov = (y_t - y_h).unsqueeze(-1)
    corr = (L @ innov).squeeze(-1)
    Bu = torch.zeros_like(f_vec, device=device)

    return dxcdt - f_vec - corr - Bu


def academic_residual(t, xa, model, device):
    """Residual function for academic system."""
    t = t.detach().requires_grad_(True)
    out = model(t)
    xc = out[:, :2]
    L = out[:, 2:].view(-1, 2, 1)
    
    dxcdt = torch.stack([
        torch.autograd.grad(xc[:, i].sum(), t, create_graph=True)[0]
        for i in range(2)
    ], dim=1).squeeze(-1)
    
    a1, a2 = 1.0, -1.0
    x1, x2 = xc[:, 0:1], xc[:, 1:2]
    f_vec = torch.cat([
        a1*x2*torch.sqrt(1+x1**2),
        a2*(x1/torch.sqrt(1+x1**2))*x2**2
    ], 1)
    
    y_t, y_h = xa[:, 0:1], xc[:, 0:1]
    corr = (L * (y_t - y_h).unsqueeze(-1)).squeeze(-1)
    Bu = torch.zeros_like(f_vec, device=device)
    
    return dxcdt - f_vec - corr - Bu


def modified_academic_residual(t, xa, model, device):
    """Residual function for modified academic system."""
    t = t.detach().requires_grad_(True)
    out = model(t)
    xc = out[:, :2]
    L = out[:, 2:].view(-1, 2, 1)
    
    dxcdt = torch.stack([
        torch.autograd.grad(xc[:, i].sum(), t, create_graph=True)[0]
        for i in range(2)
    ], dim=1).squeeze(-1)
    
    a1, a2 = 1.0, -1.0
    x1, x2 = xc[:, 0:1], xc[:, 1:2]
    f_vec = torch.cat([
        a1*x2*torch.sqrt(1+x2**2),
        a2*(x1/torch.sqrt(1+x2**2))*x2**2
    ], 1)
    
    y_t, y_h = xa[:, 0:1], xc[:, 0:1]
    corr = (L * (y_t - y_h).unsqueeze(-1)).squeeze(-1)
    Bu = torch.zeros_like(f_vec, device=device)
    
    return dxcdt - f_vec - corr - Bu


# Mapping of system names to residual functions
RESIDUAL_MAP = {
    "reverse":           reverse_residual,
    "satellite":         satellite_residual,
    "harmonic":          harmonic_residual,
    "induction_motor":   induction_motor_residual,
    "academic":          academic_residual,
    "modified_academic": modified_academic_residual,
}

# Measurement indices for each system
MEAS_IDX = {
    "reverse":           [0],      
    "satellite":         [0, 2],   
    "harmonic":          [0],      
    "induction_motor":   [0, 1],   
    "academic":          [0],     
    "modified_academic": [0],     
}