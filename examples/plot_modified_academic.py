

#!/usr/bin/env python3
"""
Example script for plotting PINN observer results for the modified academic system.

This script demonstrates how to visualize the performance of a trained
PINN observer by comparing predicted vs. true trajectories and errors.
"""

import os
import sys

# Add the parent directory to the path to import pinn_observer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinn_observer import plot_pinn_observer_separate_figures
import matplotlib.pyplot as plt


def main():
    """Generate plots for trained modified academic system PINN observer."""
    
    # Configuration
    system = "modified_academic"
    checkpoint_dir = "checkpoints"
    output_dir = "figures"
    
    # Time configuration
    t0 = 0.0
    t_end = 20.0
    N_eval = 1000  # Number of evaluation points
    
    # Plotting options
    use_log_error = False  # Set to True for logarithmic error scale
    
    print("="*60)
    print("PINN Observer Visualization - Modified Academic System")
    print("="*60)
    print(f"System: {system}")
    print(f"Evaluation Time: [{t0}, {t_end}]")
    print(f"Evaluation Points: {N_eval:,}")
    print(f"Output Directory: {output_dir}")
    print("-"*60)
    
    # Setup checkpoint path
    ckpt_path = os.path.join(checkpoint_dir, f"{system}.pth")
    
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: Checkpoint not found at {ckpt_path}")
        print("Please train the model first using train_modified_academic.py")
        return
    
    print(f"✓ Found checkpoint: {ckpt_path}")
    
    # Generate plots
    try:
        fig_traj, fig_err = plot_pinn_observer_separate_figures(
            system=system,
            ckpt_path=ckpt_path,
            t0=t0,
            t_end=t_end,
            N=N_eval,
            out_dir=output_dir,
            use_log_error=use_log_error
        )
        
        print("="*60)
        print("Plotting Completed Successfully!")
        print(f"Figures saved to: {output_dir}/")
        print(f"  - {system}_trajectories.pdf")
        print(f"  - {system}_errors.pdf")
        print("="*60)
        
        # Optionally display the plots
        print("\nDisplaying plots... (close windows to exit)")
        plt.show()
        
    except Exception as e:
        print(f"❌ Error during plotting: {e}")
        return


if __name__ == "__main__":
    main()