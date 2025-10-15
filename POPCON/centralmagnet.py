"""simple_magnet_scaling.py - Clean magnet scaling verification plots"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import from our modular files
from equations import calculate_loss_coefficient, C_90deg_interp
from POPCON_Bmax_Bcond_inputs import (
    E_b_keV_fixed,
    beta_c_default,
    T_i_coeff,
    N_25,
    N_rho,
    r_baker,
    r_shield,
    figures_dir,
    figure_dpi
)

# Physical constants
mu_0 = 4 * np.pi * 1e-7  # H/m


def calculate_solenoid_Am(a_0, B_central):
    """Calculate solenoid A-m requirement"""
    R_solenoid = 1.1 * a_0 + r_baker
    L_solenoid = N_rho * a_0
    A_m = (2 * np.pi * R_solenoid * B_central * L_solenoid) / mu_0
    return A_m / 1000  # Convert to kA-m


def calculate_ring_Am(a_0_FLR, B_max):
    """Calculate end cell ring A-m requirement"""
    R_ring = 1.1 * a_0_FLR + r_shield
    A_m = (4 * np.pi * B_max * R_ring**2) / mu_0
    return A_m / 1000  # Convert to kA-m


def find_optimal_n20(B_conductor, E_b_100keV, beta_c, max_iterations=100, tolerance=1e-8):
    """Find n_20 at optimal constraint point where a_FLR = a_abs"""
    B_conductor_squared = B_conductor**2
    n_20_guess = 0.5

    for iteration in range(max_iterations):
        beta = (3 * n_20_guess * E_b_100keV) / (B_conductor_squared + 3 * n_20_guess * E_b_100keV)

        if beta >= beta_c:
            return None, None, None

        B_0 = B_conductor / np.sqrt(1 - beta)
        n_20_new = 0.3 * B_0 / N_25

        if abs(n_20_new - n_20_guess) / (n_20_guess + 1e-12) < tolerance:
            return n_20_new, beta, B_0

        n_20_guess = n_20_new

    return None, None, None


def generate_data():
    """Generate data for magnet scaling plots"""
    print("\nGenerating data...")

    B_central_range = np.linspace(3.0, 8.0, 100)
    B_max_values = [24, 26, 28, 30]
    E_b_100keV = E_b_keV_fixed / 100.0

    data = {
        'B_central': B_central_range,
        'B_max_values': B_max_values
    }

    for B_max in B_max_values:
        print(f"  B_max = {B_max} T...")

        A_m_solenoid_array = []
        A_m_ring_array = []
        A_m_total_array = []

        for B_conductor in B_central_range:
            n_20, beta, B_0 = find_optimal_n20(B_conductor, E_b_100keV, beta_c_default)

            if n_20 is None:
                A_m_solenoid_array.append(np.nan)
                A_m_ring_array.append(np.nan)
                A_m_total_array.append(np.nan)
                continue

            a_0 = 0.3 * np.sqrt(E_b_100keV) / n_20
            a_0_FLR = N_25 * np.sqrt(E_b_100keV) / B_0

            A_m_sol = calculate_solenoid_Am(a_0, B_conductor)
            A_m_ring = calculate_ring_Am(a_0_FLR, B_max)
            A_m_total = A_m_sol + 2 * A_m_ring

            A_m_solenoid_array.append(A_m_sol)
            A_m_ring_array.append(A_m_ring)
            A_m_total_array.append(A_m_total)

        key = f'B{B_max}'
        data[key] = {
            'A_m_solenoid': np.array(A_m_solenoid_array),
            'A_m_ring': np.array(A_m_ring_array),
            'A_m_total': np.array(A_m_total_array)
        }

    print("Data generation complete!\n")
    return data


def plot_magnet_scaling(data):
    """Create three-panel plot: Solenoid, End Cell, and Total HTS vs B_central"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    B_central = data['B_central']
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(data['B_max_values'])))

    # Left panel: Solenoid
    for B_max, color in zip(data['B_max_values'], colors):
        key = f'B{B_max}'
        ax1.plot(B_central, data[key]['A_m_solenoid'], linewidth=3,
                color=color, label=f'$B_{{max}}$ = {B_max} T')

    ax1.set_xlabel('$B_{central}$ [T]', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Solenoid HTS [kA-m]', fontsize=14, fontweight='bold')
    ax1.set_title(f'Solenoid ($r_{{baker}}$ = {r_baker} m)',
                  fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    # Middle panel: End Cell
    for B_max, color in zip(data['B_max_values'], colors):
        key = f'B{B_max}'
        ax2.plot(B_central, data[key]['A_m_ring'], linewidth=3,
                color=color, label=f'$B_{{max}}$ = {B_max} T')

    ax2.set_xlabel('$B_{central}$ [T]', fontsize=14, fontweight='bold')
    ax2.set_ylabel('End Cell HTS [kA-m]', fontsize=14, fontweight='bold')
    ax2.set_title(f'End Cell ($r_{{shield}}$ = {r_shield} m)',
                  fontsize=15, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)

    # Right panel: Total
    for B_max, color in zip(data['B_max_values'], colors):
        key = f'B{B_max}'
        ax3.plot(B_central, data[key]['A_m_total'], linewidth=3.5,
                color=color, label=f'$B_{{max}}$ = {B_max} T')

    ax3.set_xlabel('$B_{central}$ [T]', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Total HTS [kA-m]', fontsize=14, fontweight='bold')
    ax3.set_title(f'Total = Solenoid + 2×End Cell',
                  fontsize=15, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """Generate magnet scaling plots"""
    print("\n" + "="*80)
    print("MAGNET SCALING PLOTS")
    print("="*80)

    # Check if data is loaded
    if len(C_90deg_interp) == 0:
        print("\n❌ Error: No extracted data available")
        return

    print(f"✓ Using extracted graph data")
    print(f"Beam energy: {E_b_keV_fixed} keV")
    print(f"r_baker: {r_baker} m")
    print(f"r_shield: {r_shield} m")

    # Generate data
    data = generate_data()

    # Create plot
    print("Creating plots...")
    fig = plot_magnet_scaling(data)

    # Save
    output_path = figures_dir / 'Magnet_Scaling.png'
    fig.savefig(output_path, dpi=figure_dpi, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")

    plt.show()

    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == "__main__":
    main()
