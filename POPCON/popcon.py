import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from pathlib import Path

# Create figures directory if it doesn't exist
figures_dir = Path(__file__).parent / 'figures'
figures_dir.mkdir(exist_ok=True)

# ============================================================================
# INPUT PARAMETERS - MODIFY THESE
# ============================================================================

# Fixed physics parameters for single plots and when fixed in grids
B_max_default = 22.0      # Maximum mirror field [T]
R_M_default = 10.0        # Mirror ratio
beta_c_default = 0.3      # MHD stability limit

# Fixed values to use when a parameter is held constant in grid plots
fixed_B_max = 22.0        # Use this when B_max is fixed
fixed_R_M = 10.0          # Use this when R_M is fixed
fixed_beta_c = 0.3        # Use this when beta_c is fixed

N_rho = 25.0             # Number of gyroradii across plasma
N_25 = 1.0               # Normalized parameter (N_rho/25)

# Energy range - THIS APPLIES TO ALL PLOTS
E_b_min = 0.8            # Minimum beam energy [100 keV units]
E_b_max = 1.2            # Maximum beam energy [100 keV units]
P_max = 70               # Maximum NBI power [MW]

# Practical engineering limit
a0_practical_limit = 1.5  # Maximum practical radius [m]

# Plotting parameters
a0_min = 0.4             # Minimum radius for contours [m]
a0_step = 0.1            # Step size for radius contours [m]
a0_max = a0_practical_limit + 0.1  # Maximum radius for contours [m]

# Grid resolution
n_grid_points = 400      # Number of grid points (higher = smoother)
n_grid_points_small = 200  # For grid plots (faster)

# Parameter arrays for grid plots
# Grid 1: R_M vs B_max (beta_c is fixed)
R_M_array_1 = [8, 10, 12]
B_max_array_1 = [20, 22, 25]

# Grid 2: beta_c vs B_max (R_M is fixed)
beta_c_array_2 = [0.2, 0.3, 0.4]
B_max_array_2 = [20, 22, 25]

# Grid 3: R_M vs beta_c (B_max is fixed)
R_M_array_3 = [8, 10, 12]
beta_c_array_3 = [0.2, 0.3, 0.4]

# ============================================================================
# MAIN POPCON FUNCTION
# ============================================================================

def create_single_popcon(ax, B_max, R_M, beta_c, E_b_min, E_b_max,
                         a0_practical_limit, a0_min, a0_step, a0_max,
                         n_grid_points=200, show_legend=False):
    """Generate a single POPCON plot on given axes - ALWAYS with Q/NWL labels"""

    # Calculate derived parameters
    B_0 = B_max / R_M
    L_practical = 25 * a0_practical_limit

    # Create grid
    E_b100 = np.linspace(E_b_min, E_b_max, n_grid_points)
    n_20_max = beta_c * B_0**2 / (3 * E_b_min)
    n_20 = np.linspace(0.01, n_20_max, n_grid_points)

    E_b100_grid, n_20_grid = np.meshgrid(E_b100, n_20)

    # Calculate constraints
    n_20_beta_limit = beta_c * B_0**2 / (3 * E_b100_grid)

    a_0_abs = 0.3 * np.sqrt(E_b100_grid) / n_20_grid
    a_0_FLR = N_25 * E_b100_grid / B_0
    a_0_combined = 0.9 * E_b100_grid**1.5 / (beta_c * B_0**2)
    a_0_min_calc = np.maximum(a_0_abs, np.maximum(a_0_FLR, a_0_combined))

    # Calculate NBI power
    V = 25 * np.pi * a_0_min_calc**3
    P_NBI = 6.4 * n_20_grid**2 * V / (E_b100_grid**0.5 * np.log10(R_M))

    # Masks
    mask_beta = n_20_grid > n_20_beta_limit
    mask_impractical = a_0_min_calc > a0_practical_limit
    mask_invalid = mask_beta | mask_impractical

    # Fill with gray first
    ax.contourf(E_b100_grid, n_20_grid, np.ones_like(E_b100_grid),
                levels=[0, 2], colors=['gray'], alpha=0.9)

    # Plot P_NBI
    P_NBI_valid = P_NBI.copy()
    P_NBI_valid[mask_invalid] = np.nan

    P_levels = np.linspace(0, P_max, P_max+1)
    im = ax.contourf(E_b100_grid, n_20_grid, P_NBI_valid,
                     levels=P_levels, cmap='viridis', extend='neither')

    # Beta limit line
    ax.plot(E_b100, n_20_beta_limit[0,:], 'red', linewidth=2, zorder=5,
            label='β limit' if show_legend else '')

    # Practical floor line
    n_20_floor = 0.3 * np.sqrt(E_b100) / a0_practical_limit
    ax.plot(E_b100, n_20_floor, 'darkred', linewidth=1.5, linestyle='--', zorder=4,
            label=f'a₀={a0_practical_limit:.1f}m limit' if show_legend else '')

    # a₀ contours with labels
    a_0_min_valid = a_0_min_calc.copy()
    a_0_min_valid[mask_invalid] = np.nan

    a0_levels = np.arange(a0_min, a0_max, a0_step)
    CS = ax.contour(E_b100_grid, n_20_grid, a_0_min_valid,
                    levels=a0_levels, colors='black', linewidths=1, alpha=0.8)

    # Label every other contour
    for i, level in enumerate(a0_levels):
        if i % 2 == 0:  # Skip every other
            L_val = 25 * level
            ax.clabel(CS, levels=[level], inline=True, fontsize=6,
                     fmt={level: f'{level:.1f}m,L={L_val:.0f}m'})

    # Q and NWL lines with labels - ALWAYS SHOW LABELS
    n_lines = 10
    E_positions = np.linspace(E_b_min, E_b_max, n_lines)

    for i, E_pos in enumerate(E_positions):
        if i % 2 == 0:  # Q lines
            Q_val = np.sqrt(E_pos) * np.log10(R_M)
            ax.axvline(E_pos, color='cyan', linewidth=0.8, alpha=0.4, linestyle=':')
            # Always add Q label
            ax.text(E_pos, n_20_max*0.02, f'Q={Q_val:.2f}',
                    rotation=90, ha='right', va='bottom', fontsize=6, color='cyan')
        else:  # NWL lines
            NWL_val = (beta_c * B_0**2) / (3 * E_pos * np.log10(R_M))
            ax.axvline(E_pos, color='orange', linewidth=0.8, alpha=0.4, linestyle='--')
            # Always add NWL label
            ax.text(E_pos, n_20_max*0.02, f'NWL={NWL_val:.2f}',
                    rotation=90, ha='right', va='bottom', fontsize=6, color='orange')

    # Formatting
    ax.set_xlim([E_b_min, E_b_max])
    ax.set_ylim([0, n_20_max])
    ax.grid(True, alpha=0.15, linestyle=':', linewidth=0.3)

    # Title
    ax.set_title(f'$B_{{max}}$={B_max:.1f}T, $R_M$={R_M:.0f}, $\\beta_c$={beta_c:.2f}',
                 fontsize=9, weight='bold')

    # Add small legend only if requested
    if show_legend:
        ax.legend(loc='lower left', fontsize=5, framealpha=0.7)

    return im

# ============================================================================
# GRID POPCON PLOTS WITH SHARED COLORBAR
# ============================================================================

def create_popcon_grid(param1_array, param2_array, param1_name, param2_name,
                        fixed_param_name, fixed_param_value):
    """Create n×n grid of POPCON plots with shared colorbar"""

    n = len(param1_array)
    m = len(param2_array)

    # Create figure with space for colorbar
    fig = plt.figure(figsize=(4*m + 1, 3.5*n))
    gs = fig.add_gridspec(n, m+1, width_ratios=[1]*m + [0.05], hspace=0.3, wspace=0.2)

    fig.suptitle(f'POPCON Grid: {param1_name} vs {param2_name} ({fixed_param_name}={fixed_param_value})',
                 fontsize=14, weight='bold', y=0.98)

    axes = []
    for i in range(n):
        row = []
        for j in range(m):
            ax = fig.add_subplot(gs[i, j])
            row.append(ax)
        axes.append(row)
    axes = np.array(axes)

    # Create all subplots
    for i, p1 in enumerate(param1_array):
        for j, p2 in enumerate(param2_array):
            ax = axes[i, j]

            params = {
                'B_max': fixed_param_value if fixed_param_name == 'B_max' else None,
                'R_M': fixed_param_value if fixed_param_name == 'R_M' else None,
                'beta_c': fixed_param_value if fixed_param_name == 'beta_c' else None
            }

            params[param1_name] = p1
            params[param2_name] = p2

            # Only show legend on top-left plot
            show_legend = (i == 0 and j == 0)

            im = create_single_popcon(ax, params['B_max'], params['R_M'], params['beta_c'],
                                    E_b_min, E_b_max, a0_practical_limit,
                                    a0_min, a0_step, a0_max, n_grid_points_small,
                                    show_legend=show_legend)

            # Axis labels
            if i == n-1:
                ax.set_xlabel('$E_{NBI}$ [100 keV]', fontsize=8)
            else:
                ax.set_xticklabels([])

            if j == 0:
                ax.set_ylabel('$n_{20}$', fontsize=8)
            else:
                ax.set_yticklabels([])

            ax.tick_params(labelsize=7)

    # Add shared colorbar
    cbar_ax = fig.add_subplot(gs[:, -1])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label(r'$P_{NBI}$ [MW]', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    return fig

# ============================================================================
# SINGLE FULL-SIZE POPCON PLOT
# ============================================================================

def create_full_popcon(B_max=B_max_default, R_M=R_M_default, beta_c=beta_c_default):
    """Create a full-size detailed POPCON plot"""

    fig, ax = plt.subplots(figsize=(11, 8))

    # Calculate derived parameters
    B_0 = B_max / R_M
    L_practical = 25 * a0_practical_limit

    # Create grid using input E_b range
    E_b100 = np.linspace(E_b_min, E_b_max, n_grid_points)
    n_20_max = beta_c * B_0**2 / (3 * E_b_min)
    n_20 = np.linspace(0.01, n_20_max, n_grid_points)

    E_b100_grid, n_20_grid = np.meshgrid(E_b100, n_20)

    # Calculate constraints
    n_20_beta_limit = beta_c * B_0**2 / (3 * E_b100_grid)

    a_0_abs = 0.3 * np.sqrt(E_b100_grid) / n_20_grid
    a_0_FLR = N_25 * E_b100_grid / B_0
    a_0_combined = 0.9 * E_b100_grid**1.5 / (beta_c * B_0**2)
    a_0_min = np.maximum(a_0_abs, np.maximum(a_0_FLR, a_0_combined))

    # Calculate NBI power
    V = 25 * np.pi * a_0_min**3
    P_NBI = 6.4 * n_20_grid**2 * V / (E_b100_grid**0.5 * np.log10(R_M))

    # Masks
    mask_beta = n_20_grid > n_20_beta_limit
    mask_impractical = a_0_min > a0_practical_limit
    mask_invalid = mask_beta | mask_impractical

    # Fill with gray first
    ax.contourf(E_b100_grid, n_20_grid, np.ones_like(E_b100_grid),
                levels=[0, 2], colors=['gray'], alpha=0.9)

    # Plot P_NBI
    P_NBI_valid = P_NBI.copy()
    P_NBI_valid[mask_invalid] = np.nan

    P_levels = np.linspace(0, P_max, P_max+1)
    im = ax.contourf(E_b100_grid, n_20_grid, P_NBI_valid,
                     levels=P_levels, cmap='viridis', extend='neither')

    # Beta limit line
    ax.plot(E_b100, n_20_beta_limit[0,:], 'red', linewidth=3, zorder=5,
            label='Beta limit')

    # Practical floor line
    n_20_floor = 0.3 * np.sqrt(E_b100) / a0_practical_limit
    ax.plot(E_b100, n_20_floor, 'darkred', linewidth=2, linestyle='--',
            label=f'a₀={a0_practical_limit:.1f}m (L={L_practical:.0f}m) limit', zorder=4)

    # a₀ contours
    a_0_min_valid = a_0_min.copy()
    a_0_min_valid[mask_invalid] = np.nan

    a0_levels = np.arange(a0_min, a0_max, a0_step)
    CS = ax.contour(E_b100_grid, n_20_grid, a_0_min_valid,
                    levels=a0_levels, colors='black', linewidths=1.5, alpha=0.9)

    for i, level in enumerate(a0_levels):
        if i % 2 == 0:
            L_val = 25 * level
            ax.clabel(CS, levels=[level], inline=True, fontsize=8,
                     fmt={level: f'{level:.2f}m (L={L_val:.1f}m)'})

    # Q and NWL lines
    n_lines = 10
    E_positions = np.linspace(E_b_min, E_b_max, n_lines)

    for i, E_pos in enumerate(E_positions):
        if i % 2 == 0:
            Q_val = np.sqrt(E_pos) * np.log10(R_M)
            ax.axvline(E_pos, color='cyan', linewidth=1.5, alpha=0.5, linestyle=':')
            ax.text(E_pos, 0.02, f'Q={Q_val:.2f}',
                    rotation=90, ha='right', va='bottom', fontsize=8, color='cyan')
        else:
            NWL_val = (beta_c * B_0**2) / (3 * E_pos * np.log10(R_M))
            ax.axvline(E_pos, color='orange', linewidth=1.5, alpha=0.5, linestyle='--')
            ax.text(E_pos, 0.02, f'NWL={NWL_val:.2f}',
                    rotation=90, ha='right', va='bottom', fontsize=8, color='orange')

    # Formatting
    ax.set_xlabel(r'$E_{NBI}$ [100 keV]', fontsize=14)
    ax.set_ylabel(r'$\langle n_{20} \rangle$ [$10^{20}$ m$^{-3}$]', fontsize=14)
    ax.set_xlim([E_b_min, E_b_max])
    ax.set_ylim([0, n_20_max])

    # Legend
    ax.legend(loc='lower left', fontsize=10)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r'$P_{NBI}$ [MW]', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # Grid
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)

    # Title
    ax.set_title(f'BEAM POPCON ($B_{{max}}$={B_max}T, $R_M$={int(R_M)}, '
                 f'$B_0$={B_0:.1f}T, $\\beta_c$={beta_c}, $N_\\rho$={int(N_rho)})',
                 fontsize=14, weight='bold')

    plt.tight_layout()
    return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Create single full-size POPCON
    fig_single = create_full_popcon(B_max_default, R_M_default, beta_c_default)
    fig_single.savefig(figures_dir / 'full_popcon.png')

    # Create grid plots using the single fixed parameter values
    fig1 = create_popcon_grid(R_M_array_1, B_max_array_1, 'R_M', 'B_max',
                               'beta_c', fixed_beta_c)
    fig1.savefig(figures_dir / 'grid_R_M_B_max.png')

    fig2 = create_popcon_grid(beta_c_array_2, B_max_array_2, 'beta_c', 'B_max',
                               'R_M', fixed_R_M)
    fig2.savefig(figures_dir / 'grid_beta_c_B_max.png')

    fig3 = create_popcon_grid(R_M_array_3, beta_c_array_3, 'R_M', 'beta_c',
                               'B_max', fixed_B_max)
    fig3.savefig(figures_dir / 'grid_R_M_beta_c.png')

    plt.show()
