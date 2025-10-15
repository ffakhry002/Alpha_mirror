"""POPCON_Bmax_Bcond_graphing.py - Create POPCON in B_max vs B_conductor space"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import from equations module
from equations import (
    calculate_loss_coefficient,
    calculate_plasma_geometry,
    calculate_fusion_power,
    calculate_NBI_power,
    calculate_NWL,
    calculate_Q
)

# Import from new inputs file
from POPCON_Bmax_Bcond_inputs import (
    B_max_min,
    B_max_max,
    B_conductor_min,
    B_conductor_max,
    E_b_keV_fixed,
    beta_c_default,
    R_M_min_limit,
    T_i_coeff,
    N_25,
    N_rho,
    r_shield,
    r_baker,
    n_grid_points,
    NWL_background,
    NWL_levels,
    a0_levels,
    B_0_levels,
    R_M_levels,
    Q_levels,
    P_NBI_levels,
    P_fus_levels,
    n_20_levels,
    beta_levels,
    HTS_levels,
    figures_dir,
    figure_dpi,
    figure_size,
    get_title_string,
    print_configuration
)

# Physical constants
MU_0 = 4 * np.pi * 1e-7  # T·m/A


# ============================================================================
# HTS CALCULATION FUNCTIONS
# ============================================================================

def calculate_HTS_solenoid(a_0_min, B_conductor, N_rho_val, r_baker_val):
    """
    Calculate solenoid HTS requirement in A-m.

    Formula: A-m = (2π a₀ N_ρ B_conductor / μ₀) × (1.1a₀ + r_baker)

    Parameters:
        a_0_min: Minor radius [m]
        B_conductor: Central field [T]
        N_rho_val: Normalized length parameter
        r_baker_val: Baker shield radius [m]

    Returns:
        A-m for solenoid [A·m]
    """
    R_solenoid = 1.1 * a_0_min + r_baker_val
    L_solenoid = N_rho_val * a_0_min

    A_m_solenoid = (2 * np.pi * R_solenoid * B_conductor * L_solenoid) / MU_0

    return A_m_solenoid


def calculate_HTS_end_cell(a_0_FLR, B_max, r_shield_val):
    """
    Calculate single end cell (mirror coil) HTS requirement in A-m.

    Formula: A-m = (4π B_max / μ₀) × (1.1a₀,FLR + r_shield)²

    Parameters:
        a_0_FLR: FLR constraint radius [m]
        B_max: Maximum mirror field [T]
        r_shield_val: Shield radius [m]

    Returns:
        A-m for single end cell [A·m]
    """
    R_ring = 1.1 * a_0_FLR + r_shield_val

    A_m_ring = (4 * np.pi * B_max * R_ring**2) / MU_0

    return A_m_ring


def calculate_HTS_total(a_0_min, a_0_FLR, B_conductor, B_max, N_rho_val, r_baker_val, r_shield_val):
    """
    Calculate total HTS requirement: 1 solenoid + 2 end cells.

    Returns:
        Total A-m [A·m], solenoid A-m [A·m], single end cell A-m [A·m]
    """
    A_m_solenoid = calculate_HTS_solenoid(a_0_min, B_conductor, N_rho_val, r_baker_val)
    A_m_end_cell = calculate_HTS_end_cell(a_0_FLR, B_max, r_shield_val)
    A_m_total = A_m_solenoid + 2 * A_m_end_cell

    return A_m_total, A_m_solenoid, A_m_end_cell


# ============================================================================
# OPTIMAL CONSTRAINT SOLVER
# ============================================================================

def find_optimal_n20(B_conductor, E_b_100keV, beta_c, max_iterations=100, tolerance=1e-8):
    """
    Find n_20 at optimal constraint point where a_FLR = a_abs.

    At optimal point: n₂₀ = 0.3 B₀ / N₂₅

    Where B₀ = B_conductor / √(1 - β)
    And β = 3n₂₀E_b / (B_conductor² + 3n₂₀E_b)

    Returns:
        n_20, beta, B_0 if successful
        None, None, None if fails (beta limit exceeded)
    """
    B_conductor_squared = B_conductor**2

    # Initial guess
    n_20_guess = 0.5

    for iteration in range(max_iterations):
        # Calculate beta from current n_20
        beta = (3 * n_20_guess * E_b_100keV) / (B_conductor_squared + 3 * n_20_guess * E_b_100keV)

        # Check beta limit
        if beta >= beta_c:
            return None, None, None

        # Calculate B_0 from beta
        B_0 = B_conductor / np.sqrt(1 - beta)

        # Calculate new n_20 from optimal constraint
        n_20_new = 0.3 * B_0 / N_25

        # Check convergence
        if abs(n_20_new - n_20_guess) / (n_20_guess + 1e-12) < tolerance:
            return n_20_new, beta, B_0

        # Update guess
        n_20_guess = n_20_new

    # Failed to converge
    print(f"Warning: Failed to converge for B_conductor={B_conductor:.2f}")
    return None, None, None


# ============================================================================
# MAIN POPCON CREATION
# ============================================================================

def create_Bmax_Bcond_POPCON():
    """Create POPCON in B_max vs B_conductor space at optimal constraint point"""

    print("\n" + "="*80)
    print("Creating B_max vs B_conductor POPCON...")
    print("="*80)

    # Create grid
    B_max_grid_1d = np.linspace(B_max_min, B_max_max, n_grid_points)
    B_conductor_grid_1d = np.linspace(B_conductor_min, B_conductor_max, n_grid_points)
    B_max_grid, B_conductor_grid = np.meshgrid(B_max_grid_1d, B_conductor_grid_1d)

    # Calculate R_M at each point
    R_M_grid = B_max_grid / B_conductor_grid

    # Fixed beam energy
    E_b_100keV = E_b_keV_fixed / 100.0

    # Initialize arrays
    n_20_grid = np.zeros_like(B_max_grid)
    beta_grid = np.zeros_like(B_max_grid)
    B_0_grid = np.zeros_like(B_max_grid)
    a_0_grid = np.zeros_like(B_max_grid)
    a_0_FLR_grid = np.zeros_like(B_max_grid)
    P_NBI_grid = np.zeros_like(B_max_grid)
    P_fus_grid = np.zeros_like(B_max_grid)
    Q_grid = np.zeros_like(B_max_grid)
    NWL_grid = np.zeros_like(B_max_grid)
    HTS_total_grid = np.zeros_like(B_max_grid)
    HTS_solenoid_grid = np.zeros_like(B_max_grid)
    HTS_end_cell_grid = np.zeros_like(B_max_grid)

    # Masks for invalid regions
    mask_valid = np.ones_like(B_max_grid, dtype=bool)

    print(f"Solving for optimal operating point at {n_grid_points}x{n_grid_points} grid points...")
    print("(This may take a moment...)")

    # Progress tracking
    total_points = n_grid_points * n_grid_points
    progress_interval = total_points // 20

    # Solve at each grid point
    for i in range(n_grid_points):
        for j in range(n_grid_points):
            point_num = i * n_grid_points + j
            if point_num % progress_interval == 0:
                progress = 100 * point_num / total_points
                print(f"  Progress: {progress:.0f}%")

            B_max = B_max_grid[i, j]
            B_conductor = B_conductor_grid[i, j]
            R_M = R_M_grid[i, j]

            # Check R_M constraint
            if R_M < R_M_min_limit:
                mask_valid[i, j] = False
                continue

            # Find optimal n_20
            n_20, beta, B_0 = find_optimal_n20(B_conductor, E_b_100keV, beta_c_default)

            if n_20 is None:
                # Beta limit exceeded or failed to converge
                mask_valid[i, j] = False
                continue

            # Store values
            n_20_grid[i, j] = n_20
            beta_grid[i, j] = beta
            B_0_grid[i, j] = B_0

            # Calculate a_0 (at optimal point, a_FLR = a_abs)
            a_0 = 0.3 * np.sqrt(E_b_100keV) / n_20
            a_0_grid[i, j] = a_0

            # Calculate a_0_FLR for HTS calculation
            a_0_FLR = N_25 * np.sqrt(E_b_100keV) / B_0
            a_0_FLR_grid[i, j] = a_0_FLR

            # Calculate HTS requirements
            HTS_total, HTS_solenoid, HTS_end_cell = calculate_HTS_total(
                a_0, a_0_FLR, B_conductor, B_max, N_rho, r_baker, r_shield
            )

            HTS_total_grid[i, j] = HTS_total / 1000.0  # Convert to kA-m
            HTS_solenoid_grid[i, j] = HTS_solenoid / 1000.0  # Convert to kA-m
            HTS_end_cell_grid[i, j] = HTS_end_cell / 1000.0  # Convert to kA-m

            # Calculate plasma geometry
            L_plasma, V_plasma, vessel_surface_area = calculate_plasma_geometry(a_0, N_rho)

            # Calculate powers
            try:
                C_loss = calculate_loss_coefficient(E_b_100keV, R_M)

                T_i = T_i_coeff * E_b_keV_fixed
                P_fus = calculate_fusion_power(E_b_100keV, n_20, V_plasma, T_i)
                P_NBI = calculate_NBI_power(n_20, V_plasma, E_b_100keV, R_M, C_loss)
                NWL = calculate_NWL(P_fus, vessel_surface_area)
                Q = calculate_Q(P_fus, P_NBI)

                P_fus_grid[i, j] = P_fus
                P_NBI_grid[i, j] = P_NBI
                NWL_grid[i, j] = NWL
                Q_grid[i, j] = Q

            except Exception as e:
                print(f"  Error at B_max={B_max:.1f}, B_cond={B_conductor:.1f}: {e}")
                mask_valid[i, j] = False

    print("  Progress: 100%")
    print("Calculation complete!\n")

    # Create a full beta grid for ALL points (including invalid ones) for beta_c contour
    # This allows us to draw the beta = beta_c contour even where it's operationally invalid
    print("Calculating beta = beta_c contour data...")
    beta_grid_full = np.zeros_like(B_max_grid)
    for i in range(n_grid_points):
        for j in range(n_grid_points):
            B_conductor = B_conductor_grid[i, j]
            R_M = R_M_grid[i, j]

            # Skip if R_M is too low
            if R_M < R_M_min_limit:
                beta_grid_full[i, j] = np.nan
                continue

            # Solve for n_20 and beta at optimal constraint, ignoring beta_c limit
            # Use same solver but with very high beta_c so it doesn't stop
            result = find_optimal_n20(B_conductor, E_b_100keV, beta_c=0.99, max_iterations=100)
            n_20_temp, beta_temp, B_0_temp = result

            if beta_temp is not None:
                beta_grid_full[i, j] = beta_temp
            else:
                beta_grid_full[i, j] = np.nan
    print(f"Beta contour calculation complete. Max beta: {np.nanmax(beta_grid_full):.4f}\n")

    # Set invalid regions to NaN for plotting (except beta_grid_full which we keep)
    n_20_grid[~mask_valid] = np.nan
    beta_grid[~mask_valid] = np.nan
    B_0_grid[~mask_valid] = np.nan
    a_0_grid[~mask_valid] = np.nan
    a_0_FLR_grid[~mask_valid] = np.nan
    P_NBI_grid[~mask_valid] = np.nan
    P_fus_grid[~mask_valid] = np.nan
    Q_grid[~mask_valid] = np.nan
    NWL_grid[~mask_valid] = np.nan
    HTS_total_grid[~mask_valid] = np.nan
    HTS_solenoid_grid[~mask_valid] = np.nan
    HTS_end_cell_grid[~mask_valid] = np.nan

    # ========================================================================
    # CREATE PLOT
    # ========================================================================

    fig, ax = plt.subplots(figsize=figure_size)

    # Shade invalid regions (R_M < 4 or beta > beta_c)
    invalid_mask_for_plot = ~mask_valid
    ax.contourf(B_max_grid, B_conductor_grid, invalid_mask_for_plot.astype(int),
                levels=[0.5, 1.5], colors=['lightgray'], alpha=0.8, zorder=1)

    # Plot NWL heatmap
    im = ax.contourf(B_max_grid, B_conductor_grid, NWL_grid,
                     levels=NWL_background, cmap='viridis', extend='both', zorder=2)

    # ========================================================================
    # BOUNDARY LINES
    # ========================================================================

    # R_M = 4 boundary line
    B_conductor_Rm4 = B_max_grid_1d / R_M_min_limit
    ax.plot(B_max_grid_1d, B_conductor_Rm4, 'r--', linewidth=2.5,
            label=f'$R_M = {R_M_min_limit:.0f}$ limit', zorder=10)

    # Beta limit boundary - Draw beta = beta_c contour
    # Use beta_grid_full which includes calculated beta for ALL points
    try:
        # Create a contour specifically for beta = beta_c using full grid
        CS_beta_limit = ax.contour(B_max_grid, B_conductor_grid, beta_grid_full,
                                   levels=[beta_c_default], colors='darkred',
                                   linewidths=3.5, linestyles='-', zorder=10)

        # Add to legend manually since contour doesn't go through ax.plot
        from matplotlib.lines import Line2D
        beta_limit_line = Line2D([0], [0], color='darkred', linewidth=3.5,
                                linestyle='-', label=f'$\\beta = \\beta_c = {beta_c_default}$ limit')

        # Get beta values along the contour for diagnostics
        paths = CS_beta_limit.get_paths() if hasattr(CS_beta_limit, 'get_paths') else CS_beta_limit.allsegs[0]
        if len(paths) > 0:
            if hasattr(paths[0], 'vertices'):
                vertices = paths[0].vertices
            else:
                vertices = paths[0]
            B_max_beta = vertices[:, 0]
            B_cond_beta = vertices[:, 1]

            print(f"\nBeta = {beta_c_default} contour found:")
            print(f"  Number of points on contour: {len(B_cond_beta)}")
            print(f"  B_conductor range: [{np.min(B_cond_beta):.3f}, {np.max(B_cond_beta):.3f}] T")
            print(f"  Average B_conductor: {np.mean(B_cond_beta):.3f} ± {np.std(B_cond_beta):.3f} T")
            print(f"  (Should be approximately horizontal if optimal constraint is correct)")
        else:
            print(f"\nWarning: Beta = {beta_c_default} contour not found")
            print(f"  Max beta in full grid: {np.nanmax(beta_grid_full):.4f}")
            print(f"  Max beta in valid region: {np.nanmax(beta_grid[mask_valid]):.4f}")
            beta_limit_line = None

    except Exception as e:
        print(f"\nWarning: Could not draw beta limit contour: {e}")
        print(f"  Max beta in full grid: {np.nanmax(beta_grid_full):.4f}")
        print(f"  Max beta in valid region: {np.nanmax(beta_grid[mask_valid]):.4f}")
        beta_limit_line = None

    # ========================================================================
    # CONTOURS
    # ========================================================================

    # NWL contours (white lines)
    CS_NWL = ax.contour(B_max_grid, B_conductor_grid, NWL_grid,
                        levels=NWL_levels, colors='white', linewidths=1.5,
                        alpha=0.9, linestyles='-', zorder=5)
    ax.clabel(CS_NWL, inline=True, fontsize=9, fmt='%.2f MW/m²')

    # a_0 contours (pink)
    CS_a0 = ax.contour(B_max_grid, B_conductor_grid, a_0_grid,
                       levels=a0_levels, colors='pink', linewidths=1.5,
                       alpha=0.8, linestyles='-', zorder=4)
    ax.clabel(CS_a0, inline=True, fontsize=8, fmt='a₀=%.2fm')

    # B_0 contours (orange)
    CS_B0 = ax.contour(B_max_grid, B_conductor_grid, B_0_grid,
                       levels=B_0_levels, colors='orange', linewidths=1.5,
                       alpha=0.7, linestyles='-', zorder=4)
    ax.clabel(CS_B0, inline=True, fontsize=8, fmt='B₀=%.1fT')

    # R_M contours (cyan, will be diagonal)
    CS_RM = ax.contour(B_max_grid, B_conductor_grid, R_M_grid,
                       levels=R_M_levels, colors='cyan', linewidths=2,
                       alpha=0.8, linestyles='--', zorder=4)
    ax.clabel(CS_RM, inline=True, fontsize=9, fmt='Rₘ=%.0f')

    # Q contours (lime green)
    CS_Q = ax.contour(B_max_grid, B_conductor_grid, Q_grid,
                      levels=Q_levels, colors='lime', linewidths=1.5,
                      alpha=0.8, linestyles='-', zorder=4)
    ax.clabel(CS_Q, inline=True, fontsize=8, fmt='Q=%.2f')

    # P_NBI contours (red)
    CS_PNBI = ax.contour(B_max_grid, B_conductor_grid, P_NBI_grid,
                         levels=P_NBI_levels, colors='red', linewidths=2,
                         alpha=0.9, linestyles='-', zorder=4)
    ax.clabel(CS_PNBI, inline=True, fontsize=9, fmt='P_NBI=%.0fMW')

    # P_fus contours (magenta)
    if len(P_fus_levels) > 0:
        CS_Pfus = ax.contour(B_max_grid, B_conductor_grid, P_fus_grid,
                             levels=P_fus_levels, colors='magenta', linewidths=1.5,
                             alpha=0.8, linestyles='-', zorder=4)
        ax.clabel(CS_Pfus, inline=True, fontsize=8, fmt='P_fus=%.0fMW')

    # n_20 contours (yellow)
    CS_n20 = ax.contour(B_max_grid, B_conductor_grid, n_20_grid,
                        levels=n_20_levels, colors='yellow', linewidths=1.5,
                        alpha=0.8, linestyles=':', zorder=4)
    ax.clabel(CS_n20, inline=True, fontsize=8, fmt='n₂₀=%.1f')

    # Beta contours (brown) - optional, only if levels defined
    if len(beta_levels) > 0:
        CS_beta = ax.contour(B_max_grid, B_conductor_grid, beta_grid,
                             levels=beta_levels, colors='brown', linewidths=1.5,
                             alpha=0.7, linestyles='-.', zorder=4)
        ax.clabel(CS_beta, inline=True, fontsize=8, fmt='β=%.2f')

    # HTS total contours (purple) - NEW
    # HTS total contours (purple) - NEW
    if len(HTS_levels) > 0:
        CS_HTS = ax.contour(B_max_grid, B_conductor_grid, HTS_total_grid,
                        levels=HTS_levels, colors='purple', linewidths=2.5,
                        alpha=0.9, linestyles='-', zorder=6)
        ax.clabel(CS_HTS, inline=True, fontsize=9, fmt='HTS=%.0f kA-m')

    # ========================================================================
    # FORMATTING
    # ========================================================================

    ax.set_xlabel('$B_{max}$ [T]', fontsize=16, fontweight='bold')
    ax.set_ylabel('$B_{central,conductor}$ [T]', fontsize=16, fontweight='bold')

    ax.set_xlim([B_max_min, B_max_max])
    ax.set_ylim([B_conductor_min, B_conductor_max])

    # Legend - add beta limit line if it exists
    handles, labels = ax.get_legend_handles_labels()
    if 'beta_limit_line' in locals() and beta_limit_line is not None:
        handles.append(beta_limit_line)
        labels.append(beta_limit_line.get_label())
    ax.legend(handles=handles, labels=labels, loc='upper left', fontsize=11, framealpha=0.95)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('NWL [MW/m²]', fontsize=14, fontweight='bold')
    cbar.set_ticks(NWL_levels)
    cbar.ax.tick_params(labelsize=10)

    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, zorder=3)

    # Title
    title_str = get_title_string()
    ax.set_title(title_str, fontsize=13, weight='bold', pad=15)

    plt.tight_layout()

    return fig, {
        'B_max_grid': B_max_grid,
        'B_conductor_grid': B_conductor_grid,
        'R_M_grid': R_M_grid,
        'n_20_grid': n_20_grid,
        'beta_grid': beta_grid,
        'B_0_grid': B_0_grid,
        'a_0_grid': a_0_grid,
        'a_0_FLR_grid': a_0_FLR_grid,
        'P_NBI_grid': P_NBI_grid,
        'P_fus_grid': P_fus_grid,
        'Q_grid': Q_grid,
        'NWL_grid': NWL_grid,
        'HTS_total_grid': HTS_total_grid,
        'HTS_solenoid_grid': HTS_solenoid_grid,
        'HTS_end_cell_grid': HTS_end_cell_grid,
        'mask_valid': mask_valid
    }


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_specific_point(B_max, B_conductor):
    """Analyze specific (B_max, B_conductor) design point"""
    print("\n" + "="*80)
    print(f"DESIGN POINT ANALYSIS: B_max={B_max}T, B_conductor={B_conductor}T")
    print("="*80)

    E_b_100keV = E_b_keV_fixed / 100.0
    R_M = B_max / B_conductor

    print(f"\nDerived parameters:")
    print(f"  R_M = B_max / B_conductor = {R_M:.3f}")
    print(f"  E_b = {E_b_keV_fixed:.0f} keV")

    if R_M < R_M_min_limit:
        print(f"\n*** INVALID: R_M < {R_M_min_limit} ***")
        return

    # Find optimal n_20
    n_20, beta, B_0 = find_optimal_n20(B_conductor, E_b_100keV, beta_c_default)

    if n_20 is None:
        print("\n*** INVALID: Beta limit exceeded ***")
        return

    print(f"\nOptimal operating point (a_FLR = a_abs):")
    print(f"  n_20 = {n_20:.4f} × 10²⁰ m⁻³")
    print(f"  beta = {beta:.6f}")
    print(f"  B_0 = {B_0:.4f} T")
    print(f"  Beta enhancement: {(B_0/B_conductor - 1)*100:.1f}%")

    # Calculate a_0
    a_0 = 0.3 * np.sqrt(E_b_100keV) / n_20
    a_0_FLR = N_25 * np.sqrt(E_b_100keV) / B_0

    print(f"\nGeometry:")
    print(f"  a_0 = {a_0:.4f} m")
    print(f"  a_0,FLR = {a_0_FLR:.4f} m")

    # Verify constraints
    a_0_abs = 0.3 * np.sqrt(E_b_100keV) / n_20
    print(f"  Verification: a_0,abs = {a_0_abs:.4f} m, a_0,FLR = {a_0_FLR:.4f} m")
    print(f"  Difference: {abs(a_0_abs - a_0_FLR)/a_0*100:.2e}%")

    # Calculate plasma geometry
    L_plasma, V_plasma, vessel_surface_area = calculate_plasma_geometry(a_0, N_rho)
    print(f"  L = {L_plasma:.4f} m")
    print(f"  V = {V_plasma:.4f} m³")
    print(f"  Surface area = {vessel_surface_area:.4f} m²")

    # Calculate HTS requirements
    print(f"\nHTS Requirements:")
    HTS_total, HTS_solenoid, HTS_end_cell = calculate_HTS_total(
        a_0, a_0_FLR, B_conductor, B_max, N_rho, r_baker, r_shield
    )

    # Solenoid details
    R_solenoid = 1.1 * a_0 + r_baker
    print(f"  Solenoid:")
    print(f"    R_solenoid = 1.1×a₀ + r_baker = {R_solenoid:.4f} m")
    print(f"    L_solenoid = N_ρ×a₀ = {L_plasma:.4f} m")
    print(f"    B_conductor = {B_conductor:.2f} T")
    print(f"    HTS_solenoid = {HTS_solenoid/1000:.1f} kA-m")

    # End cell details
    R_ring = 1.1 * a_0_FLR + r_shield
    print(f"  End cell (single):")
    print(f"    R_ring = 1.1×a₀,FLR + r_shield = {R_ring:.4f} m")
    print(f"    B_max = {B_max:.2f} T")
    print(f"    HTS_end_cell = {HTS_end_cell/1000:.1f} kA-m")

    print(f"  Total (1 solenoid + 2 end cells):")
    print(f"    HTS_total = {HTS_total/1000:.1f} kA-m")
    print(f"    Breakdown: {HTS_solenoid/1000:.1f} + 2×{HTS_end_cell/1000:.1f} = {HTS_total/1000:.1f} kA-m")

    # Calculate powers
    C_loss = calculate_loss_coefficient(E_b_100keV, R_M)
    T_i = T_i_coeff * E_b_keV_fixed
    P_fus = calculate_fusion_power(E_b_100keV, n_20, V_plasma, T_i)
    P_NBI = calculate_NBI_power(n_20, V_plasma, E_b_100keV, R_M, C_loss)
    NWL = calculate_NWL(P_fus, vessel_surface_area)
    Q = calculate_Q(P_fus, P_NBI)

    print(f"\nPhysics:")
    print(f"  C_loss = {C_loss:.6f} s")
    print(f"  T_i = {T_i:.2f} keV")
    print(f"  P_fusion = {P_fus:.2f} MW")
    print(f"  P_NBI = {P_NBI:.2f} MW")
    print(f"  NWL = {NWL:.4f} MW/m²")
    print(f"  Q = {Q:.4f}")
    print("="*80 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Print configuration
    print_configuration()

    # Analyze a specific point
    print("\nAnalyzing example design point...")
    analyze_specific_point(B_max=26.0, B_conductor=4.5)

    # Create POPCON
    print("\nCreating B_max vs B_conductor POPCON...")
    fig, data = create_Bmax_Bcond_POPCON()

    # Save figure
    output_path = figures_dir / 'POPCON_Bmax_Bcond.png'
    fig.savefig(output_path, dpi=figure_dpi, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")

    # Print some statistics
    valid_points = np.sum(data['mask_valid'])
    total_points = data['mask_valid'].size
    print(f"\nStatistics:")
    print(f"  Valid operating points: {valid_points:,} / {total_points:,} ({100*valid_points/total_points:.1f}%)")
    print(f"  NWL range: {np.nanmin(data['NWL_grid']):.3f} - {np.nanmax(data['NWL_grid']):.3f} MW/m²")
    print(f"  Q range: {np.nanmin(data['Q_grid']):.3f} - {np.nanmax(data['Q_grid']):.3f}")
    print(f"  P_NBI range: {np.nanmin(data['P_NBI_grid']):.1f} - {np.nanmax(data['P_NBI_grid']):.1f} MW")
    print(f"  HTS total range: {np.nanmin(data['HTS_total_grid']):.0f} - {np.nanmax(data['HTS_total_grid']):.0f} kA-m")
    print(f"  HTS solenoid range: {np.nanmin(data['HTS_solenoid_grid']):.0f} - {np.nanmax(data['HTS_solenoid_grid']):.0f} kA-m")
    print(f"  HTS end cell range: {np.nanmin(data['HTS_end_cell_grid']):.0f} - {np.nanmax(data['HTS_end_cell_grid']):.0f} kA-m")

    plt.show()

    print("\n" + "="*80)
    print("Done!")
    print("="*80)
    analyze_specific_point(B_max=26.0, B_conductor=3.0)  # Low B_cond → high HTS?
    analyze_specific_point(B_max=26.0, B_conductor=6.0)  # High B_cond → low HTS?
