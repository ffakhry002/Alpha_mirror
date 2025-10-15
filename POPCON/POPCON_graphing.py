"""POPCON_graphing.py - Create POPCON plot and analyze design points"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import from our modular files
from equations import (
    calculate_loss_coefficient,
    calculate_beta_local,
    calculate_B0_with_diamagnetic,
    calculate_beta_limit,
    calculate_a0_absorption,
    calculate_a0_FLR,
    calculate_plasma_geometry,
    calculate_fusion_power,
    calculate_NBI_power,
    calculate_NWL,
    calculate_Q,
    test_loss_coefficient,
    C_90deg_interp,
    ratio_45_90_interp
)

from POPCON_inputs import (
    B_max_default,
    R_M_default,
    beta_c_default,
    T_i_coeff,
    T_e_coeff,
    N_25,
    N_rho,
    E_b_min,
    E_b_max,
    n_grid_points,
    Q_levels,
    NWL_background,
    NWL_levels,
    a0_levels,
    P_fus_levels,
    P_NBI_levels,
    B_0_levels,
    beta_levels,
    C_levels,
    test_point_E_b100,
    test_point_n_20,
    test_points_list,
    figures_dir,
    figure_dpi,
    figure_size,
    get_size_limit_label,
    calculate_effective_a0_limit,
    get_temperature_info_string
)


def create_full_popcon(B_max=B_max_default, R_M=R_M_default, beta_c=beta_c_default):
    """Create full POPCON plot with beam-target fusion physics"""
    fig, ax = plt.subplots(figsize=figure_size)

    # Calculate derived parameters
    B_0 = B_max / R_M

    # Create grid using input E_b range
    E_b100 = np.linspace(E_b_min, E_b_max, n_grid_points)
    n_20_max = calculate_beta_limit(E_b_min, B_max, R_M, beta_c)
    n_20 = np.linspace(0.01, n_20_max, n_grid_points)

    E_b100_grid, n_20_grid = np.meshgrid(E_b100, n_20)

    # Calculate constraints with diamagnetic effects
    n_20_beta_limit = calculate_beta_limit(E_b100_grid, B_max, R_M, beta_c)

    # Calculate local beta and on-axis field
    beta_local = calculate_beta_local(n_20_grid, E_b100_grid, B_max, R_M)
    B_0_grid = calculate_B0_with_diamagnetic(B_max, R_M, beta_local)

    # Calculate geometry constraints
    a_0_abs = calculate_a0_absorption(E_b100_grid, n_20_grid)
    a_0_FLR = calculate_a0_FLR(E_b100_grid, B_0_grid, N_25)
    a_0_min = np.maximum(a_0_abs, a_0_FLR)

    # Calculate plasma geometry
    L_plasma, V_plasma, vessel_surface_area = calculate_plasma_geometry(a_0_min, N_rho)

    # Calculate loss coefficient
    C_loss = calculate_loss_coefficient(E_b100_grid, R_M)

    # Calculate required NBI power
    P_NBI_required = calculate_NBI_power(n_20_grid, V_plasma, E_b100_grid, R_M, C_loss)

    # Calculate beam-target fusion for full grid
    print(f"Calculating beam-target physics for {n_grid_points}×{n_grid_points} grid points...")

    P_fusion_beam_target = np.zeros_like(E_b100_grid)
    Q_beam_target = np.zeros_like(E_b100_grid)

    # Calculate for each grid point
    for i in range(n_grid_points):
        for j in range(n_grid_points):
            E_b_100_point = E_b100_grid[i, j]
            n_20_point = n_20_grid[i, j]
            E_NBI_keV = E_b_100_point * 100

            try:
                # Calculate temperatures
                T_i = T_i_coeff * E_NBI_keV
                V = V_plasma[i, j]

                # Calculate fusion power
                P_fusion = calculate_fusion_power(E_b_100_point, n_20_point, V, T_i)

                # Calculate Q
                if P_NBI_required[i, j] > 0:
                    Q = calculate_Q(P_fusion, P_NBI_required[i, j])
                else:
                    Q = 0

                P_fusion_beam_target[i, j] = P_fusion
                Q_beam_target[i, j] = Q

            except Exception as e:
                P_fusion_beam_target[i, j] = 0
                Q_beam_target[i, j] = 0

    # Calculate NWL
    NWL_beam_target = calculate_NWL(P_fusion_beam_target, vessel_surface_area)

    # Calculate effective a0 limit
    a0_eff_limit = calculate_effective_a0_limit(N_rho)

    # Create masks for different regions
    mask_beta = n_20_grid > n_20_beta_limit
    mask_impractical = a_0_min > a0_eff_limit
    mask_low_NWL = NWL_beam_target < 0.0

    mask_gray = mask_beta | mask_impractical
    mask_black = np.zeros_like(mask_gray, dtype=bool)
    mask_white = (~mask_gray) & mask_low_NWL

    # Fill regions
    ax.contourf(E_b100_grid, n_20_grid, mask_gray.astype(int),
                levels=[0.5, 1.5], colors=['lightgray'], alpha=0.8)

    # Plot NWL contours
    NWL_valid = NWL_beam_target.copy()
    NWL_valid[mask_gray | mask_black | mask_white] = np.nan

    im = ax.contourf(E_b100_grid, n_20_grid, NWL_valid,
                     levels=NWL_background, cmap='viridis', extend='max')

    # Beta limit line
    ax.plot(E_b100, n_20_beta_limit[0, :], 'purple', linewidth=3, zorder=5,
            label='Beta limit')

    # Size limit boundary line
    size_limit_boundary = a_0_min - a0_eff_limit
    ax.contour(E_b100_grid, n_20_grid, size_limit_boundary,
               levels=[0], colors=['darkred'], linewidths=2, linestyles='--', zorder=4)
    ax.plot([], [], color='darkred', linewidth=2, linestyle='--',
            label=get_size_limit_label())

    # a₀ contours
    a_0_min_valid = a_0_min.copy()
    a_0_min_valid[mask_gray | mask_black | mask_white] = np.nan

    CS = ax.contour(E_b100_grid, n_20_grid, a_0_min_valid,
                    levels=a0_levels, colors='pink', linewidths=1.5, alpha=0.9)

    for level in a0_levels:
        label = f'a₀={level:.2f}m'
        ax.clabel(CS, levels=[level], inline=True, fontsize=7, fmt=label)

    # Q contour lines
    Q_valid = Q_beam_target.copy()
    Q_valid[mask_gray | mask_black | mask_white] = np.nan

    CS_Q = ax.contour(E_b100_grid, n_20_grid, Q_valid,
                      levels=Q_levels, colors='cyan', linewidths=1.5,
                      alpha=0.8, linestyles='-')
    ax.clabel(CS_Q, inline=True, fontsize=8, fmt='Q=%.2f')

    # P_fusion contours
    if len(P_fus_levels) > 0:
        P_fusion_valid = P_fusion_beam_target.copy()
        P_fusion_valid[mask_gray | mask_black | mask_white] = np.nan

        CS_Pfus = ax.contour(E_b100_grid, n_20_grid, P_fusion_valid,
                             levels=P_fus_levels, colors='magenta', linewidths=1.5,
                             alpha=0.8, linestyles='-')
        ax.clabel(CS_Pfus, inline=True, fontsize=8, fmt='P_fus=%.0f MW')

    # NWL contour lines
    CS_NWL = ax.contour(E_b100_grid, n_20_grid, NWL_valid,
                        levels=NWL_levels, colors='white', linewidths=1.0,
                        alpha=0.9, linestyles='-')
    ax.clabel(CS_NWL, inline=True, fontsize=8, fmt='%.1f MW/m²')

    # B₀ contours
    if len(B_0_levels) > 0:
        B_0_valid = B_0_grid.copy()
        B_0_valid[mask_gray | mask_black | mask_white] = np.nan
        CS_B0 = ax.contour(E_b100_grid, n_20_grid, B_0_valid,
                           levels=B_0_levels, colors='orange', linewidths=1.5,
                           alpha=0.7, linestyles='-')
        ax.clabel(CS_B0, inline=True, fontsize=8, fmt='B₀=%.1f T')

    # P_NBI contours
    P_NBI_valid = P_NBI_required.copy()
    P_NBI_valid[mask_gray | mask_black | mask_white] = np.nan
    CS_PNBI = ax.contour(E_b100_grid, n_20_grid, P_NBI_valid,
                         levels=P_NBI_levels, colors='red', linewidths=2.5,
                         alpha=1.0, linestyles='-')
    ax.clabel(CS_PNBI, inline=True, fontsize=8, fmt='P_NBI=%.0f MW')

    # Beta contours
    if len(beta_levels) > 0:
        beta_valid = beta_local.copy()
        beta_valid[mask_gray | mask_black | mask_white] = np.nan
        CS_beta = ax.contour(E_b100_grid, n_20_grid, beta_valid,
                             levels=beta_levels, colors='orange', linewidths=1.5,
                             alpha=0.8, linestyles='-.')
        ax.clabel(CS_beta, inline=True, fontsize=8, fmt='β=%.2f')

    # C (Loss Coefficient) contours
    if len(C_levels) > 0:
        C_valid = C_loss.copy()
        C_valid[mask_gray | mask_black | mask_white] = np.nan
        CS_C = ax.contour(E_b100_grid, n_20_grid, C_valid,
                          levels=C_levels, colors='brown', linewidths=1.5,
                          alpha=0.8, linestyles=':')
        ax.clabel(CS_C, inline=True, fontsize=8, fmt='C=%.2f s')

    # Formatting
    ax.set_xlabel(r'$E_{NBI}$ [100 keV]', fontsize=14)
    ax.set_ylabel(r'$\langle n_{20} \rangle$ [$10^{20}$ m$^{-3}$]', fontsize=14)
    ax.set_xlim([E_b_min, E_b_max])
    ax.set_ylim([0, n_20_max])

    # Force linear tick formatting
    ax.ticklabel_format(style='plain', axis='x')
    ax.ticklabel_format(style='plain', axis='y')
    x_ticks = np.arange(E_b_min, E_b_max + 0.1, 0.2)
    ax.set_xticks(x_ticks)

    # Legend
    ax.legend(loc='lower left', fontsize=10)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r'NWL [MW/m²] (Beam-Target Fusion)', fontsize=12)
    cbar.set_ticks(NWL_levels)
    cbar.ax.tick_params(labelsize=10)

    # Grid
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)

    # Title
    temp_info = get_temperature_info_string()
    size_info = get_size_limit_label()

    ax.set_title(f'($B_{{max}}$={B_max}T, $R_M$={int(R_M)}, '
                 f'$B_0$={B_0:.1f}T, $\\beta_c$={beta_c}, {size_info})\n'
                 f'{temp_info}',
                 fontsize=14, weight='bold')

    plt.tight_layout()

    return fig


def analyze_design_point(target_E_b100=test_point_E_b100, target_n_20=test_point_n_20,
                         B_max=B_max_default, R_M=R_M_default, beta_c=beta_c_default):
    """Analyze specific (E_b, n_20) design point"""
    # Create grid around target point
    E_b100 = np.linspace(E_b_min, E_b_max, n_grid_points)
    n_20_max = calculate_beta_limit(E_b_min, B_max, R_M, beta_c)
    n_20 = np.linspace(0.01, n_20_max, n_grid_points)

    # Find closest grid indices
    E_idx = np.argmin(np.abs(E_b100 - target_E_b100))
    n_idx = np.argmin(np.abs(n_20 - target_n_20))

    # Get actual values
    E_b100_actual = E_b100[E_idx]
    n_20_actual = n_20[n_idx]
    E_NBI_keV = E_b100_actual * 100

    # Calculate all parameters at this point
    beta_local = calculate_beta_local(n_20_actual, E_b100_actual, B_max, R_M)
    B_0 = calculate_B0_with_diamagnetic(B_max, R_M, beta_local)

    a_0_abs = calculate_a0_absorption(E_b100_actual, n_20_actual)
    a_0_FLR = calculate_a0_FLR(E_b100_actual, B_0, N_25)
    a_0_min = max(a_0_abs, a_0_FLR)

    L_plasma, V_plasma, vessel_surface_area = calculate_plasma_geometry(a_0_min, N_rho)

    C_loss = calculate_loss_coefficient(E_b100_actual, R_M)

    T_i = T_i_coeff * E_NBI_keV
    P_fusion = calculate_fusion_power(E_b100_actual, n_20_actual, V_plasma, T_i)
    P_NBI = calculate_NBI_power(n_20_actual, V_plasma, E_b100_actual, R_M, C_loss)
    NWL = calculate_NWL(P_fusion, vessel_surface_area)
    Q = calculate_Q(P_fusion, P_NBI)

    # Print analysis
    print("\n" + "="*70)
    print(f"DESIGN POINT ANALYSIS: R_M={R_M}, E_b={E_NBI_keV:.0f} keV, n_20={target_n_20}")
    print("="*70)
    print(f"Actual grid point: E_b={E_NBI_keV:.1f} keV, n_20={n_20_actual:.4f}")
    print(f"\nGeometry:")
    print(f"  beta_local = {beta_local:.6f}")
    print(f"  B_0 = {B_0:.4f} T")
    print(f"  a_0_abs = {a_0_abs:.4f} m")
    print(f"  a_0_FLR = {a_0_FLR:.4f} m")
    print(f"  a_0_min = {a_0_min:.4f} m (limiting: {'Absorption' if a_0_abs > a_0_FLR else 'FLR'})")
    print(f"  L = {L_plasma:.4f} m")
    print(f"  V = {V_plasma:.4f} m³")
    print(f"  Surface area = {vessel_surface_area:.4f} m²")
    print(f"\nPhysics:")
    print(f"  C_loss = {C_loss:.6f} s")
    print(f"  T_i = {T_i:.2f} keV")
    print(f"  P_fusion = {P_fusion:.2f} MW")
    print(f"  P_NBI = {P_NBI:.2f} MW")
    print(f"  NWL = {NWL:.4f} MW/m²")
    print(f"  Q = {Q:.4f}")
    print("="*70 + "\n")


def test_multiple_points(test_points=test_points_list, B_max=B_max_default,
                        R_M=R_M_default, beta_c=beta_c_default):
    """Test multiple design points and print summary table"""
    if not test_points:
        print("\nNo test points defined in test_points_list")
        return

    print("\n" + "="*100)
    print(f"TESTING MULTIPLE DESIGN POINTS: R_M={R_M}, B_max={B_max}T")
    print("="*100)

    # Header
    print(f"\n{'E_b':>6} {'n_20':>6} {'β':>8} {'B_0':>6} {'a0_abs':>7} {'a0_FLR':>7} "
          f"{'a0_min':>7} {'L':>6} {'V':>7} {'C':>7} {'P_fus':>7} {'P_NBI':>7} "
          f"{'NWL':>6} {'Q':>6} {'Limit':>6}")
    print(f"{'[keV]':>6} {'[e20]':>6} {'':>8} {'[T]':>6} {'[m]':>7} {'[m]':>7} "
          f"{'[m]':>7} {'[m]':>6} {'[m³]':>7} {'[s]':>7} {'[MW]':>7} {'[MW]':>7} "
          f"{'[MW/m²]':>6} {'':>6} {'':>6}")
    print("-"*100)

    for E_b_100, n_20_target in test_points:
        E_NBI_keV = E_b_100 * 100

        # Calculate all parameters
        beta_local = calculate_beta_local(n_20_target, E_b_100, B_max, R_M)
        B_0 = calculate_B0_with_diamagnetic(B_max, R_M, beta_local)

        a_0_abs = calculate_a0_absorption(E_b_100, n_20_target)
        a_0_FLR = calculate_a0_FLR(E_b_100, B_0, N_25)
        a_0_min = max(a_0_abs, a_0_FLR)
        limiting_constraint = "Abs" if a_0_abs > a_0_FLR else "FLR"

        L_plasma, V_plasma, vessel_surface_area = calculate_plasma_geometry(a_0_min, N_rho)

        C_loss = calculate_loss_coefficient(E_b_100, R_M)

        T_i = T_i_coeff * E_NBI_keV
        P_fusion = calculate_fusion_power(E_b_100, n_20_target, V_plasma, T_i)
        P_NBI = calculate_NBI_power(n_20_target, V_plasma, E_b_100, R_M, C_loss)
        NWL = calculate_NWL(P_fusion, vessel_surface_area)
        Q = calculate_Q(P_fusion, P_NBI)

        # Print row
        print(f"{E_NBI_keV:6.0f} {n_20_target:6.2f} {beta_local:8.5f} {B_0:6.3f} "
              f"{a_0_abs:7.4f} {a_0_FLR:7.4f} {a_0_min:7.4f} {L_plasma:6.2f} "
              f"{V_plasma:7.3f} {C_loss:7.4f} {P_fusion:7.2f} {P_NBI:7.2f} "
              f"{NWL:6.3f} {Q:6.3f} {limiting_constraint:>6}")

    print("="*100 + "\n")


if __name__ == "__main__":
    print("Creating Beam-Target Fusion POPCON plots...")

    # Print extracted data status
    if len(C_90deg_interp) > 0:
        print(f"✓ Using extracted graph data for loss coefficient calculation")
        print(f"  Available mirror ratios: {sorted(C_90deg_interp.keys())}")
        print(f"  Available C@45°/C@90° ratios: {sorted(ratio_45_90_interp.keys())}")

        # Test the loss coefficient calculation
        test_loss_coefficient(120.0, R_M_default)
    else:
        print("⚠ Using fallback loss coefficient calculation (no extracted data)")

    # Test multiple design points
    print("\nTesting design points...")
    test_multiple_points()

    # Analyze single design point in detail
    print("\nAnalyzing single design point in detail...")
    analyze_design_point()

    # Create main POPCON
    print("\nCreating main beam-target POPCON...")
    fig_single = create_full_popcon(B_max_default, R_M_default, beta_c_default)

    # Save figure
    output_path = figures_dir / 'POPCON_Full.png'
    fig_single.savefig(output_path, dpi=figure_dpi, bbox_inches='tight')
    print(f"Saved: {output_path}")

    plt.show()
