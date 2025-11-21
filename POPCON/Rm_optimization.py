"""Rm_optimization.py - Find optimal mirror ratio for maximum fusion power at given PNBI

For a fixed beam energy, this program:
1. Scans through different vacuum mirror ratios (Rm_vac)
2. For each Rm, finds points along a constant PNBI contour in the n20 space (at fixed Eb)
3. Calculates the maximum fusion power achievable among intersection points
4. Handles both upper and lower density branches where PNBI contours intersect

Output: P_fusion vs Rm_vac for specified PNBI levels (e.g., 30MW, 50MW)
"""

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
    calculate_a0_FLR_at_mirror,
    calculate_plasma_geometry_frustum,
    calculate_fusion_power,
    calculate_NBI_power,
    calculate_Q,
    calculate_Bw,
    calculate_heat_flux,
    calculate_voltage_closed_lines,
    calculate_voltage_field_reversal,
)

from n20_Eb_inputs import (
    B_max_default,
    beta_c_default,
    T_i_coeff,
    N_25,
    N_rho,
    min_a0,
    qw_limit,
    figures_dir,
    figure_dpi,
)


def check_feasibility(E_b_100keV, n_20, B_central, B_max):
    """
    Check if a point (E_b, n_20) is feasible given constraints:
    - Beta limit
    - Minimum radius
    - Heat flux limit
    """
    # Beta limit check
    n_20_beta_limit = calculate_beta_limit(E_b_100keV, B_central, beta_c_default)
    if n_20 > n_20_beta_limit:
        return False, "beta_limit"

    # Calculate local beta and diamagnetic field
    beta_local = calculate_beta_local(n_20, E_b_100keV, B_central)
    if beta_local is None or np.isnan(beta_local):
        return False, "beta_invalid"

    B_0 = calculate_B0_with_diamagnetic(B_central, beta_local)

    # Calculate vacuum mirror ratio
    R_M_vac = B_max / B_central

    # Calculate geometry constraints
    a_0_abs = calculate_a0_absorption(E_b_100keV, n_20)
    a_0_FLR = calculate_a0_FLR(E_b_100keV, B_0, N_25)
    a_0_min = max(a_0_abs, a_0_FLR)

    # Minimum radius check
    if a_0_min < min_a0:
        return False, "min_radius"

    # Calculate geometry
    a_0_FLR_mirror = calculate_a0_FLR_at_mirror(E_b_100keV, B_max, N_25)
    L_plasma, V_plasma, vessel_surface_area = calculate_plasma_geometry_frustum(
        a_0_min, a_0_FLR_mirror, N_rho
    )

    # Calculate loss coefficient
    C_loss = calculate_loss_coefficient(E_b_100keV, R_M_vac)

    # Calculate required NBI power
    P_NBI = calculate_NBI_power(n_20, V_plasma, E_b_100keV, R_M_vac, C_loss)

    # Calculate fusion power
    T_i = T_i_coeff * E_b_100keV * 100  # Convert to keV
    P_fusion = calculate_fusion_power(E_b_100keV, n_20, V_plasma, T_i)
    Q = calculate_Q(P_fusion, P_NBI)

    # Heat flux check
    Bw = calculate_Bw(E_b_100keV, B_0, a_0_min)
    q_w = calculate_heat_flux(P_NBI, Q, a_0_min, B_0, Bw)
    if q_w >= qw_limit:
        return False, "heat_flux"

    return True, None


def find_n20_for_PNBI(E_b_100keV, P_NBI_target, B_central, B_max, n_20_search_range):
    """
    Find all n20 values that satisfy P_NBI = P_NBI_target for given E_b and Rm

    Returns: list of (n_20, P_fusion, feasible) tuples
    """
    R_M_vac = B_max / B_central

    # Create fine grid to search for intersections
    n_20_grid = np.linspace(n_20_search_range[0], n_20_search_range[1], 1000)
    P_NBI_grid = []

    for n_20 in n_20_grid:
        # Calculate beta and check validity
        beta_local = calculate_beta_local(n_20, E_b_100keV, B_central)
        if beta_local is None or np.isnan(beta_local):
            P_NBI_grid.append(np.nan)
            continue

        B_0 = calculate_B0_with_diamagnetic(B_central, beta_local)

        # Calculate geometry
        a_0_abs = calculate_a0_absorption(E_b_100keV, n_20)
        a_0_FLR = calculate_a0_FLR(E_b_100keV, B_0, N_25)
        a_0_min = max(a_0_abs, a_0_FLR)

        a_0_FLR_mirror = calculate_a0_FLR_at_mirror(E_b_100keV, B_max, N_25)
        L_plasma, V_plasma, vessel_surface_area = calculate_plasma_geometry_frustum(
            a_0_min, a_0_FLR_mirror, N_rho
        )

        # Calculate loss coefficient
        C_loss = calculate_loss_coefficient(E_b_100keV, R_M_vac)

        # Calculate NBI power
        P_NBI = calculate_NBI_power(n_20, V_plasma, E_b_100keV, R_M_vac, C_loss)
        P_NBI_grid.append(P_NBI)

    P_NBI_grid = np.array(P_NBI_grid)

    # Find where P_NBI crosses P_NBI_target
    crossings = []
    for i in range(len(P_NBI_grid) - 1):
        if np.isnan(P_NBI_grid[i]) or np.isnan(P_NBI_grid[i+1]):
            continue

        # Check if there's a crossing
        if (P_NBI_grid[i] - P_NBI_target) * (P_NBI_grid[i+1] - P_NBI_target) < 0:
            # Refine crossing location using interpolation
            try:
                n_20_cross = n_20_grid[i] + (P_NBI_target - P_NBI_grid[i]) * \
                             (n_20_grid[i+1] - n_20_grid[i]) / (P_NBI_grid[i+1] - P_NBI_grid[i])

                # Calculate P_fusion at this point
                beta_local = calculate_beta_local(n_20_cross, E_b_100keV, B_central)
                B_0 = calculate_B0_with_diamagnetic(B_central, beta_local)

                a_0_abs = calculate_a0_absorption(E_b_100keV, n_20_cross)
                a_0_FLR = calculate_a0_FLR(E_b_100keV, B_0, N_25)
                a_0_min = max(a_0_abs, a_0_FLR)

                a_0_FLR_mirror = calculate_a0_FLR_at_mirror(E_b_100keV, B_max, N_25)
                L_plasma, V_plasma, vessel_surface_area = calculate_plasma_geometry_frustum(
                    a_0_min, a_0_FLR_mirror, N_rho
                )

                T_i = T_i_coeff * E_b_100keV * 100
                P_fusion = calculate_fusion_power(E_b_100keV, n_20_cross, V_plasma, T_i)

                # Check feasibility
                feasible, reason = check_feasibility(E_b_100keV, n_20_cross, B_central, B_max)

                crossings.append((n_20_cross, P_fusion, feasible, reason))
            except:
                continue

    return crossings


def scan_Rm_for_PNBI(E_b_100keV, P_NBI_target, Rm_range, n_Rm_points=50):
    """
    Scan through different mirror ratios and find maximum P_fusion for given PNBI

    For each Rm, finds all points where PNBI contour intersects (typically 2 branches),
    and returns the maximum feasible P_fusion.

    Returns:
        Rm_vac_array: array of mirror ratios
        P_fusion_max_array: maximum fusion power at each Rm
        n_20_opt_array: optimal density at each Rm
    """
    Rm_vac_array = np.linspace(Rm_range[0], Rm_range[1], n_Rm_points)
    P_fusion_max_array = []
    n_20_opt_array = []

    for Rm_vac in Rm_vac_array:
        B_central = B_max_default / Rm_vac

        # Calculate beta limit for this configuration
        n_20_max = calculate_beta_limit(E_b_100keV, B_central, beta_c_default)

        # Find intersections with PNBI contour
        crossings = find_n20_for_PNBI(
            E_b_100keV, P_NBI_target, B_central, B_max_default,
            n_20_search_range=(0.01, n_20_max)
        )

        # Find maximum feasible P_fusion
        max_P_fusion = 0
        opt_n20 = None

        for n_20, P_fusion, feasible, reason in crossings:
            if feasible and P_fusion > max_P_fusion:
                max_P_fusion = P_fusion
                opt_n20 = n_20

        P_fusion_max_array.append(max_P_fusion if max_P_fusion > 0 else np.nan)
        n_20_opt_array.append(opt_n20 if opt_n20 is not None else np.nan)

    return Rm_vac_array, np.array(P_fusion_max_array), np.array(n_20_opt_array)


def plot_Pfusion_vs_Rm(E_b_100keV, P_NBI_list, Rm_range=(4, 16), n_Rm_points=50):
    """
    Create plot of P_fusion vs Rm for different PNBI levels at a fixed beam energy

    Args:
        E_b_100keV: beam energy [100 keV units]
        P_NBI_list: list of PNBI values [MW]
        Rm_range: (min, max) mirror ratio range
        n_Rm_points: number of points in Rm scan
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    E_b_keV = E_b_100keV * 100
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(P_NBI_list)))

    for P_NBI_idx, P_NBI_target in enumerate(P_NBI_list):
        print(f"\nScanning Rm for E_b = {E_b_keV} keV, P_NBI = {P_NBI_target} MW...")

        Rm_array, P_fusion_array, n20_array = scan_Rm_for_PNBI(
            E_b_100keV, P_NBI_target, Rm_range, n_Rm_points
        )

        # Plot only where P_fusion is valid
        valid_mask = ~np.isnan(P_fusion_array) & (P_fusion_array > 0)

        if np.any(valid_mask):
            ax.plot(Rm_array[valid_mask], P_fusion_array[valid_mask],
                   linewidth=1.5, color=colors[P_NBI_idx],
                   label=f'$P_{{NBI}}$ = {P_NBI_target} MW', marker='o', markersize=2)

            # Find and report maximum (but don't plot the star)
            max_idx = np.nanargmax(P_fusion_array)
            print(f"  Max P_fusion = {P_fusion_array[max_idx]:.2f} MW at Rm = {Rm_array[max_idx]:.2f}")
            print(f"  Optimal n_20 = {n20_array[max_idx]:.3f}")
        else:
            print(f"  No feasible points found!")

    ax.set_xlabel(r'$R_{M,vac}$ (Vacuum Mirror Ratio)', fontsize=16)
    ax.set_ylabel(r'$P_{fusion}$ [MW]', fontsize=16)
    ax.set_title(f'Maximum Fusion Power vs Mirror Ratio for Fixed $P_{{NBI}}$ Contours\n'
                 f'($E_b$ = {E_b_keV:.0f} keV, $B_{{max}}$ = {B_max_default} T)',
                 fontsize=18, weight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    ax.legend(fontsize=14, loc='best', framealpha=0.9)
    ax.tick_params(labelsize=13)

    plt.tight_layout()

    return fig


def find_optimal_Rm_for_Eb_PNBI(E_b_100keV, P_NBI_target, Rm_range, n_Rm_points=50):
    """
    For a given (Eb, PNBI), find the optimal Rm that maximizes P_fusion

    Returns:
        Rm_optimal: Optimal mirror ratio
        P_fusion_max: Maximum fusion power at optimal Rm
        n_20_opt: Optimal density
    """
    Rm_array, P_fusion_array, n20_array = scan_Rm_for_PNBI(
        E_b_100keV, P_NBI_target, Rm_range, n_Rm_points
    )

    valid_mask = ~np.isnan(P_fusion_array) & (P_fusion_array > 0)

    if np.any(valid_mask):
        max_idx = np.nanargmax(P_fusion_array)
        return Rm_array[max_idx], P_fusion_array[max_idx], n20_array[max_idx]
    else:
        return None, None, None


def plot_Eb_optimization_for_PNBI(P_NBI_list, Eb_range=(0.4, 1.2), Rm_range=(4, 16), n_Eb_points=25, n_Rm_points=50):
    """
    Create 3-subplot figure showing optimal operating points vs beam energy

    For each PNBI level, scans through Eb and finds optimal Rm that maximizes P_fusion

    Left: P_fusion vs Eb (at optimal Rm for each point)
    Middle: Optimal Rm vs Eb
    Right: Voltage requirements vs Eb

    Args:
        P_NBI_list: list of PNBI values [MW]
        Eb_range: (min, max) beam energy range [100 keV units]
        Rm_range: (min, max) mirror ratio range for optimization
        n_Eb_points: number of beam energy points to scan
        n_Rm_points: number of Rm points for optimization at each Eb
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 7))

    colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(P_NBI_list)))

    all_results = {}

    for P_NBI_idx, P_NBI_target in enumerate(P_NBI_list):
        print(f"\n{'='*80}")
        print(f"Optimizing Rm for each Eb at P_NBI = {P_NBI_target} MW...")
        print(f"{'='*80}")

        Eb_array = np.linspace(Eb_range[0], Eb_range[1], n_Eb_points)
        P_fusion_opt_array = []
        Rm_opt_array = []
        n20_opt_array = []
        Q_opt_array = []
        voltage_cl_array = []
        voltage_fr_array = []
        Eb_valid = []

        for E_b_100keV in Eb_array:
            E_b_keV = E_b_100keV * 100

            # Find optimal Rm for this (Eb, PNBI)
            Rm_opt, P_fusion_max, n20_opt = find_optimal_Rm_for_Eb_PNBI(
                E_b_100keV, P_NBI_target, Rm_range, n_Rm_points
            )

            if Rm_opt is not None:
                Q_opt = P_fusion_max / P_NBI_target

                # Calculate voltage requirements at this optimal point
                B_central = B_max_default / Rm_opt
                beta_local = calculate_beta_local(n20_opt, E_b_100keV, B_central)
                B_0 = calculate_B0_with_diamagnetic(B_central, beta_local)

                # Calculate geometry
                a_0_abs = calculate_a0_absorption(E_b_100keV, n20_opt)
                a_0_FLR = calculate_a0_FLR(E_b_100keV, B_0, N_25)
                a_0_min = max(a_0_abs, a_0_FLR)

                a_0_FLR_mirror = calculate_a0_FLR_at_mirror(E_b_100keV, B_max_default, N_25)
                L_plasma, V_plasma, vessel_surface_area = calculate_plasma_geometry_frustum(
                    a_0_min, a_0_FLR_mirror, N_rho
                )

                # Diamagnetic mirror ratio
                R_M_dmag = B_max_default / B_0

                # Calculate voltages
                voltage_cl = calculate_voltage_closed_lines(E_b_100keV, B_0, a_0_min, L_plasma, R_M_dmag)
                voltage_fr = calculate_voltage_field_reversal(E_b_100keV, B_0, a_0_min, L_plasma, R_M_dmag)

                P_fusion_opt_array.append(P_fusion_max)
                Rm_opt_array.append(Rm_opt)
                n20_opt_array.append(n20_opt)
                Q_opt_array.append(Q_opt)
                voltage_cl_array.append(voltage_cl)
                voltage_fr_array.append(voltage_fr)
                Eb_valid.append(E_b_keV)

                print(f"  E_b = {E_b_keV:.1f} keV: Rm_opt = {Rm_opt:.2f}, "
                      f"P_fusion_max = {P_fusion_max:.2f} MW, Q = {Q_opt:.2f}")

        # Store results
        all_results[P_NBI_target] = {
            'Eb': np.array(Eb_valid),
            'P_fusion_opt': np.array(P_fusion_opt_array),
            'Rm_opt': np.array(Rm_opt_array),
            'n20_opt': np.array(n20_opt_array),
            'Q_opt': np.array(Q_opt_array),
            'voltage_cl': np.array(voltage_cl_array),
            'voltage_fr': np.array(voltage_fr_array)
        }

        if len(Eb_valid) > 0:
            # LEFT: P_fusion vs Eb
            ax1.plot(Eb_valid, P_fusion_opt_array, color=colors[P_NBI_idx],
                    linewidth=1.5, marker='o', markersize=2,
                    label=f'$P_{{NBI}}$ = {P_NBI_target} MW')

            # MIDDLE: Optimal Rm vs Eb
            ax2.plot(Eb_valid, Rm_opt_array, color=colors[P_NBI_idx],
                    linewidth=1.5, marker='o', markersize=2,
                    label=f'$P_{{NBI}}$ = {P_NBI_target} MW')

            # RIGHT: Voltage requirements (use different colors for each type)
            # Closed lines - use blue shades
            color_cl = plt.cm.Blues(0.5 + 0.3 * P_NBI_idx / len(P_NBI_list))
            ax3.plot(Eb_valid, voltage_cl_array, color=color_cl,
                    linewidth=1.5, marker='o', markersize=2, linestyle='-',
                    label=f'$P_{{NBI}}$ = {P_NBI_target} MW (closed lines)')

            # Field reversal - use orange/red shades
            color_fr = plt.cm.Oranges(0.5 + 0.3 * P_NBI_idx / len(P_NBI_list))
            ax3.plot(Eb_valid, voltage_fr_array, color=color_fr,
                    linewidth=1.5, marker='s', markersize=2, linestyle='-',
                    label=f'$P_{{NBI}}$ = {P_NBI_target} MW (field reversal)')

    # LEFT subplot formatting
    ax1.set_xlabel(r'Beam Energy $E_{NBI}$ [keV]', fontsize=13, fontweight='bold')
    ax1.set_ylabel(r'Maximum $P_{fusion}$ [MW]', fontsize=13, fontweight='bold')
    ax1.set_title('Maximum Fusion Power vs Beam Energy\n(at optimal $R_M$ for each point)',
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    ax1.tick_params(labelsize=11)

    # MIDDLE subplot formatting
    ax2.set_xlabel(r'Beam Energy $E_{NBI}$ [keV]', fontsize=13, fontweight='bold')
    ax2.set_ylabel(r'Optimal $R_{M,vac}$ (Vacuum Mirror Ratio)', fontsize=13, fontweight='bold')
    ax2.set_title('Optimal Mirror Ratio vs Beam Energy\n(for maximum $P_{fusion}$ at fixed $P_{NBI}$)',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    ax2.tick_params(labelsize=11)

    # RIGHT subplot formatting
    ax3.set_xlabel(r'Beam Energy $E_{NBI}$ [keV]', fontsize=13, fontweight='bold')
    ax3.set_ylabel(r'Voltage $(e\phi/T_e)$', fontsize=13, fontweight='bold')
    ax3.set_title('Vortex Stabilization Voltage\n(solid=closed lines, dashed=field reversal)',
                  fontsize=12, fontweight='bold')

    # Add red dashed line at voltage = 2
    ax3.axhline(y=2, color='red', linestyle='--', linewidth=2, label='Target voltage = 2')

    ax3.legend(fontsize=8, loc='best', framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    ax3.tick_params(labelsize=11)

    # Main title
    fig.suptitle(f'Optimal Operating Points for Fixed $P_{{NBI}}$ Contours ($B_{{max}}$ = {B_max_default} T)',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()

    return fig, all_results


if __name__ == "__main__":
    print("="*80)
    print("Mirror Ratio Optimization for Maximum Fusion Power")
    print(f"B_max = {B_max_default} T")
    print("="*80)

    # Define beam energy [100 keV units] - can be changed as input
    E_b_100keV = 1.0  # 100 keV

    # Define PNBI levels [MW]
    P_NBI_list = [34]  # Main analysis with 2 levels

    # Mirror ratio range
    Rm_range = (4, 40)

    # Beam energy range for optimization analysis
    Eb_range = (0.4, 1.2)  # 40 to 120 keV

    # =========================================================================
    # ANALYSIS 1: P_fusion vs Rm at fixed Eb
    # =========================================================================
    print("\n" + "="*80)
    print("ANALYSIS 1: P_fusion vs Rm at Fixed Beam Energy")
    print("="*80)

    # Create summary table (for this specific beam energy)
    print(f"\n{'E_b [keV]':>10} {'P_NBI [MW]':>12} {'Rm_opt':>10} {'P_fus_max [MW]':>16} "
          f"{'n_20_opt':>12} {'Q':>8}")
    print("-"*80)

    E_b_keV = E_b_100keV * 100
    for P_NBI_target in P_NBI_list:
        Rm_array, P_fusion_array, n20_array = scan_Rm_for_PNBI(
            E_b_100keV, P_NBI_target, Rm_range, n_Rm_points=500
        )

        valid_mask = ~np.isnan(P_fusion_array) & (P_fusion_array > 0)

        if np.any(valid_mask):
            max_idx = np.nanargmax(P_fusion_array)
            Rm_opt = Rm_array[max_idx]
            P_fus_max = P_fusion_array[max_idx]
            n20_opt = n20_array[max_idx]
            Q_opt = P_fus_max / P_NBI_target

            print(f"{E_b_keV:10.0f} {P_NBI_target:12.1f} {Rm_opt:10.2f} {P_fus_max:16.2f} "
                  f"{n20_opt:12.4f} {Q_opt:8.3f}")
        else:
            print(f"{E_b_keV:10.0f} {P_NBI_target:12.1f} {'N/A':>10} {'N/A':>16} "
                  f"{'N/A':>12} {'N/A':>8}")

    # Create plot 1
    print("\nGenerating plot 1 (P_fusion vs Rm at fixed Eb)...")
    fig1 = plot_Pfusion_vs_Rm(E_b_100keV, P_NBI_list, Rm_range, n_Rm_points=500)

    # Save figure 1
    output_path1 = figures_dir / f'Pfus_vs_Rm_for_PNBI_contours_Eb{int(E_b_keV)}keV.png'
    fig1.savefig(output_path1, dpi=figure_dpi, bbox_inches='tight')
    print(f"Saved: {output_path1}")

    # =========================================================================
    # ANALYSIS 2: Optimal Rm for each Eb (maximizing P_fusion)
    # =========================================================================
    print("\n" + "="*80)
    print("ANALYSIS 2: Optimal Rm vs Eb (Maximizing P_fusion)")
    print("="*80)

    fig2, all_results = plot_Eb_optimization_for_PNBI(
        P_NBI_list, Eb_range, Rm_range, n_Eb_points=100, n_Rm_points=450
    )

    # Print summary for analysis 2
    print("\n" + "="*80)
    print("SUMMARY: Optimal Operating Points")
    print("="*80)

    for P_NBI_target in P_NBI_list:
        data = all_results[P_NBI_target]

        if len(data['Eb']) == 0:
            continue

        print(f"\nP_NBI = {P_NBI_target} MW:")
        print(f"{'E_b [keV]':>10} {'Rm_opt':>10} {'P_fus_max [MW]':>16} {'n_20_opt':>12} {'Q':>8}")
        print("-"*80)

        for i in range(len(data['Eb'])):
            print(f"{data['Eb'][i]:10.1f} {data['Rm_opt'][i]:10.2f} "
                  f"{data['P_fusion_opt'][i]:16.2f} {data['n20_opt'][i]:12.4f} {data['Q_opt'][i]:8.3f}")

        # Find maximum P_fusion for this PNBI
        max_idx = np.argmax(data['P_fusion_opt'])
        print(f"\n*** MAXIMUM P_fusion for P_NBI = {P_NBI_target} MW:")
        print(f"    E_b = {data['Eb'][max_idx]:.1f} keV")
        print(f"    Rm_opt = {data['Rm_opt'][max_idx]:.2f}")
        print(f"    P_fusion_max = {data['P_fusion_opt'][max_idx]:.2f} MW")
        print(f"    Q = {data['Q_opt'][max_idx]:.2f}")

    # Save figure 2
    output_path2 = figures_dir / 'Optimization_Rm_vs_Eb_for_PNBI_contours.png'
    fig2.savefig(output_path2, dpi=figure_dpi, bbox_inches='tight')
    print(f"\nSaved: {output_path2}")

    # =========================================================================
    # ANALYSIS 3: Find optimal point meeting voltage stabilization requirements
    # =========================================================================
    print("\n" + "="*80)
    print("ANALYSIS 3: Optimal Point Meeting Voltage Stabilization Requirements")
    print("="*80)
    print("\nSearching for the highest fusion power point where BOTH voltages < 2...")

    # Store all feasible points across all PNBI values
    best_point = None
    best_P_fusion = 0

    for P_NBI_target in P_NBI_list:
        data = all_results[P_NBI_target]

        if len(data['Eb']) == 0:
            continue

        # Find all points where both voltages are below 2
        voltage_satisfied = (data['voltage_cl'] < 2) & (data['voltage_fr'] < 2)

        for i in range(len(data['Eb'])):
            if voltage_satisfied[i] and data['P_fusion_opt'][i] > best_P_fusion:
                best_P_fusion = data['P_fusion_opt'][i]
                best_point = {
                    'P_NBI': P_NBI_target,
                    'Eb': data['Eb'][i],
                    'Eb_100keV': data['Eb'][i] / 100,
                    'Rm_opt': data['Rm_opt'][i],
                    'n20_opt': data['n20_opt'][i],
                    'P_fusion': data['P_fusion_opt'][i],
                    'Q': data['Q_opt'][i],
                    'voltage_cl': data['voltage_cl'][i],
                    'voltage_fr': data['voltage_fr'][i]
                }

    if best_point is None:
        print("\n*** WARNING: No points found that satisfy both voltage requirements (< 2)!")
    else:
        print(f"\n*** OPTIMAL DESIGN POINT (Highest P_fusion with both voltages < 2) ***\n")

        # Recalculate all detailed parameters for this point
        E_b_100keV = best_point['Eb_100keV']
        Rm_opt = best_point['Rm_opt']
        n20_opt = best_point['n20_opt']
        P_NBI = best_point['P_NBI']

        B_central = B_max_default / Rm_opt
        beta_local = calculate_beta_local(n20_opt, E_b_100keV, B_central)
        B_0 = calculate_B0_with_diamagnetic(B_central, beta_local)

        # Calculate geometry
        a_0_abs = calculate_a0_absorption(E_b_100keV, n20_opt)
        a_0_FLR = calculate_a0_FLR(E_b_100keV, B_0, N_25)
        a_0_min = max(a_0_abs, a_0_FLR)

        a_0_FLR_mirror = calculate_a0_FLR_at_mirror(E_b_100keV, B_max_default, N_25)
        L_plasma, V_plasma, vessel_surface_area = calculate_plasma_geometry_frustum(
            a_0_min, a_0_FLR_mirror, N_rho
        )

        # Calculate additional parameters
        R_M_dmag = B_max_default / B_0
        C_loss = calculate_loss_coefficient(E_b_100keV, Rm_opt)
        T_i = T_i_coeff * E_b_100keV * 100  # Convert to keV

        # Calculate heat flux and wall field
        Bw = calculate_Bw(E_b_100keV, B_0, a_0_min)
        q_w = calculate_heat_flux(P_NBI, best_point['Q'], a_0_min, B_0, Bw)

        # Print ALL design parameters
        print("="*80)
        print("PLASMA PARAMETERS:")
        print("="*80)
        print(f"  Beam Energy (E_b):                    {best_point['Eb']:.2f} keV")
        print(f"  Ion Temperature (T_i):                {T_i:.2f} keV")
        print(f"  Density (n_20):                       {n20_opt:.4f} × 10²⁰ m⁻³")
        print(f"  Density (n):                          {n20_opt * 1e20:.3e} m⁻³")
        print(f"  Local Beta (β_local):                 {beta_local:.4f}")
        print(f"  Beta Limit Percentage:                {beta_local / beta_c_default * 100:.2f}%")
        print()
        print("="*80)
        print("MAGNETIC FIELD PARAMETERS:")
        print("="*80)
        print(f"  Maximum Field (B_max):                {B_max_default:.2f} T")
        print(f"  Central Vacuum Field (B_central):     {B_central:.4f} T")
        print(f"  Beta-Adjusted Central Field (B_0):    {B_0:.4f} T")
        print(f"  Wall Field (B_w):                     {Bw:.4f} T")
        print(f"  Vacuum Mirror Ratio (R_M_vac):        {Rm_opt:.4f}")
        print(f"  Diamagnetic Mirror Ratio (R_M_dmag):  {R_M_dmag:.4f}")
        print()
        print("="*80)
        print("GEOMETRY PARAMETERS:")
        print("="*80)
        print(f"  Minor Radius - Center (a_0_min):      {a_0_min:.4f} m")
        print(f"  Minor Radius - Mirror (a_0_FLR_mir):  {a_0_FLR_mirror:.4f} m")
        print(f"  Minor Radius from Absorption:         {a_0_abs:.4f} m")
        print(f"  Minor Radius from FLR:                {a_0_FLR:.4f} m")
        print(f"  Plasma Length (L_plasma):             {L_plasma:.4f} m")
        print(f"  Plasma Volume (V_plasma):             {V_plasma:.4f} m³")
        print(f"  Vessel Surface Area:                  {vessel_surface_area:.4f} m²")
        print(f"  Aspect Ratio (L/2a_0):                {L_plasma / (2 * a_0_min):.4f}")
        print()
        print("="*80)
        print("POWER PARAMETERS:")
        print("="*80)
        print(f"  NBI Power (P_NBI):                    {P_NBI:.2f} MW")
        print(f"  Fusion Power (P_fusion):              {best_point['P_fusion']:.2f} MW")
        print(f"  Power Gain (Q):                       {best_point['Q']:.4f}")
        print(f"  Loss Coefficient (C_loss):            {C_loss:.6f}")
        print(f"  Heat Flux (q_w):                      {q_w:.4f} MW/m²")
        print(f"  Heat Flux Limit:                      {qw_limit:.4f} MW/m²")
        print(f"  Heat Flux Usage:                      {q_w/qw_limit*100:.2f}%")
        print()
        print("="*80)
        print("VOLTAGE STABILITY PARAMETERS:")
        print("="*80)
        print(f"  Voltage (Closed Lines):               {best_point['voltage_cl']:.4f}")
        print(f"  Voltage (Field Reversal):             {best_point['voltage_fr']:.4f}")
        print(f"  Both voltages < 2:                    {'✓ YES' if best_point['voltage_cl'] < 2 and best_point['voltage_fr'] < 2 else '✗ NO'}")
        print()
        print("="*80)
        print("DESIGN PARAMETERS (N_25, N_rho, beta_c):")
        print("="*80)
        print(f"  N_25 (FLR constraint parameter):      {N_25:.2f}")
        print(f"  N_rho (aspect ratio parameter):       {N_rho:.2f}")
        print(f"  Beta_c (critical beta):               {beta_c_default:.4f}")
        print("="*80)
        print()

        # Also save this to a file
        output_file = figures_dir / 'optimal_design_point.txt'
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("OPTIMAL DESIGN POINT (Highest P_fusion with both voltages < 2)\n")
            f.write("="*80 + "\n\n")

            f.write("="*80 + "\n")
            f.write("PLASMA PARAMETERS:\n")
            f.write("="*80 + "\n")
            f.write(f"  Beam Energy (E_b):                    {best_point['Eb']:.2f} keV\n")
            f.write(f"  Ion Temperature (T_i):                {T_i:.2f} keV\n")
            f.write(f"  Density (n_20):                       {n20_opt:.4f} × 10²⁰ m⁻³\n")
            f.write(f"  Density (n):                          {n20_opt * 1e20:.3e} m⁻³\n")
            f.write(f"  Local Beta (β_local):                 {beta_local:.4f}\n")
            f.write(f"  Beta Limit Percentage:                {beta_local / beta_c_default * 100:.2f}%\n")
            f.write("\n")
            f.write("="*80 + "\n")
            f.write("MAGNETIC FIELD PARAMETERS:\n")
            f.write("="*80 + "\n")
            f.write(f"  Maximum Field (B_max):                {B_max_default:.2f} T\n")
            f.write(f"  Central Vacuum Field (B_central):     {B_central:.4f} T\n")
            f.write(f"  Beta-Adjusted Central Field (B_0):    {B_0:.4f} T\n")
            f.write(f"  Wall Field (B_w):                     {Bw:.4f} T\n")
            f.write(f"  Vacuum Mirror Ratio (R_M_vac):        {Rm_opt:.4f}\n")
            f.write(f"  Diamagnetic Mirror Ratio (R_M_dmag):  {R_M_dmag:.4f}\n")
            f.write("\n")
            f.write("="*80 + "\n")
            f.write("GEOMETRY PARAMETERS:\n")
            f.write("="*80 + "\n")
            f.write(f"  Minor Radius - Center (a_0_min):      {a_0_min:.4f} m\n")
            f.write(f"  Minor Radius - Mirror (a_0_FLR_mir):  {a_0_FLR_mirror:.4f} m\n")
            f.write(f"  Minor Radius from Absorption:         {a_0_abs:.4f} m\n")
            f.write(f"  Minor Radius from FLR:                {a_0_FLR:.4f} m\n")
            f.write(f"  Plasma Length (L_plasma):             {L_plasma:.4f} m\n")
            f.write(f"  Plasma Volume (V_plasma):             {V_plasma:.4f} m³\n")
            f.write(f"  Vessel Surface Area:                  {vessel_surface_area:.4f} m²\n")
            f.write(f"  Aspect Ratio (L/2a_0):                {L_plasma / (2 * a_0_min):.4f}\n")
            f.write("\n")
            f.write("="*80 + "\n")
            f.write("POWER PARAMETERS:\n")
            f.write("="*80 + "\n")
            f.write(f"  NBI Power (P_NBI):                    {P_NBI:.2f} MW\n")
            f.write(f"  Fusion Power (P_fusion):              {best_point['P_fusion']:.2f} MW\n")
            f.write(f"  Power Gain (Q):                       {best_point['Q']:.4f}\n")
            f.write(f"  Loss Coefficient (C_loss):            {C_loss:.6f}\n")
            f.write(f"  Heat Flux (q_w):                      {q_w:.4f} MW/m²\n")
            f.write(f"  Heat Flux Limit:                      {qw_limit:.4f} MW/m²\n")
            f.write(f"  Heat Flux Usage:                      {q_w/qw_limit*100:.2f}%\n")
            f.write("\n")
            f.write("="*80 + "\n")
            f.write("VOLTAGE STABILITY PARAMETERS:\n")
            f.write("="*80 + "\n")
            f.write(f"  Voltage (Closed Lines):               {best_point['voltage_cl']:.4f}\n")
            f.write(f"  Voltage (Field Reversal):             {best_point['voltage_fr']:.4f}\n")
            f.write(f"  Both voltages < 2:                    {'✓ YES' if best_point['voltage_cl'] < 2 and best_point['voltage_fr'] < 2 else '✗ NO'}\n")
            f.write("\n")
            f.write("="*80 + "\n")
            f.write("DESIGN PARAMETERS (N_25, N_rho, beta_c):\n")
            f.write("="*80 + "\n")
            f.write(f"  N_25 (FLR constraint parameter):      {N_25:.2f}\n")
            f.write(f"  N_rho (aspect ratio parameter):       {N_rho:.2f}\n")
            f.write(f"  Beta_c (critical beta):               {beta_c_default:.4f}\n")
            f.write("="*80 + "\n")

        print(f"Design point details saved to: {output_file}")

    plt.show()
