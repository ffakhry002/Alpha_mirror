"""optimization_analysis.py - Find optimal mirror ratios for different beam energies
For each (E_b, NWL) combination, scan R_M to find minimum P_NBI operating point
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from equations import (
    calculate_loss_coefficient, calculate_beta_local, calculate_B0_with_diamagnetic,
    calculate_beta_limit, calculate_a0_absorption, calculate_a0_FLR,
    calculate_a0_adiabaticity, calculate_plasma_geometry_frustum, calculate_a0_end,
    calculate_fusion_power, calculate_NBI_power, calculate_NWL,
    get_dt_reactivity
)

from n20_Eb_inputs import (
    B_max_default, B_central_default, beta_c_default, T_i_coeff,
    figures_dir, figure_dpi, min_a0
)

# Analysis parameters
E_b_range = np.linspace(20, 120, 25)  # Beam energy range (keV)
R_M_scan_range = np.linspace(4, 16, 500)  # Vacuum mirror ratio scan range (increased resolution)
NWL_targets = [0.75, 1.0, 1.25, 1.5]  # Target NWLs


def find_n20_for_target_NWL(E_b_keV, R_M, target_NWL, B_max, beta_c, n_20_min=0.01):
    """Solve for n_20 that gives target NWL using bisection method"""
    E_b_100keV = E_b_keV / 100.0
    B_central = B_max / R_M

    n_20_beta_max = calculate_beta_limit(E_b_100keV, B_central, beta_c)
    beta_local_max = calculate_beta_local(n_20_beta_max, E_b_100keV, B_central)

    if beta_local_max is None or np.isnan(beta_local_max):
        return None, None, None, None, None, None, None

    B_0_max = calculate_B0_with_diamagnetic(B_central, beta_local_max)
    a_0_abs_max = calculate_a0_absorption(E_b_100keV, n_20_beta_max)
    a_0_FLR_max = calculate_a0_FLR(E_b_100keV, B_0_max, N_25)
    a_0_min_max = np.maximum(a_0_abs_max, a_0_FLR_max)

    a_0_FLR_mirror_max = calculate_a0_FLR_at_mirror(E_b_100keV, B_max, N_25)
    L_plasma_max, V_plasma_max, vessel_surface_area_max = calculate_plasma_geometry_frustum(
        a_0_min_max, a_0_FLR_mirror_max, N_rho
    )

    T_i = T_i_coeff * E_b_keV
    P_fusion_max = calculate_fusion_power(E_b_100keV, n_20_beta_max, V_plasma_max, T_i)
    NWL_max = calculate_NWL(P_fusion_max, vessel_surface_area_max)

    if target_NWL > NWL_max * 1.01:
        return None, None, None, None, None, None, None

    # Bisection method
    tolerance, max_iterations = 1e-6, 100
    n_20_search_max, n_20_search_min = n_20_beta_max, n_20_min

    for iteration in range(max_iterations):
        n_20_mid = (n_20_search_min + n_20_search_max) / 2
        beta_local = calculate_beta_local(n_20_mid, E_b_100keV, B_central)

        if beta_local is None or np.isnan(beta_local):
            n_20_search_max = n_20_mid
            continue

        B_0 = calculate_B0_with_diamagnetic(B_central, beta_local)
        B_0_conductor = B_central

        a_0_abs = calculate_a0_absorption(E_b_100keV, n_20_mid)
        a_0_FLR = calculate_a0_FLR(E_b_100keV, B_0, N_25)
        a_0_min = np.maximum(a_0_abs, a_0_FLR)

        a_0_FLR_mirror = calculate_a0_FLR_at_mirror(E_b_100keV, B_max, N_25)
        L_plasma, V_plasma, vessel_surface_area = calculate_plasma_geometry_frustum(
            a_0_min, a_0_FLR_mirror, N_rho
        )

        C_loss = calculate_loss_coefficient(E_b_100keV, R_M)
        P_fusion = calculate_fusion_power(E_b_100keV, n_20_mid, V_plasma, T_i)
        NWL_current = calculate_NWL(P_fusion, vessel_surface_area)

        if abs(NWL_current - target_NWL) / target_NWL < tolerance:
            P_NBI = calculate_NBI_power(n_20_mid, V_plasma, E_b_100keV, R_M, C_loss)
            return n_20_mid, P_NBI, NWL_current, a_0_abs, a_0_FLR, B_0, B_0_conductor

        if NWL_current < target_NWL:
            n_20_search_min = n_20_mid
        else:
            n_20_search_max = n_20_mid

        if abs(n_20_search_max - n_20_search_min) < 1e-10:
            break

    return None, None, None, None, None, None, None


def find_optimal_Rm_for_Eb_NWL(E_b_keV, target_NWL, B_max, beta_c):
    """Scan R_M from 4-16 to find the one that minimizes P_NBI

    Returns:
        R_M_vacuum_optimal: Optimal vacuum mirror ratio
        R_M_diamagnetic_optimal: Optimal diamagnetic mirror ratio
        P_NBI_min: Minimum NBI power (MW)
        P_fusion_opt: Fusion power at optimal point (MW)
        n_20_opt: Density at optimal point
        a_0_opt: Minor radius at optimal point
        B_0_opt: On-axis field at optimal point
        B_0_conductor_opt: Conductor field at optimal point
        regime: 'Absorption' or 'FLR' limited
    """
    P_NBI_array = []
    P_fusion_array = []
    R_M_valid = []
    n_20_array = []
    a_0_array = []
    B_0_array = []
    B_0_conductor_array = []
    a_0_abs_array = []
    a_0_FLR_array = []

    E_b_100keV = E_b_keV / 100.0
    T_i = T_i_coeff * E_b_keV

    for R_M_vacuum in R_M_scan_range:
        try:
            result = find_n20_for_target_NWL(E_b_keV, R_M_vacuum, target_NWL, B_max, beta_c)
            n_20, P_NBI, NWL_achieved, a_0_abs, a_0_FLR, B_0, B_0_conductor = result

            if P_NBI is not None and not np.isnan(P_NBI):
                # Calculate P_fusion at this point
                a_0_min = max(a_0_abs, a_0_FLR)

                # Skip if below minimum a0 threshold
                if a_0_min < min_a0:
                    continue

                a_0_FLR_mirror = calculate_a0_FLR_at_mirror(E_b_100keV, B_max, N_25)
                L_plasma, V_plasma, _ = calculate_plasma_geometry_frustum(a_0_min, a_0_FLR_mirror, N_rho)
                P_fusion = calculate_fusion_power(E_b_100keV, n_20, V_plasma, T_i)

                P_NBI_array.append(P_NBI)
                P_fusion_array.append(P_fusion)
                R_M_valid.append(R_M_vacuum)
                n_20_array.append(n_20)
                a_0_array.append(a_0_min)
                B_0_array.append(B_0)
                B_0_conductor_array.append(B_0_conductor)
                a_0_abs_array.append(a_0_abs)
                a_0_FLR_array.append(a_0_FLR)
        except Exception as e:
            continue

    if len(P_NBI_array) == 0:
        return None, None, None, None, None, None, None, None, None, None

    # Find minimum P_NBI
    min_idx = np.argmin(P_NBI_array)

    R_M_vacuum_optimal = R_M_valid[min_idx]
    P_NBI_min = P_NBI_array[min_idx]
    P_fusion_opt = P_fusion_array[min_idx]
    n_20_opt = n_20_array[min_idx]
    a_0_opt = a_0_array[min_idx]
    B_0_opt = B_0_array[min_idx]
    B_0_conductor_opt = B_0_conductor_array[min_idx]

    # Calculate diamagnetic mirror ratio
    R_M_diamagnetic_optimal = B_max / B_0_opt

    # Determine regime
    a_0_abs_opt = a_0_abs_array[min_idx]
    a_0_FLR_opt = a_0_FLR_array[min_idx]
    regime = 'Absorption' if a_0_abs_opt > a_0_FLR_opt else 'FLR'

    # Calculate a_0_FLR at mirror (B_max)
    a_0_FLR_mirror_opt = calculate_a0_FLR_at_mirror(E_b_100keV, B_max, N_25)

    return (R_M_vacuum_optimal, R_M_diamagnetic_optimal, P_NBI_min, P_fusion_opt,
            n_20_opt, a_0_opt, B_0_opt, B_0_conductor_opt, regime, a_0_FLR_mirror_opt)


def plot_PNBI_vs_Eb_optimization(B_max=B_max_default):
    """Create 4 subplots: P_NBI, R_M, a0, and Q"""

    print(f"\n{'='*70}")
    print(f"OPTIMIZATION ANALYSIS: Finding minimum P_NBI for each (E_b, NWL)")
    print(f"B_max = {B_max} T, β_c = {beta_c_default}")
    print(f"Scanning E_b from {E_b_range[0]:.0f} to {E_b_range[-1]:.0f} keV")
    print(f"Scanning R_M from {R_M_scan_range[0]:.0f} to {R_M_scan_range[-1]:.0f}")
    print(f"{'='*70}\n")

    # Create figure with 2x2 grid
    fig = plt.figure(figsize=(16, 12))
    ax1 = plt.subplot(2, 2, 1)  # Top left: P_NBI
    ax2 = plt.subplot(2, 2, 2)  # Top right: R_M
    ax3 = plt.subplot(2, 2, 3)  # Bottom left: a0
    ax4 = plt.subplot(2, 2, 4)  # Bottom right: Q

    colors_nwl = plt.cm.viridis(np.linspace(0.15, 0.95, len(NWL_targets)))

    # Storage for all results
    all_results = {}

    for NWL_target, color in zip(NWL_targets, colors_nwl):
        print(f"\nProcessing NWL = {NWL_target} MW/m²...")

        P_NBI_min_array = []
        P_fusion_opt_array = []
        Q_array = []
        R_M_vacuum_opt_array = []
        R_M_diamagnetic_opt_array = []
        n_20_opt_array = []
        a_0_opt_array = []
        a_0_FLR_mirror_array = []
        B_0_opt_array = []
        B_0_conductor_opt_array = []
        regime_array = []
        reactivity_array = []
        E_b_valid = []

        for E_b_keV in E_b_range:
            result = find_optimal_Rm_for_Eb_NWL(E_b_keV, NWL_target, B_max, beta_c_default)

            if result[0] is not None:
                R_M_vac_opt, R_M_dia_opt, P_NBI_min, P_fusion_opt, n_20_opt, a_0_opt, B_0_opt, B_0_cond_opt, regime, a_0_FLR_mirror = result

                # Calculate Q = P_fusion / P_NBI
                Q = P_fusion_opt / P_NBI_min

                # Calculate reactivity at T_i
                T_i = T_i_coeff * E_b_keV
                reactivity = get_dt_reactivity(T_i)

                P_NBI_min_array.append(P_NBI_min)
                P_fusion_opt_array.append(P_fusion_opt)
                Q_array.append(Q)
                R_M_vacuum_opt_array.append(R_M_vac_opt)
                R_M_diamagnetic_opt_array.append(R_M_dia_opt)
                n_20_opt_array.append(n_20_opt)
                a_0_opt_array.append(a_0_opt)
                a_0_FLR_mirror_array.append(a_0_FLR_mirror)
                B_0_opt_array.append(B_0_opt)
                B_0_conductor_opt_array.append(B_0_cond_opt)
                regime_array.append(regime)
                reactivity_array.append(reactivity)
                E_b_valid.append(E_b_keV)

                print(f"  E_b={E_b_keV:.0f} keV: R_M_vac={R_M_vac_opt:.2f}, "
                      f"P_NBI={P_NBI_min:.1f} MW, Q={Q:.2f} ({regime})")

        # Store results for later use
        all_results[NWL_target] = {
            'E_b': np.array(E_b_valid),
            'P_NBI_min': np.array(P_NBI_min_array),
            'P_fusion_opt': np.array(P_fusion_opt_array),
            'Q': np.array(Q_array),
            'R_M_vacuum_opt': np.array(R_M_vacuum_opt_array),
            'R_M_diamagnetic_opt': np.array(R_M_diamagnetic_opt_array),
            'n_20_opt': np.array(n_20_opt_array),
            'a_0_opt': np.array(a_0_opt_array),
            'a_0_FLR_mirror': np.array(a_0_FLR_mirror_array),
            'B_0_opt': np.array(B_0_opt_array),
            'B_0_conductor_opt': np.array(B_0_conductor_opt_array),
            'reactivity': np.array(reactivity_array),
            'regime': regime_array
        }

        if len(P_NBI_min_array) > 0:
            # TOP LEFT: P_NBI vs E_b
            ax1.plot(E_b_valid, P_NBI_min_array, color=color, linewidth=3.5,
                    label=f'NWL = {NWL_target} MW/m²')

            # TOP RIGHT: R_M vs E_b (plot without label, we'll add custom legend)
            ax2.plot(E_b_valid, R_M_vacuum_opt_array, color=color, linewidth=3.5)

            # BOTTOM LEFT: a0 values (solid = a0_min, dashed = a0_FLR at mirror)
            ax3.plot(E_b_valid, a_0_opt_array, color=color, linewidth=3, linestyle='-',
                    label=f'NWL = {NWL_target} MW/m²')
            ax3.plot(E_b_valid, a_0_FLR_mirror_array, color=color, linewidth=2.5, linestyle='--',
                    alpha=0.7)

            # BOTTOM RIGHT: Q vs E_b
            ax4.plot(E_b_valid, Q_array, color=color, linewidth=3.5,
                    label=f'NWL = {NWL_target} MW/m²')

    # TOP LEFT: P_NBI vs E_b
    ax1.set_xlabel('Beam Energy $E_b$ [keV]', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Minimum $P_{NBI}$ [MW]', fontsize=14, fontweight='bold')
    ax1.set_title('Minimum NBI Power vs Beam Energy', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # TOP RIGHT: R_M vs E_b with B_conductor twin axis
    ax2.set_xlabel('Beam Energy $E_b$ [keV]', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Optimal Vacuum Mirror Ratio $R_M$', fontsize=14, fontweight='bold')
    ax2.set_title('Optimal Mirror Ratio vs Beam Energy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Add twin axis for B_conductor on top right plot
    ax2_twin = ax2.twinx()

    # Plot B_conductor for each NWL (use same colors)
    for NWL_target, color in zip(NWL_targets, colors_nwl):
        data = all_results[NWL_target]
        if len(data['E_b']) > 0:
            ax2_twin.plot(data['E_b'], data['B_0_conductor_opt'], color=color,
                         linewidth=2, linestyle='--', alpha=0.5)

    ax2_twin.set_ylabel('$B_0$ Conductor [T]', fontsize=14, fontweight='bold')
    ax2_twin.tick_params(axis='y')

    # Create custom legend for ax2 with just R_M_vac and B_0_cond
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='gray', linewidth=3, linestyle='-'),
                    Line2D([0], [0], color='gray', linewidth=2, linestyle='--', alpha=0.5)]
    ax2.legend(custom_lines, ['$R_M$ (vacuum)', '$B_0$ (conductor)'],
               loc='best', fontsize=11, framealpha=0.9)

    # BOTTOM LEFT: a0 values (solid = a0_min, dashed = a0_FLR at mirror)
    ax3.set_xlabel('Beam Energy $E_b$ [keV]', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Minor Radius $a_0$ [m]', fontsize=14, fontweight='bold')
    ax3.set_title('Plasma Radius vs Beam Energy', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')

    # Add twin axis for reactivity on bottom left plot
    ax3_twin = ax3.twinx()

    # Plot reactivity for each NWL (use same colors, dotted lines)
    for NWL_target, color in zip(NWL_targets, colors_nwl):
        data = all_results[NWL_target]
        if len(data['E_b']) > 0:
            ax3_twin.plot(data['E_b'], data['reactivity'], color=color,
                         linewidth=2, linestyle=':', alpha=0.8)

    ax3_twin.set_ylabel('Reactivity $\\langle\\sigma v\\rangle$ [m$^3$/s]', fontsize=14, fontweight='bold')
    ax3_twin.tick_params(axis='y')

    # Create custom legend for ax3
    custom_lines_a0 = [Line2D([0], [0], color='gray', linewidth=3, linestyle='-'),
                       Line2D([0], [0], color='gray', linewidth=2.5, linestyle='--', alpha=0.7),
                       Line2D([0], [0], color='gray', linewidth=2, linestyle=':', alpha=0.8)]
    ax3.legend(custom_lines_a0, ['$a_{0,min}$ (solid)', '$a_{0,FLR}$ at $B_{max}$ (dashed)',
                                  'Reactivity $\\langle\\sigma v\\rangle$ (dotted)'],
               loc='upper left', fontsize=11, framealpha=0.9)

    # BOTTOM RIGHT: Q vs E_b with P_fusion on twin axis
    ax4.set_xlabel('Beam Energy $E_b$ [keV]', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Fusion Gain $Q$ ($P_{fusion}/P_{NBI}$)', fontsize=14, fontweight='bold')
    ax4.set_title('Fusion Gain vs Beam Energy', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')

    # Add twin axis for P_fusion on bottom right plot
    ax4_twin = ax4.twinx()

    # Plot P_fusion for each NWL (use same colors, dashed lines)
    for NWL_target, color in zip(NWL_targets, colors_nwl):
        data = all_results[NWL_target]
        if len(data['E_b']) > 0:
            ax4_twin.plot(data['E_b'], data['P_fusion_opt'], color=color,
                         linewidth=2.5, linestyle='--', alpha=0.6)

    ax4_twin.set_ylabel('Fusion Power $P_{fusion}$ [MW]', fontsize=14, fontweight='bold')
    ax4_twin.tick_params(axis='y')

    # Create custom legend for ax4 with Q (solid) and P_fusion (dashed)
    custom_lines_Q = [Line2D([0], [0], color='gray', linewidth=3.5, linestyle='-'),
                      Line2D([0], [0], color='gray', linewidth=2.5, linestyle='--', alpha=0.6)]
    ax4.legend(custom_lines_Q, ['$Q$ (solid)', '$P_{fusion}$ [MW] (dashed)'],
               loc='upper left', fontsize=11, framealpha=0.9)

    # Main title
    fig.suptitle(f'Optimal Operating Points ($B_{{max}}$ = {B_max} T, $E_b$ scanned, $\\beta_c$ = {beta_c_default})',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()

    return fig, all_results


def print_optimization_summary(all_results, B_max=B_max_default):
    """Print a summary table of optimal operating points"""

    print(f"\n{'='*120}")
    print(f"OPTIMIZATION SUMMARY: Optimal Operating Points")
    print(f"B_max = {B_max} T, β_c = {beta_c_default}")
    print(f"{'='*120}")

    for NWL_target in NWL_targets:
        data = all_results[NWL_target]

        if len(data['E_b']) == 0:
            continue

        print(f"\n{'='*120}")
        print(f"NWL = {NWL_target} MW/m²")
        print(f"{'='*120}")
        print(f"{'E_b [keV]':>10} {'R_M_vac':>10} {'R_M_dia':>10} {'P_NBI [MW]':>12} "
              f"{'Q':>8} {'n_20':>8} {'a_0 [m]':>8} {'B_0 [T]':>8} {'B_cond [T]':>10} {'Regime':>12}")
        print(f"{'-'*120}")

        for i in range(len(data['E_b'])):
            print(f"{data['E_b'][i]:>10.1f} {data['R_M_vacuum_opt'][i]:>10.2f} "
                  f"{data['R_M_diamagnetic_opt'][i]:>10.2f} {data['P_NBI_min'][i]:>12.2f} "
                  f"{data['Q'][i]:>8.2f} {data['n_20_opt'][i]:>8.4f} {data['a_0_opt'][i]:>8.3f} "
                  f"{data['B_0_opt'][i]:>8.3f} {data['B_0_conductor_opt'][i]:>10.3f} "
                  f"{data['regime'][i]:>12}")

        # Find overall minimum P_NBI for this NWL
        min_idx = np.argmin(data['P_NBI_min'])
        print(f"\n*** GLOBAL MINIMUM for NWL = {NWL_target} MW/m²:")
        print(f"    E_b = {data['E_b'][min_idx]:.1f} keV")
        print(f"    R_M_vacuum = {data['R_M_vacuum_opt'][min_idx]:.2f}")
        print(f"    R_M_diamagnetic = {data['R_M_diamagnetic_opt'][min_idx]:.2f}")
        print(f"    P_NBI_min = {data['P_NBI_min'][min_idx]:.2f} MW")
        print(f"    Q = {data['Q'][min_idx]:.2f}")
        print(f"    B_0_conductor = {data['B_0_conductor_opt'][min_idx]:.3f} T")
        print(f"    Regime: {data['regime'][min_idx]}")


if __name__ == "__main__":
    print(f"{'='*70}")
    print(f"BEAM ENERGY OPTIMIZATION ANALYSIS")
    print(f"Finding optimal mirror ratios for minimum P_NBI")
    print(f"{'='*70}")

    # Create main optimization plot
    print("\nGenerating optimization analysis plot...")
    fig, all_results = plot_PNBI_vs_Eb_optimization()

    # Save figure
    output_path = figures_dir / 'Optimization_PNBI_vs_Eb.png'
    fig.savefig(output_path, dpi=figure_dpi, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

    # Print summary table
    print_optimization_summary(all_results)

    plt.show()

    print(f"\n{'='*70}")
    print(f"Done!")
    print(f"{'='*70}")
