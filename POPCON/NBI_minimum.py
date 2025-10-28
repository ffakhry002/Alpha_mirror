"""tradespace.py - Mirror ratio (R_M) tradespace analysis with NWL targets
Updated with new beta formulation from corrected equations.py
Modified to show both vacuum and diamagnetic mirror ratios
AXIS FORMAT: Bottom axis shows conductor/plasma, top axis shows vacuum/diamagnetic
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from equations import (
    calculate_loss_coefficient, calculate_beta_local, calculate_B0_with_diamagnetic,
    calculate_beta_limit, calculate_a0_absorption, calculate_a0_FLR,
    calculate_plasma_geometry_frustum, calculate_a0_FLR_at_mirror,
    calculate_fusion_power, calculate_NBI_power, calculate_NWL
)

from n20_Eb_inputs import (
    B_max_default, B_central_default, beta_c_default, T_i_coeff,
    N_25, N_rho, figures_dir, figure_dpi, min_a0
)

# Tradespace parameters
E_b_keV_fixed = 100
R_M_min, R_M_max, R_M_resolution = 4, 10, 200
NWL_targets = [0.75, 1.0, 1.25, 1.5]

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


def test_specific_point(R_M=5.93, E_b_keV=100, target_NWL=1.0, B_max=B_max_default):
    """Test solver at specific point for debugging"""
    print(f"\n{'='*70}\nTESTING POINT: R_M={R_M}, E_b={E_b_keV} keV, NWL={target_NWL}\n{'='*70}")

    result = find_n20_for_target_NWL(E_b_keV, R_M, target_NWL, B_max, beta_c_default)

    if result[0] is None:
        print("*** SOLVER FAILED - target NWL exceeds beta limit ***")
        E_b_100keV = E_b_keV / 100.0
        B_central = B_max / R_M
        n_20_beta_max = calculate_beta_limit(E_b_100keV, B_central, beta_c_default)
        print(f"  Beta-limited density: n_20_max = {n_20_beta_max:.4f}")
        return

    n_20, P_NBI, NWL_achieved, a_0_abs, a_0_FLR, B_0, B_0_conductor = result

    if n_20 is not None:
        E_b_100keV = E_b_keV / 100.0
        B_central = B_max / R_M
        beta_local = calculate_beta_local(n_20, E_b_100keV, B_central)
        a_0_min = max(a_0_abs, a_0_FLR)

        a_0_FLR_mirror = calculate_a0_FLR_at_mirror(E_b_100keV, B_max, N_25)
        L_plasma, V_plasma, vessel_surface_area = calculate_plasma_geometry_frustum(
            a_0_min, a_0_FLR_mirror, N_rho
        )

        print(f"\nResults:")
        print(f"  n_20 = {n_20:.4f}, beta = {beta_local:.6f} ({beta_local/beta_c_default:.3f} of limit)")
        print(f"  B_0 = {B_0:.4f} T (conductor: {B_0_conductor:.4f} T, reduction: {(1-B_0/B_0_conductor)*100:.1f}%)")
        print(f"  a_0_abs = {a_0_abs:.4f} m, a_0_FLR = {a_0_FLR:.4f} m")
        print(f"  a_0_min = {a_0_min:.4f} m ({'Absorption' if a_0_abs > a_0_FLR else 'FLR'} limited)")
        print(f"  L = {L_plasma:.4f} m, V = {V_plasma:.4f} m³")
        print(f"  P_NBI = {P_NBI:.2f} MW, NWL = {NWL_achieved:.4f} MW/m²")
    print(f"{'='*70}\n")


def plot_four_panel_analysis(E_b_keV=E_b_keV_fixed, B_max=B_max_default):
    """Create 4-panel plot: P_NBI, a0 constraints, n_20, and B_0 vs R_M"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    R_M_values = np.linspace(R_M_min, R_M_max, R_M_resolution)
    colors_nwl = plt.cm.viridis(np.linspace(0.15, 0.95, len(NWL_targets)))

    # TOP LEFT: P_NBI vs R_M
    ax1 = axes[0, 0]
    for NWL_target, color in zip(NWL_targets, colors_nwl):
        P_NBI_array, R_M_valid = [], []
        for R_M in R_M_values:
            try:
                result = find_n20_for_target_NWL(E_b_keV, R_M, NWL_target, B_max, beta_c_default)
                if result[1] is not None:
                    # Check minimum a0 constraint
                    a_0_min = max(result[3], result[4])  # max(a_0_abs, a_0_FLR)
                    if a_0_min < min_a0:
                        continue
                    P_NBI_array.append(result[1])
                    R_M_valid.append(R_M)
            except:
                continue

        if len(P_NBI_array) > 0:
            ax1.plot(R_M_valid, P_NBI_array, color=color, linewidth=3.5,
                    label=f'NWL = {NWL_target} MW/m²')

    ax1.set_xlabel('Mirror Ratio $R_M$', fontsize=14, fontweight='bold')
    ax1.set_ylabel('$P_{NBI}$ [MW]', fontsize=14, fontweight='bold')
    ax1.set_title(f'NBI Power vs $R_M$', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=12, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # TOP RIGHT: Minor radius constraints vs R_M
    ax2 = axes[0, 1]
    for NWL_target, color in zip(NWL_targets, colors_nwl):
        a0_abs_array, a0_FLR_array, a0_min_array, R_M_valid = [], [], [], []

        for R_M in R_M_values:
            try:
                result = find_n20_for_target_NWL(E_b_keV, R_M, NWL_target, B_max, beta_c_default)
                n_20, P_NBI, NWL_achieved, a_0_abs, a_0_FLR = result[0], result[1], result[2], result[3], result[4]

                if n_20 is not None:
                    a_0_min = np.maximum(a_0_abs, a_0_FLR)
                    # Check minimum a0 constraint
                    if a_0_min < min_a0:
                        continue
                    a0_abs_array.append(a_0_abs)
                    a0_FLR_array.append(a_0_FLR)
                    a0_min_array.append(a_0_min)
                    R_M_valid.append(R_M)
            except:
                continue

        if len(a0_min_array) > 0:
            ax2.plot(R_M_valid, a0_abs_array, color=color, linewidth=2, linestyle=':', alpha=0.7)
            ax2.plot(R_M_valid, a0_FLR_array, color=color, linewidth=2, linestyle='-.', alpha=0.7)
            ax2.plot(R_M_valid, a0_min_array, color=color, linewidth=3.5, linestyle='-',
                    label=f'NWL = {NWL_target} MW/m²')

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_elements = [
        Line2D([0], [0], color='gray', linewidth=2, linestyle=':',
               label='Absorption: $a_{0,abs} = 0.3\\sqrt{E_b}/n_{20}$'),
        Line2D([0], [0], color='gray', linewidth=2, linestyle='-.',
               label='FLR/Adiabaticity: $a_{0,FLR} = N_{25}\\sqrt{E_b}/B_0$'),
        Line2D([0], [0], color='gray', linewidth=3.5, linestyle='-',
               label='Minimum: $a_{0,min} = \\max(a_{0,abs}, a_{0,FLR})$')
    ]

    first_legend = ax2.legend(handles=legend_elements, loc='upper left', fontsize=11,
                            framealpha=0.9, title='Constraints')
    ax2.add_artist(first_legend)

    nwl_elements = [Patch(facecolor=color, label=f'NWL = {nwl} MW/m²')
                   for nwl, color in zip(NWL_targets, colors_nwl)]
    ax2.legend(handles=nwl_elements, loc='lower right', fontsize=11,
             framealpha=0.9, title='Target NWL')

    ax2.set_xlabel('Mirror Ratio $R_M$', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Minor Radius $a_0$ [m]', fontsize=14, fontweight='bold')
    ax2.set_title(f'Minor Radius Constraints vs $R_M$', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')

    # BOTTOM LEFT: n_20 vs R_M
    ax3 = axes[1, 0]
    for NWL_target, color in zip(NWL_targets, colors_nwl):
        n_20_array, R_M_valid = [], []
        for R_M in R_M_values:
            try:
                result = find_n20_for_target_NWL(E_b_keV, R_M, NWL_target, B_max, beta_c_default)
                if result[0] is not None:
                    # Check minimum a0 constraint
                    a_0_min = max(result[3], result[4])  # max(a_0_abs, a_0_FLR)
                    if a_0_min < min_a0:
                        continue
                    n_20_array.append(result[0])
                    R_M_valid.append(R_M)
            except:
                continue

        if len(n_20_array) > 0:
            ax3.plot(R_M_valid, n_20_array, color=color, linewidth=3.5,
                    label=f'NWL = {NWL_target} MW/m²')

    ax3.set_xlabel('Mirror Ratio $R_M$', fontsize=14, fontweight='bold')
    ax3.set_ylabel('$n_{20}$ [10$^{20}$ m$^{-3}$]', fontsize=14, fontweight='bold')
    ax3.set_title(f'Density vs $R_M$', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=12, framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--')

    # BOTTOM RIGHT: B_0 vs R_M
    ax4 = axes[1, 1]
    R_M_ref = np.linspace(R_M_min, R_M_max, 100)
    ax4.plot(R_M_ref, B_max / R_M_ref, 'k--', linewidth=2, alpha=0.5,
            label='No diamagnetic effects', zorder=1)

    for NWL_target, color in zip(NWL_targets, colors_nwl):
        B_0_array, R_M_valid = [], []
        for R_M in R_M_values:
            try:
                result = find_n20_for_target_NWL(E_b_keV, R_M, NWL_target, B_max, beta_c_default)
                if result[5] is not None:
                    # Check minimum a0 constraint
                    a_0_min = max(result[3], result[4])  # max(a_0_abs, a_0_FLR)
                    if a_0_min < min_a0:
                        continue
                    B_0_array.append(result[5])
                    R_M_valid.append(R_M)
            except:
                continue

        if len(B_0_array) > 0:
            ax4.plot(R_M_valid, B_0_array, color=color, linewidth=3.5,
                    label=f'NWL = {NWL_target} MW/m²', zorder=2)

    ax4.set_xlabel('Mirror Ratio $R_M$', fontsize=14, fontweight='bold')
    ax4.set_ylabel('$B_0$ [T]', fontsize=14, fontweight='bold')
    ax4.set_title(f'On-Axis Field vs $R_M$', fontsize=14, fontweight='bold')
    ax4.legend(loc='best', fontsize=12, framealpha=0.9)
    ax4.grid(True, alpha=0.3, linestyle='--')

    fig.suptitle(f'Mirror Performance Analysis ($B_{{max}}$={B_max}T, $E_b$={E_b_keV} keV, $\\beta_c$={beta_c_default})',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()
    return fig


def plot_PNBI_vs_B0(E_b_keV=E_b_keV_fixed, B_max=B_max_default):
    """Create P_NBI vs B_0 plot with dual axis showing conductor/plasma and vacuum/diamagnetic"""
    fig, ax = plt.subplots(figsize=(12, 8))
    R_M_values = np.linspace(R_M_min, R_M_max, R_M_resolution)
    colors_nwl = plt.cm.viridis(np.linspace(0.15, 0.95, len(NWL_targets)))

    # Collect data for all NWL targets
    data_by_nwl = {}
    for NWL_target in NWL_targets:
        B_0_array, B_0_conductor_array, R_M_array, P_NBI_array = [], [], [], []

        for R_M in R_M_values:
            try:
                result = find_n20_for_target_NWL(E_b_keV, R_M, NWL_target, B_max, beta_c_default)
                n_20, P_NBI, NWL_achieved, a_0_abs, a_0_FLR, B_0, B_0_conductor = result

                if P_NBI is not None and B_0 is not None:
                    # Check minimum a0 constraint
                    a_0_min = max(a_0_abs, a_0_FLR)
                    if a_0_min < min_a0:
                        continue
                    B_0_array.append(B_0)
                    B_0_conductor_array.append(B_0_conductor)
                    R_M_array.append(R_M)
                    P_NBI_array.append(P_NBI)
            except:
                continue

        data_by_nwl[NWL_target] = {
            'B_0': np.array(B_0_array),
            'B_0_conductor': np.array(B_0_conductor_array),
            'R_M': np.array(R_M_array),
            'P_NBI': np.array(P_NBI_array)
        }

    # Plot P_NBI vs B_0
    for i, NWL_target in enumerate(NWL_targets):
        data = data_by_nwl[NWL_target]
        if len(data['B_0']) > 0:
            ax.plot(data['B_0'], data['P_NBI'], color=colors_nwl[i],
                   linewidth=3.5, label=f'NWL = {NWL_target} MW/m²')

    ax.set_ylabel('$P_{NBI}$ [MW]', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Set up tick marks
    first_nwl = NWL_targets[0]
    if len(data_by_nwl[first_nwl]['B_0']) > 0:
        B0_for_interp = data_by_nwl[first_nwl]['B_0']
        Rm_for_interp = data_by_nwl[first_nwl]['R_M']
        B0_conductor_for_interp = data_by_nwl[first_nwl]['B_0_conductor']

        sort_idx = np.argsort(B0_for_interp)
        B0_for_interp = B0_for_interp[sort_idx]
        Rm_for_interp = Rm_for_interp[sort_idx]
        B0_conductor_for_interp = B0_conductor_for_interp[sort_idx]

        tick_spacing = 0.5
        B0_tick_values = np.arange(
            np.ceil(np.min(B0_for_interp) / tick_spacing) * tick_spacing,
            np.floor(np.max(B0_for_interp) / tick_spacing) * tick_spacing + tick_spacing/2,
            tick_spacing
        )

        Rm_at_ticks = np.interp(B0_tick_values, B0_for_interp, Rm_for_interp)
        B0_conductor_at_ticks = np.interp(B0_tick_values, B0_for_interp, B0_conductor_for_interp)

        R_M_vacuum = B_max / B0_conductor_at_ticks
        R_M_diamagnetic = B_max / B0_tick_values

        ax.set_xticks(B0_tick_values)
        ax.set_xticklabels([f'{b0_cond:.1f} / {b0_plasma:.1f}'
                            for b0_plasma, b0_cond in zip(B0_tick_values, B0_conductor_at_ticks)],
                           fontsize=9, rotation=45, ha='right')
        ax.set_xlabel('$B_0$ [T]  (format: conductor / plasma)', fontsize=13, fontweight='bold')

        # Top axis for mirror ratios
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(B0_tick_values)
        ax_top.set_xticklabels([f'{rm_vac:.2f} / {rm_dia:.2f}'
                                for rm_dia, rm_vac in zip(R_M_diamagnetic, R_M_vacuum)],
                               fontsize=9, rotation=45, ha='left')
        ax_top.set_xlabel('Mirror Ratio $R_M$ (format: vacuum / diamagnetic)',
                         fontsize=13, fontweight='bold')

    ax.set_title(f'NBI Power vs On-Axis Field\n$B_{{max}}$ = {B_max} T, $\\beta_c$ = {beta_c_default}, $E_b$ = {E_b_keV} keV',
                 fontsize=15, fontweight='bold', pad=40)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, top=0.85)
    return fig


def find_minimum_PNBI_points(E_b_keV=E_b_keV_fixed, B_max=B_max_default):
    """Find and print minimum P_NBI for each NWL target"""
    E_b_100keV = E_b_keV / 100.0
    R_M_search = np.linspace(R_M_min, R_M_max, R_M_resolution)

    print(f"\n{'='*70}\nMINIMUM P_NBI OPERATING POINTS:\n{'='*70}")

    for NWL_target in NWL_targets:
        print(f"\n{'='*70}\nTarget NWL = {NWL_target} MW/m²:\n{'='*70}")

        P_NBI_array, R_M_valid, n_20_array = [], [], []
        a_0_array, L_array, beta_array, B_0_array = [], [], [], []

        for R_M in R_M_search:
            try:
                result = find_n20_for_target_NWL(E_b_keV, R_M, NWL_target, B_max, beta_c_default)
                n_20, P_NBI, NWL_achieved, a_0_abs, a_0_FLR, B_0, B_0_conductor = result

                if P_NBI is not None:
                    a_0_min = max(a_0_abs, a_0_FLR)
                    # Check minimum a0 constraint
                    if a_0_min < min_a0:
                        continue

                    P_NBI_array.append(P_NBI)
                    R_M_valid.append(R_M)
                    n_20_array.append(n_20)
                    B_0_array.append(B_0)

                    a_0_FLR_mirror = calculate_a0_FLR_at_mirror(E_b_100keV, B_max, N_25)
                    L_plasma, V_plasma, _ = calculate_plasma_geometry_frustum(a_0_min, a_0_FLR_mirror, N_rho)

                    a_0_array.append(a_0_min)
                    L_array.append(L_plasma)

                    B_central = B_max / R_M
                    beta = calculate_beta_local(n_20, E_b_100keV, B_central)
                    beta_array.append(beta)
            except:
                continue

        if len(P_NBI_array) > 0:
            min_idx = np.argmin(P_NBI_array)
            print(f"  Minimum P_NBI = {P_NBI_array[min_idx]:.2f} MW at R_M = {R_M_valid[min_idx]:.2f}")
            print(f"  n_20 = {n_20_array[min_idx]:.4f}, Beta = {beta_array[min_idx]:.4f} ({beta_array[min_idx]/beta_c_default*100:.1f}% of limit)")
            print(f"  B_0 = {B_0_array[min_idx]:.3f} T, a₀ = {a_0_array[min_idx]:.3f} m, L = {L_array[min_idx]:.3f} m")
            print(f"  Volume = {np.pi * a_0_array[min_idx]**2 * L_array[min_idx]:.3f} m³")
        else:
            print(f"  No valid operating points found")


if __name__ == "__main__":
    print(f"{'='*70}\nMIRROR RATIO TRADESPACE ANALYSIS\n{'='*70}")

    test_specific_point()

    print("\nGenerating four-panel tradespace plot...")
    fig1 = plot_four_panel_analysis()
    output_path1 = figures_dir / 'Mirror_Tradespace_Analysis_NEW.png'
    fig1.savefig(output_path1, dpi=figure_dpi, bbox_inches='tight')
    print(f"Saved: {output_path1}")

    print("\nGenerating P_NBI vs B_0 plot...")
    fig2 = plot_PNBI_vs_B0()
    output_path2 = figures_dir / 'PNBI_vs_B0_NEW.png'
    fig2.savefig(output_path2, dpi=figure_dpi, bbox_inches='tight')
    print(f"Saved: {output_path2}")

    find_minimum_PNBI_points()

    print(f"\n{'='*70}\nExample calculations:\n{'='*70}")
    for NWL_target in NWL_targets:
        print(f"\nNWL = {NWL_target} MW/m²:")
        for R_M in [5, 7, 9]:
            result = find_n20_for_target_NWL(E_b_keV_fixed, R_M, NWL_target, B_max_default, beta_c_default)
            n_20, P_NBI, NWL_achieved, a_0_abs, a_0_FLR = result[0], result[1], result[2], result[3], result[4]

            if P_NBI is not None:
                a_0_min = max(a_0_abs, a_0_FLR)
                if a_0_min < min_a0:
                    print(f"  R_M={R_M}: Below minimum a0 threshold ({a_0_min:.3f} < {min_a0:.2f} m)")
                    continue
                constraint = "Absorption" if a_0_abs > a_0_FLR else "FLR"
                print(f"  R_M={R_M}: P_NBI={P_NBI:.1f} MW, a₀={a_0_min:.3f} m ({constraint})")
            else:
                print(f"  R_M={R_M}: Not achievable (exceeds beta limit)")

    plt.show()
    print(f"\n{'='*70}\nDone!\n{'='*70}")
