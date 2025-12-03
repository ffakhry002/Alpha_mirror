"""Rm_optimization.py - Find optimal mirror ratio for maximum ⟨P_fus⟩ at given PNBI

For a fixed beam energy, this program:
1. Scans through different vacuum mirror ratios (Rm_vac)
2. For each Rm, finds points along a constant PNBI contour in the n20 space (at fixed Eb)
3. Calculates the maximum capacity factor adjusted fusion power achievable
4. Handles both upper and lower density branches where PNBI contours intersect

Output: ⟨P_fus⟩ vs Rm_vac for specified PNBI levels (e.g., 30MW, 50MW)
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
    calculate_a0_DCLC,
    calculate_a0_adiabaticity,
    calculate_a0_end,
    calculate_plasma_geometry_frustum,
    calculate_fusion_power,
    calculate_NBI_power,
    calculate_Q,
    calculate_Bw,
    calculate_heat_flux,
    calculate_voltage_closed_lines,
    calculate_voltage_field_reversal,
    calculate_max_mirror_ratio_vortex,
    calculate_grid_lifetime,
    calculate_average_fusion_power,
    calculate_isotope_revenue,
    calculate_revenue_per_volume,
)

from n20_Eb_inputs import (
    B_max_default,
    beta_c_default,
    T_i_coeff,
    min_a0,
    qw_limit,
    figures_dir,
    figure_dpi,
    d_grid,
    t_replace,
    eta_duty,
    sigma_x_beam,
    sigma_y_beam,
    num_grids,
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

    # Calculate vacuum mirror ratio and diamagnetic mirror ratio
    R_M_vac = B_max / B_central
    R_M_dmag = B_max / B_0

    # Calculate geometry constraints
    a_0_abs = calculate_a0_absorption(E_b_100keV, n_20)
    a_0_DCLC = calculate_a0_DCLC(E_b_100keV, B_0)
    a_0_adiabatic = calculate_a0_adiabaticity(E_b_100keV, B_0, beta_local)
    a_0_min = max(a_0_abs, a_0_DCLC, a_0_adiabatic)

    # Minimum radius check
    if a_0_min < min_a0:
        return False, "min_radius"

    # Calculate geometry with new frustum model
    a_0_end = calculate_a0_end(a_0_min, B_0, B_max)
    L_plasma, V_plasma, vessel_surface_area = calculate_plasma_geometry_frustum(
        a_0_min, a_0_end, E_b_100keV, B_0
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

    Returns: list of tuples with:
        (n_20, P_fusion_avg, V_plasma, Revenue, Rev_per_Vol, a_0_min, a_0_end, L_plasma, feasible, reason)
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

        # Calculate geometry with new model
        a_0_abs = calculate_a0_absorption(E_b_100keV, n_20)
        a_0_DCLC = calculate_a0_DCLC(E_b_100keV, B_0)
        a_0_adiabatic = calculate_a0_adiabaticity(E_b_100keV, B_0, beta_local)
        a_0_min = max(a_0_abs, a_0_DCLC, a_0_adiabatic)

        a_0_end = calculate_a0_end(a_0_min, B_0, B_max)
        L_plasma, V_plasma, vessel_surface_area = calculate_plasma_geometry_frustum(
            a_0_min, a_0_end, E_b_100keV, B_0
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

                # Calculate P_fusion and capacity factor at this point
                beta_local = calculate_beta_local(n_20_cross, E_b_100keV, B_central)
                B_0 = calculate_B0_with_diamagnetic(B_central, beta_local)

                a_0_abs = calculate_a0_absorption(E_b_100keV, n_20_cross)
                a_0_DCLC = calculate_a0_DCLC(E_b_100keV, B_0)
                a_0_adiabatic = calculate_a0_adiabaticity(E_b_100keV, B_0, beta_local)
                a_0_min = max(a_0_abs, a_0_DCLC, a_0_adiabatic)

                a_0_end = calculate_a0_end(a_0_min, B_0, B_max)
                L_plasma, V_plasma, vessel_surface_area = calculate_plasma_geometry_frustum(
                    a_0_min, a_0_end, E_b_100keV, B_0
                )

                T_i = T_i_coeff * E_b_100keV * 100
                P_fusion = calculate_fusion_power(E_b_100keV, n_20_cross, V_plasma, T_i)

                # Calculate capacity factor adjusted fusion power
                t_grid = calculate_grid_lifetime(
                    E_b_100keV * 100,  # Convert to keV
                    P_NBI_target,
                    d_mm=d_grid,
                    sigma_x_cm=sigma_x_beam,
                    sigma_y_cm=sigma_y_beam,
                    num_grids=num_grids
                )
                P_fusion_avg = calculate_average_fusion_power(
                    P_fusion, t_grid,
                    t_replace_months=t_replace,
                    eta_duty=eta_duty
                )

                # Calculate Revenue/Volume using capacity factor adjusted fusion power
                Revenue = calculate_isotope_revenue(P_fusion_avg)
                Rev_per_Vol = Revenue / V_plasma

                # Check feasibility
                feasible, reason = check_feasibility(E_b_100keV, n_20_cross, B_central, B_max)

                crossings.append((n_20_cross, P_fusion_avg, V_plasma, Revenue, Rev_per_Vol,
                                  a_0_min, a_0_end, L_plasma, feasible, reason))
            except:
                continue

    return crossings


def scan_Rm_for_PNBI(E_b_100keV, P_NBI_target, Rm_range, n_Rm_points=50):
    """
    Scan through different mirror ratios and find maximum Revenue/Volume for given PNBI

    For each Rm, finds all points where PNBI contour intersects (typically 2 branches),
    and returns the maximum feasible Revenue/Volume (optimized for economic figure of merit).

    Returns:
        Rm_vac_array: array of mirror ratios
        Rev_per_Vol_max_array: maximum Revenue/Volume at each Rm [$/yr/m³]
        P_fusion_max_array: capacity factor adjusted fusion power at optimal Rev/Vol point
        n_20_opt_array: optimal density at each Rm
        Revenue_opt_array: total revenue at each Rm [$/yr]
        V_plasma_opt_array: plasma volume at each Rm [m³]
        a0_min_opt_array: center radius at each Rm [m]
        a0_end_opt_array: end radius at each Rm [m]
        L_plasma_opt_array: plasma length at each Rm [m]
    """
    Rm_vac_array = np.linspace(Rm_range[0], Rm_range[1], n_Rm_points)
    Rev_per_Vol_max_array = []
    P_fusion_max_array = []
    n_20_opt_array = []
    Revenue_opt_array = []
    V_plasma_opt_array = []
    a0_min_opt_array = []
    a0_end_opt_array = []
    L_plasma_opt_array = []

    found_solution = False
    consecutive_failures = 0

    for Rm_vac in Rm_vac_array:
        B_central = B_max_default / Rm_vac

        # Calculate beta limit for this configuration
        n_20_max = calculate_beta_limit(E_b_100keV, B_central, beta_c_default)

        # Find intersections with PNBI contour
        crossings = find_n20_for_PNBI(
            E_b_100keV, P_NBI_target, B_central, B_max_default,
            n_20_search_range=(0.01, n_20_max)
        )

        # Find maximum feasible Revenue/Volume
        max_Rev_per_Vol = 0
        opt_P_fusion = 0
        opt_n20 = None
        opt_Revenue = 0
        opt_V_plasma = 0
        opt_a0_min = 0
        opt_a0_end = 0
        opt_L_plasma = 0

        for n_20, P_fusion, V_plasma, Revenue, Rev_per_Vol, a0_min, a0_end, L_plasma, feasible, reason in crossings:
            if feasible and Rev_per_Vol > max_Rev_per_Vol:
                max_Rev_per_Vol = Rev_per_Vol
                opt_P_fusion = P_fusion
                opt_n20 = n_20
                opt_Revenue = Revenue
                opt_V_plasma = V_plasma
                opt_a0_min = a0_min
                opt_a0_end = a0_end
                opt_L_plasma = L_plasma

        Rev_per_Vol_max_array.append(max_Rev_per_Vol if max_Rev_per_Vol > 0 else np.nan)
        P_fusion_max_array.append(opt_P_fusion if opt_P_fusion > 0 else np.nan)
        n_20_opt_array.append(opt_n20 if opt_n20 is not None else np.nan)
        Revenue_opt_array.append(opt_Revenue if opt_Revenue > 0 else np.nan)
        V_plasma_opt_array.append(opt_V_plasma if opt_V_plasma > 0 else np.nan)
        a0_min_opt_array.append(opt_a0_min if opt_a0_min > 0 else np.nan)
        a0_end_opt_array.append(opt_a0_end if opt_a0_end > 0 else np.nan)
        L_plasma_opt_array.append(opt_L_plasma if opt_L_plasma > 0 else np.nan)

        # Early exit logic: stop if we've found solutions and then hit consecutive failures
        if max_Rev_per_Vol > 0:
            found_solution = True
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if found_solution and consecutive_failures >= 3:
                # Pad remaining arrays with nan
                remaining = n_Rm_points - len(Rev_per_Vol_max_array)
                Rev_per_Vol_max_array.extend([np.nan] * remaining)
                P_fusion_max_array.extend([np.nan] * remaining)
                n_20_opt_array.extend([np.nan] * remaining)
                Revenue_opt_array.extend([np.nan] * remaining)
                V_plasma_opt_array.extend([np.nan] * remaining)
                a0_min_opt_array.extend([np.nan] * remaining)
                a0_end_opt_array.extend([np.nan] * remaining)
                L_plasma_opt_array.extend([np.nan] * remaining)
                break

    return (Rm_vac_array, np.array(Rev_per_Vol_max_array), np.array(P_fusion_max_array),
            np.array(n_20_opt_array), np.array(Revenue_opt_array), np.array(V_plasma_opt_array),
            np.array(a0_min_opt_array), np.array(a0_end_opt_array), np.array(L_plasma_opt_array))


def plot_Pfusion_vs_Rm(E_b_100keV, P_NBI_list, Rm_range=(4, 16), n_Rm_points=50):
    """
    Create plot of Revenue/Volume vs Rm for different PNBI levels at a fixed beam energy

    Note: Optimizes for Revenue/Volume using capacity factor adjusted fusion power.

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

        (Rm_array, Rev_per_Vol_array, P_fusion_array, n20_array,
         Revenue_array, V_plasma_array, a0_min_array, a0_end_array, L_plasma_array) = scan_Rm_for_PNBI(
            E_b_100keV, P_NBI_target, Rm_range, n_Rm_points
        )

        # Plot only where Revenue/Volume is valid
        valid_mask = ~np.isnan(Rev_per_Vol_array) & (Rev_per_Vol_array > 0)

        if np.any(valid_mask):
            ax.plot(Rm_array[valid_mask], Rev_per_Vol_array[valid_mask] / 1e6,  # Convert to $M/yr/m³
                   linewidth=2.5, color=colors[P_NBI_idx],
                   label=f'$P_{{NBI}}$ = {P_NBI_target} MW', marker='o', markersize=3)

            # Find and report maximum
            max_idx = np.nanargmax(Rev_per_Vol_array)
            print(f"  Max Rev/Vol = ${Rev_per_Vol_array[max_idx]/1e6:.2f}M/yr/m³ at Rm = {Rm_array[max_idx]:.2f}")
            print(f"  Revenue = ${Revenue_array[max_idx]/1e6:.2f}M/yr")
            print(f"  ⟨P_fus⟩ = {P_fusion_array[max_idx]:.2f} MW, n_20 = {n20_array[max_idx]:.3f}")
            print(f"  V = {V_plasma_array[max_idx]:.3f} m³, a0_center = {a0_min_array[max_idx]:.3f} m, a0_end = {a0_end_array[max_idx]:.3f} m, L = {L_plasma_array[max_idx]:.3f} m")
        else:
            print(f"  No feasible points found!")

    ax.set_xlabel(r'$R_{M,vac}$ (Vacuum Mirror Ratio)', fontsize=16)
    ax.set_ylabel(r'Revenue/Volume [$M/yr/m³]', fontsize=16)
    ax.set_title(f'Maximum Revenue/Volume vs Mirror Ratio for Fixed $P_{{NBI}}$ Contours\n'
                 f'($E_b$ = {E_b_keV:.0f} keV, $B_{{max}}$ = {B_max_default} T)\n'
                 f'(Grid Lifetime: {eta_duty*100:.0f}% duty cycle, {t_replace} mo replacement)',
                 fontsize=16, weight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    ax.legend(fontsize=14, loc='best', framealpha=0.9)
    ax.tick_params(labelsize=13)

    plt.tight_layout()

    return fig


def find_optimal_Rm_for_Eb_PNBI(E_b_100keV, P_NBI_target, Rm_range, n_Rm_points=50):
    """
    For a given (Eb, PNBI), find the optimal Rm that maximizes Revenue/Volume

    Returns:
        Rm_optimal, Rev_per_Vol_max, P_fusion_opt, n_20_opt,
        Revenue_opt, V_plasma_opt, a0_min_opt, a0_end_opt, L_plasma_opt
    """
    (Rm_array, Rev_per_Vol_array, P_fusion_array, n20_array,
     Revenue_array, V_plasma_array, a0_min_array, a0_end_array, L_plasma_array) = scan_Rm_for_PNBI(
        E_b_100keV, P_NBI_target, Rm_range, n_Rm_points
    )

    valid_mask = ~np.isnan(Rev_per_Vol_array) & (Rev_per_Vol_array > 0)

    if np.any(valid_mask):
        max_idx = np.nanargmax(Rev_per_Vol_array)
        return (Rm_array[max_idx], Rev_per_Vol_array[max_idx], P_fusion_array[max_idx], n20_array[max_idx],
                Revenue_array[max_idx], V_plasma_array[max_idx], a0_min_array[max_idx],
                a0_end_array[max_idx], L_plasma_array[max_idx])
    else:
        return None, None, None, None, None, None, None, None, None


def find_optimal_Rm_for_Eb_PNBI_Pfus(E_b_100keV, P_NBI_target, Rm_range, n_Rm_points=50):
    """
    For a given (Eb, PNBI), find the optimal Rm that maximizes ⟨P_fus⟩

    Returns:
        Rm_optimal, P_fusion_max, n_20_opt, Revenue_opt, V_plasma_opt, a0_min_opt, a0_end_opt, L_plasma_opt
    """
    (Rm_array, Rev_per_Vol_array, P_fusion_array, n20_array,
     Revenue_array, V_plasma_array, a0_min_array, a0_end_array, L_plasma_array) = scan_Rm_for_PNBI(
        E_b_100keV, P_NBI_target, Rm_range, n_Rm_points
    )

    valid_mask = ~np.isnan(P_fusion_array) & (P_fusion_array > 0)

    if np.any(valid_mask):
        max_idx = np.nanargmax(P_fusion_array)  # Optimize for P_fus instead of Rev/Vol
        return (Rm_array[max_idx], P_fusion_array[max_idx], n20_array[max_idx],
                Revenue_array[max_idx], V_plasma_array[max_idx], a0_min_array[max_idx],
                a0_end_array[max_idx], L_plasma_array[max_idx])
    else:
        return None, None, None, None, None, None, None, None


def plot_Eb_optimization_for_PNBI_Pfus(P_NBI_list, Eb_range=(0.4, 1.2), Rm_range=(4, 16), n_Eb_points=25, n_Rm_points=50):
    """
    Create 2-panel figure showing optimal operating points vs beam energy (optimizing for ⟨P_fus⟩)

    Left: ⟨P_fus⟩ vs Eb (at optimal Rm for each point)
    Right: Optimal Rm vs Eb

    Args:
        P_NBI_list: list of PNBI values [MW]
        Eb_range: (min, max) beam energy range [100 keV units]
        Rm_range: (min, max) mirror ratio range for optimization
        n_Eb_points: number of beam energy points to scan
        n_Rm_points: number of Rm points for optimization at each Eb
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(P_NBI_list)))

    all_results = {}

    for P_NBI_idx, P_NBI_target in enumerate(P_NBI_list):
        print(f"\n{'='*80}")
        print(f"Optimizing Rm for each Eb at P_NBI = {P_NBI_target} MW (maximizing ⟨P_fus⟩)...")
        print(f"{'='*80}")

        Eb_array = np.linspace(Eb_range[0], Eb_range[1], n_Eb_points)
        P_fusion_opt_array = []
        Rm_opt_array = []
        n20_opt_array = []
        Q_opt_array = []
        Revenue_opt_array = []
        V_plasma_opt_array = []
        a0_min_opt_array = []
        a0_end_opt_array = []
        L_plasma_opt_array = []
        Eb_valid = []

        found_solution = False
        consecutive_failures = 0

        for E_b_100keV in Eb_array:
            E_b_keV = E_b_100keV * 100

            # Find optimal Rm for this (Eb, PNBI) - optimizes for ⟨P_fus⟩
            result = find_optimal_Rm_for_Eb_PNBI_Pfus(
                E_b_100keV, P_NBI_target, Rm_range, n_Rm_points
            )
            Rm_opt, P_fusion_max, n20_opt, Revenue_opt, V_plasma_opt, a0_min_opt, a0_end_opt, L_plasma_opt = result

            if Rm_opt is not None:
                found_solution = True
                consecutive_failures = 0
                Q_opt = P_fusion_max / P_NBI_target

                P_fusion_opt_array.append(P_fusion_max)
                Rm_opt_array.append(Rm_opt)
                n20_opt_array.append(n20_opt)
                Q_opt_array.append(Q_opt)
                Revenue_opt_array.append(Revenue_opt)
                V_plasma_opt_array.append(V_plasma_opt)
                a0_min_opt_array.append(a0_min_opt)
                a0_end_opt_array.append(a0_end_opt)
                L_plasma_opt_array.append(L_plasma_opt)
                Eb_valid.append(E_b_keV)

                print(f"  E_b = {E_b_keV:.1f} keV: Rm = {Rm_opt:.2f}, ⟨Pfus⟩ = {P_fusion_max:.2f} MW, "
                      f"Rev = ${Revenue_opt/1e6:.1f}M, V = {V_plasma_opt:.3f} m³, "
                      f"a0 = {a0_min_opt:.3f} m, a_end = {a0_end_opt:.3f} m, L = {L_plasma_opt:.2f} m")
            else:
                consecutive_failures += 1
                if found_solution and consecutive_failures >= 3:
                    print(f"  E_b = {E_b_keV:.1f} keV: No solution found. Stopping search for this P_NBI.")
                    break

        # Store results
        all_results[P_NBI_target] = {
            'Eb': np.array(Eb_valid),
            'P_fusion_opt': np.array(P_fusion_opt_array),
            'Rm_opt': np.array(Rm_opt_array),
            'n20_opt': np.array(n20_opt_array),
            'Q_opt': np.array(Q_opt_array),
            'Revenue_opt': np.array(Revenue_opt_array),
            'V_plasma_opt': np.array(V_plasma_opt_array),
            'a0_min_opt': np.array(a0_min_opt_array),
            'a0_end_opt': np.array(a0_end_opt_array),
            'L_plasma_opt': np.array(L_plasma_opt_array),
        }

        if len(Eb_valid) > 0:
            # LEFT: ⟨P_fus⟩ vs Eb
            ax1.plot(Eb_valid, P_fusion_opt_array, color=colors[P_NBI_idx],
                    linewidth=1.5, marker='o', markersize=2,
                    label=f'$P_{{NBI}}$ = {P_NBI_target} MW')

            # RIGHT: Optimal Rm vs Eb
            ax2.plot(Eb_valid, Rm_opt_array, color=colors[P_NBI_idx],
                    linewidth=1.5, marker='o', markersize=2,
                    label=f'$P_{{NBI}}$ = {P_NBI_target} MW')

    # LEFT subplot formatting
    ax1.set_xlabel(r'Beam Energy $E_{NBI}$ [keV]', fontsize=13, fontweight='bold')
    ax1.set_ylabel(r'$\langle P_{fus} \rangle$ [MW]', fontsize=13, fontweight='bold')
    ax1.set_title(r'Capacity Factor Adjusted Fusion Power vs Beam Energy' + '\n(at optimal $R_M$ for each point)',
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    ax1.tick_params(labelsize=11)

    # RIGHT subplot formatting
    ax2.set_xlabel(r'Beam Energy $E_{NBI}$ [keV]', fontsize=13, fontweight='bold')
    ax2.set_ylabel(r'Optimal $R_{M,vac}$ (Vacuum Mirror Ratio)', fontsize=13, fontweight='bold')
    ax2.set_title('Optimal Mirror Ratio vs Beam Energy\n' + r'(for maximum $\langle P_{fus} \rangle$ at fixed $P_{NBI}$)',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    ax2.tick_params(labelsize=11)

    # Main title
    fig.suptitle(f'Optimal Operating Points ($\\langle P_{{fus}} \\rangle$) for Fixed $P_{{NBI}}$ Contours ($B_{{max}}$ = {B_max_default} T)',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()

    return fig, all_results


def plot_Eb_optimization_for_PNBI(P_NBI_list, Eb_range=(0.4, 1.2), Rm_range=(4, 16), n_Eb_points=25, n_Rm_points=50):
    """
    Create 3-subplot figure showing optimal operating points vs beam energy

    For each PNBI level, scans through Eb and finds optimal Rm that maximizes Revenue/Volume

    Left: Revenue/Volume vs Eb (at optimal Rm for each point)
    Middle: Optimal Rm vs Eb
    Bottom-left: ⟨P_fus⟩ vs Eb
    Bottom-right: Revenue and Volume (dual y-axis) vs Eb

    Args:
        P_NBI_list: list of PNBI values [MW]
        Eb_range: (min, max) beam energy range [100 keV units]
        Rm_range: (min, max) mirror ratio range for optimization
        n_Eb_points: number of beam energy points to scan
        n_Rm_points: number of Rm points for optimization at each Eb
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    ax4_twin = ax4.twinx()  # Create secondary y-axis for Volume

    colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(P_NBI_list)))

    all_results = {}

    for P_NBI_idx, P_NBI_target in enumerate(P_NBI_list):
        print(f"\n{'='*80}")
        print(f"Optimizing Rm for each Eb at P_NBI = {P_NBI_target} MW...")
        print(f"{'='*80}")

        Eb_array = np.linspace(Eb_range[0], Eb_range[1], n_Eb_points)
        Rev_per_Vol_opt_array = []
        P_fusion_opt_array = []
        Rm_opt_array = []
        n20_opt_array = []
        Q_opt_array = []
        Revenue_opt_array = []
        V_plasma_opt_array = []
        a0_min_opt_array = []
        a0_end_opt_array = []
        L_plasma_opt_array = []
        voltage_cl_array = []
        voltage_fr_array = []
        Eb_valid = []

        found_solution = False  # Track if we've found any solutions
        consecutive_failures = 0  # Count consecutive failures

        for E_b_100keV in Eb_array:
            E_b_keV = E_b_100keV * 100

            # Find optimal Rm for this (Eb, PNBI) - now optimizes for Revenue/Volume
            result = find_optimal_Rm_for_Eb_PNBI(
                E_b_100keV, P_NBI_target, Rm_range, n_Rm_points
            )
            Rm_opt, Rev_per_Vol_max, P_fusion_opt, n20_opt, Revenue_opt, V_plasma_opt, a0_min_opt, a0_end_opt, L_plasma_opt = result

            if Rm_opt is not None:
                found_solution = True
                consecutive_failures = 0  # Reset failure counter
                Q_opt = P_fusion_opt / P_NBI_target

                # Calculate voltage requirements at this optimal point
                B_central = B_max_default / Rm_opt
                beta_local = calculate_beta_local(n20_opt, E_b_100keV, B_central)
                B_0 = calculate_B0_with_diamagnetic(B_central, beta_local)

                # Diamagnetic mirror ratio
                R_M_dmag = B_max_default / B_0

                # Calculate voltages
                voltage_cl = calculate_voltage_closed_lines(E_b_100keV, B_0, a0_min_opt, L_plasma_opt, R_M_dmag)
                voltage_fr = calculate_voltage_field_reversal(E_b_100keV, B_0, a0_min_opt, L_plasma_opt, R_M_dmag)

                Rev_per_Vol_opt_array.append(Rev_per_Vol_max)
                P_fusion_opt_array.append(P_fusion_opt)
                Rm_opt_array.append(Rm_opt)
                n20_opt_array.append(n20_opt)
                Q_opt_array.append(Q_opt)
                Revenue_opt_array.append(Revenue_opt)
                V_plasma_opt_array.append(V_plasma_opt)
                a0_min_opt_array.append(a0_min_opt)
                a0_end_opt_array.append(a0_end_opt)
                L_plasma_opt_array.append(L_plasma_opt)
                voltage_cl_array.append(voltage_cl)
                voltage_fr_array.append(voltage_fr)
                Eb_valid.append(E_b_keV)

                print(f"  E_b = {E_b_keV:.1f} keV: Rm = {Rm_opt:.2f}, Rev/Vol = ${Rev_per_Vol_max/1e6:.1f}M, "
                      f"Rev = ${Revenue_opt/1e6:.1f}M, ⟨Pfus⟩ = {P_fusion_opt:.2f} MW, "
                      f"V = {V_plasma_opt:.3f} m³, a0 = {a0_min_opt:.3f} m, a_end = {a0_end_opt:.3f} m, L = {L_plasma_opt:.2f} m")
            else:
                consecutive_failures += 1
                # If we've found solutions before and now have 3+ consecutive failures, stop searching
                if found_solution and consecutive_failures >= 3:
                    print(f"  E_b = {E_b_keV:.1f} keV: No solution found. Stopping search for this P_NBI.")
                    break

        # Store results
        all_results[P_NBI_target] = {
            'Eb': np.array(Eb_valid),
            'Rev_per_Vol_opt': np.array(Rev_per_Vol_opt_array),
            'P_fusion_opt': np.array(P_fusion_opt_array),
            'Rm_opt': np.array(Rm_opt_array),
            'n20_opt': np.array(n20_opt_array),
            'Q_opt': np.array(Q_opt_array),
            'Revenue_opt': np.array(Revenue_opt_array),
            'V_plasma_opt': np.array(V_plasma_opt_array),
            'a0_min_opt': np.array(a0_min_opt_array),
            'a0_end_opt': np.array(a0_end_opt_array),
            'L_plasma_opt': np.array(L_plasma_opt_array),
            'voltage_cl': np.array(voltage_cl_array),
            'voltage_fr': np.array(voltage_fr_array)
        }

        if len(Eb_valid) > 0:
            # TOP-LEFT: Revenue/Volume vs Eb
            ax1.plot(Eb_valid, np.array(Rev_per_Vol_opt_array) / 1e6, color=colors[P_NBI_idx],
                    linewidth=1.5, marker='o', markersize=2,
                    label=f'$P_{{NBI}}$ = {P_NBI_target} MW')

            # TOP-RIGHT: Optimal Rm vs Eb
            ax2.plot(Eb_valid, Rm_opt_array, color=colors[P_NBI_idx],
                    linewidth=1.5, marker='o', markersize=2,
                    label=f'$P_{{NBI}}$ = {P_NBI_target} MW')

            # BOTTOM-LEFT: ⟨P_fus⟩ vs Eb
            ax3.plot(Eb_valid, P_fusion_opt_array, color=colors[P_NBI_idx],
                    linewidth=1.5, marker='o', markersize=2,
                    label=f'$P_{{NBI}}$ = {P_NBI_target} MW')

            # BOTTOM-RIGHT: Revenue (left axis) and Volume (right axis) vs Eb
            ax4.plot(Eb_valid, np.array(Revenue_opt_array) / 1e6, color=colors[P_NBI_idx],
                    linewidth=1.5, marker='o', markersize=2, linestyle='-',
                    label=f'Rev $P_{{NBI}}$={P_NBI_target}')
            ax4_twin.plot(Eb_valid, V_plasma_opt_array, color=colors[P_NBI_idx],
                    linewidth=1.5, marker='s', markersize=2, linestyle='--',
                    label=f'Vol $P_{{NBI}}$={P_NBI_target}')

    # TOP-LEFT subplot formatting
    ax1.set_xlabel(r'Beam Energy $E_{NBI}$ [keV]', fontsize=13, fontweight='bold')
    ax1.set_ylabel(r'Maximum Revenue/Volume [$M/yr/m³]', fontsize=13, fontweight='bold')
    ax1.set_title(r'Revenue/Volume vs Beam Energy' + '\n(at optimal $R_M$ for each point)',
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    ax1.tick_params(labelsize=11)

    # TOP-RIGHT subplot formatting
    ax2.set_xlabel(r'Beam Energy $E_{NBI}$ [keV]', fontsize=13, fontweight='bold')
    ax2.set_ylabel(r'Optimal $R_{M,vac}$ (Vacuum Mirror Ratio)', fontsize=13, fontweight='bold')
    ax2.set_title(r'Optimal Mirror Ratio vs Beam Energy' + '\n(for maximum Revenue/Volume at fixed $P_{NBI}$)',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    ax2.tick_params(labelsize=11)

    # BOTTOM-LEFT subplot formatting
    ax3.set_xlabel(r'Beam Energy $E_{NBI}$ [keV]', fontsize=13, fontweight='bold')
    ax3.set_ylabel(r'$\langle P_{fus} \rangle$ [MW]', fontsize=13, fontweight='bold')
    ax3.set_title(r'Capacity Factor Adjusted Fusion Power vs Beam Energy',
                  fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9, loc='best', framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    ax3.tick_params(labelsize=11)

    # BOTTOM-RIGHT subplot formatting (dual y-axis)
    ax4.set_xlabel(r'Beam Energy $E_{NBI}$ [keV]', fontsize=13, fontweight='bold')
    ax4.set_ylabel(r'Revenue [$M/yr]', fontsize=13, fontweight='bold', color='tab:blue')
    ax4_twin.set_ylabel(r'Volume [$m³]', fontsize=13, fontweight='bold', color='tab:orange')
    ax4.set_title(r'Revenue and Volume vs Beam Energy' + '\n(solid=Revenue, dashed=Volume)',
                  fontsize=12, fontweight='bold')
    ax4.tick_params(axis='y', labelcolor='tab:blue', labelsize=11)
    ax4_twin.tick_params(axis='y', labelcolor='tab:orange', labelsize=11)
    ax4.tick_params(axis='x', labelsize=11)
    ax4.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)

    # Combined legend for ax4
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='best', framealpha=0.9, ncol=2)

    # Main title
    fig.suptitle(f'Optimal Operating Points (Revenue/Volume) for Fixed $P_{{NBI}}$ Contours ($B_{{max}}$ = {B_max_default} T)',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()

    return fig, all_results


if __name__ == "__main__":
    print("="*80)
    print("Mirror Ratio Optimization for Maximum ⟨P_fus⟩")
    print(f"B_max = {B_max_default} T")
    print(f"Grid lifetime: {eta_duty*100:.0f}% duty cycle, {t_replace} month replacement")
    print(f"Beam: σ_x = {sigma_x_beam} cm, σ_y = {sigma_y_beam} cm, {num_grids} grids")
    print("="*80)

    # Define PNBI levels [MW]
    P_NBI_list = [15, 20, 25, 30, 34]

    # Mirror ratio range
    Rm_range = (4, 40)

    # Beam energy range for optimization analysis
    Eb_range = (0.4, 1.2)  # 40 to 120 keV

    # =========================================================================
    # Optimal Rm for each Eb (maximizing ⟨P_fus⟩)
    # =========================================================================
    print("\n" + "="*80)
    print("Optimal Rm vs Eb (Maximizing ⟨P_fus⟩)")
    print("="*80)

    fig, all_results_Pfus = plot_Eb_optimization_for_PNBI_Pfus(
        P_NBI_list, Eb_range, Rm_range, n_Eb_points=100, n_Rm_points=450
    )

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY: Optimal Operating Points (⟨P_fus⟩)")
    print("="*80)

    for P_NBI_target in P_NBI_list:
        data = all_results_Pfus[P_NBI_target]

        if len(data['Eb']) == 0:
            continue

        print(f"\nP_NBI = {P_NBI_target} MW:")
        print(f"{'Eb':>6} {'Rm':>6} {'⟨Pfus⟩':>8} {'Revenue':>10} {'V':>8} {'a0':>6} {'a_end':>6} {'L':>6} {'n20':>8} {'Q':>6}")
        print(f"{'[keV]':>6} {'':>6} {'[MW]':>8} {'[M$]':>10} {'[m³]':>8} {'[m]':>6} {'[m]':>6} {'[m]':>6} {'':>8} {'':>6}")
        print("-"*100)

        for i in range(len(data['Eb'])):
            print(f"{data['Eb'][i]:6.1f} {data['Rm_opt'][i]:6.2f} "
                  f"{data['P_fusion_opt'][i]:8.2f} {data['Revenue_opt'][i]/1e6:10.1f} "
                  f"{data['V_plasma_opt'][i]:8.3f} "
                  f"{data['a0_min_opt'][i]:6.3f} {data['a0_end_opt'][i]:6.3f} {data['L_plasma_opt'][i]:6.2f} "
                  f"{data['n20_opt'][i]:8.4f} {data['Q_opt'][i]:6.3f}")

        # Find maximum P_fus for this PNBI
        max_idx = np.argmax(data['P_fusion_opt'])
        print(f"\n*** MAXIMUM ⟨P_fus⟩ for P_NBI = {P_NBI_target} MW:")
        print(f"    E_b = {data['Eb'][max_idx]:.1f} keV, Rm = {data['Rm_opt'][max_idx]:.2f}")
        print(f"    ⟨P_fus⟩ = {data['P_fusion_opt'][max_idx]:.2f} MW, Q = {data['Q_opt'][max_idx]:.2f}")
        print(f"    Revenue = ${data['Revenue_opt'][max_idx]/1e6:.2f}M/yr")
        print(f"    V = {data['V_plasma_opt'][max_idx]:.3f} m³, a0 = {data['a0_min_opt'][max_idx]:.3f} m, a_end = {data['a0_end_opt'][max_idx]:.3f} m, L = {data['L_plasma_opt'][max_idx]:.2f} m")

    # Save figure
    output_path = figures_dir / 'Optimization_Pfus_vs_Eb_for_PNBI_contours.png'
    fig.savefig(output_path, dpi=figure_dpi, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

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

        # Calculate geometry with new model
        a_0_abs = calculate_a0_absorption(E_b_100keV, n20_opt)
        a_0_DCLC = calculate_a0_DCLC(E_b_100keV, B_0)
        a_0_adiabatic = calculate_a0_adiabaticity(E_b_100keV, B_0, beta_local)
        a_0_min = max(a_0_abs, a_0_DCLC, a_0_adiabatic)

        a_0_end = calculate_a0_end(a_0_min, B_0, B_max_default)
        L_plasma, V_plasma, vessel_surface_area = calculate_plasma_geometry_frustum(
            a_0_min, a_0_end, E_b_100keV, B_0
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
        print(f"  Minor Radius - Mirror End (a_0_end):  {a_0_end:.4f} m")
        print(f"  Minor Radius from Absorption:         {a_0_abs:.4f} m")
        print(f"  Minor Radius from DCLC:               {a_0_DCLC:.4f} m")
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
        print("DESIGN PARAMETERS:")
        print("="*80)
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
            f.write(f"  Minor Radius - Mirror End (a_0_end):  {a_0_end:.4f} m\n")
            f.write(f"  Minor Radius from Absorption:         {a_0_abs:.4f} m\n")
            f.write(f"  Minor Radius from DCLC:               {a_0_DCLC:.4f} m\n")
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
            f.write("DESIGN PARAMETERS:\n")
            f.write("="*80 + "\n")
            f.write(f"  Beta_c (critical beta):               {beta_c_default:.4f}\n")
            f.write("="*80 + "\n")

        print(f"Design point details saved to: {output_file}")

    plt.show()
