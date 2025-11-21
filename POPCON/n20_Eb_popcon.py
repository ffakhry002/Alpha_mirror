"""n20_Eb_popcon.py - Create POPCON plot in n20 vs Eb space and analyze design points"""

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
    calculate_collisionality,
    calculate_voltage_closed_lines,
    calculate_voltage_field_reversal,
    calculate_max_mirror_ratio_vortex,
    calculate_fusion_power,
    calculate_NBI_power,
    calculate_NWL,
    calculate_Q,
    calculate_Bw,
    calculate_a_w,
    calculate_heat_flux,
)

from n20_Eb_inputs import (
    B_max_default,
    B_central_default,
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
    P_fus_background,
    a0_levels,
    min_a0,
    P_fus_levels,
    P_NBI_levels,
    B_0_levels,
    beta_levels,
    C_levels,
    R_M_levels,
    q_w_levels,
    qw_limit,
    Bw_levels,
    a_w_levels,
    max_R_M_vortex_levels,
    voltage_levels,
    nu_levels,
    test_points_list,
    figures_dir,
    figure_dpi,
    figure_size,
)


def create_full_popcon(B_max=B_max_default, B_central=B_central_default, beta_c=beta_c_default):
    """Create full POPCON plot with beam-target fusion physics using frustum geometry"""
    fig, ax = plt.subplots(figsize=figure_size)

    # Calculate vacuum mirror ratio
    R_M_vac = B_max / B_central

    # Create grid using input E_b range
    E_b100 = np.linspace(E_b_min, E_b_max, n_grid_points)
    n_20_max = calculate_beta_limit(E_b_min, B_central, beta_c)
    n_20 = np.linspace(0.01, n_20_max, n_grid_points)

    E_b100_grid, n_20_grid = np.meshgrid(E_b100, n_20)

    # Calculate constraints with NEW beta formulation
    n_20_beta_limit = calculate_beta_limit(E_b100_grid, B_central, beta_c)

    # Calculate local beta and on-axis field (diamagnetically adjusted)
    beta_local = calculate_beta_local(n_20_grid, E_b100_grid, B_central)
    B_0_grid = calculate_B0_with_diamagnetic(B_central, beta_local)

    # Calculate diamagnetic mirror ratio
    R_M_dmag = B_max / B_0_grid

    # Calculate geometry constraints
    a_0_abs = calculate_a0_absorption(E_b100_grid, n_20_grid)
    a_0_FLR = calculate_a0_FLR(E_b100_grid, B_0_grid, N_25)  # Use B_0 (diamagnetic)
    a_0_min = np.maximum(a_0_abs, a_0_FLR)

    # Calculate a0 at mirror field for frustum geometry - use B_max directly
    a_0_FLR_mirror = calculate_a0_FLR_at_mirror(E_b100_grid, B_max, N_25)

    # Calculate plasma geometry using FRUSTUM model
    L_plasma = np.zeros_like(a_0_min)
    V_plasma = np.zeros_like(a_0_min)
    vessel_surface_area = np.zeros_like(a_0_min)

    for i in range(n_grid_points):
        for j in range(n_grid_points):
            L, V, A = calculate_plasma_geometry_frustum(
                a_0_min[i, j], a_0_FLR_mirror[i, j], N_rho
            )
            L_plasma[i, j] = L
            V_plasma[i, j] = V
            vessel_surface_area[i, j] = A

    # Calculate loss coefficient - use vacuum mirror ratio
    C_loss = calculate_loss_coefficient(E_b100_grid, R_M_vac)

    # Calculate required NBI power
    P_NBI_required = calculate_NBI_power(n_20_grid, V_plasma, E_b100_grid, R_M_vac, C_loss)

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
                # Calculate temperature from Egedal scaling
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

    # Calculate collisionality for sanity check
    collisionality = calculate_collisionality(E_b_100keV=E_b100_grid, n_20=n_20_grid, L_plasma=L_plasma)
    print(f"Max collisionality: {np.nanmax(collisionality)}")
    print(f"Min collisionality: {np.nanmin(collisionality)}")

    # Calculate end-plate voltage bias for vortex stabilization
    voltage_cl = calculate_voltage_closed_lines(E_b100_grid, B_0_grid, a_0_min, L_plasma, R_M_dmag)
    print(f"Max Voltage for flow closure: {np.nanmax(voltage_cl)}")
    print(f"Min Voltage for flow closure: {np.nanmin(voltage_cl)}")
    voltage_fr = calculate_voltage_field_reversal(E_b100_grid, B_0_grid, a_0_min, L_plasma, R_M_dmag)
    print(f"Max Voltage for field reversal: {np.nanmax(voltage_fr)}")
    print(f"Min Voltage for field reversal: {np.nanmin(voltage_fr)}")
    end_plate_voltate = np.maximum(voltage_cl, voltage_fr)

    # Calculate mirror ratio limit for vortex stabilization
    max_R_M_vortex = calculate_max_mirror_ratio_vortex(E_b100_grid, B_0_grid, a_0_min, L_plasma)
    print(f"Max Rm for vortex stabilization: {np.nanmin(max_R_M_vortex)}")

    # Calculate end plug magnetic field and heat flux
    Bw = calculate_Bw(E_b100_grid, B_0_grid, a_0_min)
    q_w = calculate_heat_flux(P_NBI_required, Q, a_0_min, B_0_grid, Bw)
    # Calculate end plug radius
    a_w = calculate_a_w(a_0_min, B_0_grid, Bw)

    # Create masks for different regions
    mask_beta = n_20_grid > n_20_beta_limit
    # mask_impractical = a_0_min > a0_limit  # REMOVED: No max a0 limit
    mask_min_a0 = a_0_min < min_a0  # Minimum radius constraint
    mask_heat_flux = q_w >= 5
    mask_low_NWL = NWL_beam_target < 0.0

    mask_gray = mask_beta | mask_min_a0 | mask_heat_flux
    mask_black = np.zeros_like(mask_gray, dtype=bool)
    mask_white = (~mask_gray) & mask_low_NWL

    # Fill regions
    ax.contourf(E_b100_grid, n_20_grid, mask_gray.astype(int),
                levels=[0.5, 1.5], colors=['lightgray'], alpha=0.8)

    # Plot P_fus contours as background
    P_fus_valid = P_fusion_beam_target.copy()
    P_fus_valid[mask_gray | mask_black | mask_white] = np.nan

    im = ax.contourf(E_b100_grid, n_20_grid, P_fus_valid,
                     levels=P_fus_background, cmap='viridis', extend='max')

    # Also prepare NWL for contour lines (not background)
    NWL_valid = NWL_beam_target.copy()
    NWL_valid[mask_gray | mask_black | mask_white] = np.nan

    # Beta limit line
    ax.plot(E_b100, n_20_beta_limit[0, :], 'purple', linewidth=3, zorder=5,
            label='Beta limit')

    # Size limit boundary line (maximum) - REMOVED: No max a0 limit
    # size_limit_boundary = a_0_min - a0_eff_limit
    # ax.contour(E_b100_grid, n_20_grid, size_limit_boundary,
    #            levels=[0], colors=['darkred'], linewidths=2, linestyles='--', zorder=4)
    # ax.plot([], [], color='darkred', linewidth=2, linestyle='--',
    #         label=f"a0={a0_limit:.1f}m limit")

    # Minimum a0 boundary line
    min_a0_boundary = a_0_min - min_a0
    ax.contour(E_b100_grid, n_20_grid, min_a0_boundary,
               levels=[0], colors=['orange'], linewidths=2, linestyles='--', zorder=4)
    ax.plot([], [], color='orange', linewidth=2, linestyle='--',
            label=f"a0={min_a0:.2f}m min")

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

    # R_M (Mirror Ratio) contours - diamagnetic
    if len(R_M_levels) > 0:
        R_M_valid = R_M_dmag.copy()
        R_M_valid[mask_gray | mask_black | mask_white] = np.nan
        CS_RM = ax.contour(E_b100_grid, n_20_grid, R_M_valid,
                           levels=R_M_levels, colors='lime', linewidths=2,
                           alpha=0.8, linestyles='--')
        ax.clabel(CS_RM, inline=True, fontsize=9, fmt='R_M_dmag=%.0f')

    # Heat flux limit contour
    q_w_valid = q_w.copy()
    q_w_valid[mask_beta | mask_black | mask_white] = np.nan
    ax.contour(E_b100_grid, n_20_grid, q_w_valid,
               levels=[5], colors=['tab:orange'], linewidths=5, linestyles='-', zorder=4)
    ax.plot([], [], color='tab:orange', linewidth=3, linestyle='-',
            label=f"$q_w$={qw_limit} MW/m^2 limit")

    # Heat flux contours
    if len(q_w_levels) > 0:
        CS_qw = ax.contour(E_b100_grid, n_20_grid, q_w_valid,
                           levels=q_w_levels, colors='tab:orange', linewidths=2,
                           alpha=1.0, linestyles='-', label='$q_w = 5$ MW/m$^2$ limit')
        ax.clabel(CS_qw, inline=True, fontsize=12, fmt='$q_w$=%.1f')

    # End-plug magnetic field levels
    if len(Bw_levels) > 0:
        Bw_valid = Bw.copy()
        Bw_valid[mask_gray | mask_black | mask_white] = np.nan
        CS_BW = ax.contour(E_b100_grid, n_20_grid, Bw_valid,
                           levels=Bw_levels, colors='lime', linewidths=2,
                           alpha=1.0, linestyles='-')
        ax.clabel(CS_BW, inline=True, fontsize=10, fmt='$B_w$=%.2f')

    if len(a_w_levels) > 0:
        a_w_valid = a_w.copy()
        a_w_valid[mask_gray | mask_black | mask_white] = np.nan
        CS_AW = ax.contour(E_b100_grid, n_20_grid, a_w_valid,
                           levels=a_w_levels, colors='magenta', linewidths=2,
                           alpha=1.0, linestyles='-')
        ax.clabel(CS_AW, inline=True, fontsize=10, fmt='$a_w$=%.2f')

    # Max R_M contours for vortex stabilization
    if len(max_R_M_vortex_levels) > 0:
        max_R_M_vortex_valid = max_R_M_vortex.copy()
        max_R_M_vortex_valid[mask_gray | mask_black | mask_white] = np.nan
        CS_RM = ax.contour(E_b100_grid, n_20_grid, max_R_M_vortex_valid,
                           levels=max_R_M_vortex_levels, colors='magenta', linewidths=2,
                           alpha=0.8, linestyles='-')
        ax.clabel(CS_RM, inline=True, fontsize=10, fmt='R_M_max=%.0f')

    # end plate voltage contours
    if len(voltage_levels) > 0:
        voltage_valid = end_plate_voltate.copy()
        voltage_valid[mask_gray | mask_black | mask_white] = np.nan
        CS_V = ax.contour(E_b100_grid, n_20_grid, voltage_valid,
                           levels=voltage_levels, colors='#a0a0a0', linewidths=3,
                           alpha=1.0, linestyles='-')
        ax.clabel(CS_V, inline=True, fontsize=10, fmt='$e\\phi/T_e$=%.3f')

    # Collisionality contours
    if len(nu_levels) > 0:
        nu_valid = collisionality.copy()
        nu_valid[mask_gray | mask_black | mask_white] = np.nan
        CS_NU = ax.contour(E_b100_grid, n_20_grid, nu_valid,
                          levels=nu_levels, colors='tab:orange', linewidths=3,
                          alpha=0.8, linestyles='-')
        ax.clabel(CS_NU, inline=True, fontsize=8, fmt='$\\nu_{*}$=%.1e')


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
    cbar.set_label(r'$P_{fusion}$ [MW] (Beam-Target Fusion)', fontsize=12)
    # Use P_fus_background for colorbar ticks
    cbar_ticks = np.linspace(0, P_fus_background[-1], 6)
    cbar.set_ticks(cbar_ticks)
    cbar.ax.tick_params(labelsize=10)

    # Grid
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)

    ax.set_title(f'($B_{{max}}$={B_max}T, $B_{{central}}$={B_central:.1f}T, '
                 f'$R_{{M,vac}}$={R_M_vac:.2f}, $\\beta_c$={beta_c})\n'
                 f'Frustum Geometry',
                 fontsize=12, weight='bold')

    plt.tight_layout()

    return fig


def test_multiple_points(test_points=test_points_list, B_max=B_max_default,
                        B_central=B_central_default, beta_c=beta_c_default):
    """Test multiple design points and print summary table"""
    if not test_points:
        print("\nNo test points defined in test_points_list")
        return

    R_M_vac = B_max / B_central

    print("\n" + "="*100)
    print(f"TESTING MULTIPLE DESIGN POINTS: B_max={B_max}T, B_central={B_central}T, R_M_vac={R_M_vac:.2f}")
    print("="*100)

    # Header
    print(f"\n{'E_b':>6} {'n_20':>6} {'β':>8} {'B_0':>6} {'R_dmag':>7} {'a0_abs':>7} {'a0_FLR':>7} "
          f"{'a0_min':>7} {'L':>6} {'V':>7} {'C':>7} {'P_fus':>7} {'P_NBI':>7} "
          f"{'NWL':>6} {'Q':>6} {'Limit':>6}")
    print(f"{'[keV]':>6} {'[e20]':>6} {'':>8} {'[T]':>6} {'':>7} {'[m]':>7} {'[m]':>7} "
          f"{'[m]':>7} {'[m]':>6} {'[m³]':>7} {'[s]':>7} {'[MW]':>7} {'[MW]':>7} "
          f"{'[MW/m²]':>6} {'':>6} {'':>6}")
    print("-"*100)

    for E_b_100, n_20_target in test_points:
        E_NBI_keV = E_b_100 * 100

        # Calculate all parameters
        beta_local = calculate_beta_local(n_20_target, E_b_100, B_central)
        B_0 = calculate_B0_with_diamagnetic(B_central, beta_local)
        R_M_dmag = B_max / B_0

        a_0_abs = calculate_a0_absorption(E_b_100, n_20_target)
        a_0_FLR = calculate_a0_FLR(E_b_100, B_0, N_25)
        a_0_min = max(a_0_abs, a_0_FLR)
        limiting_constraint = "Abs" if a_0_abs > a_0_FLR else "FLR"

        a_0_FLR_mirror = calculate_a0_FLR_at_mirror(E_b_100, B_max, N_25)

        L_plasma, V_plasma, vessel_surface_area = calculate_plasma_geometry_frustum(
            a_0_min, a_0_FLR_mirror, N_rho
        )

        C_loss = calculate_loss_coefficient(E_b_100, R_M_vac)

        T_i = T_i_coeff * E_NBI_keV
        P_fusion = calculate_fusion_power(E_b_100, n_20_target, V_plasma, T_i)
        P_NBI = calculate_NBI_power(n_20_target, V_plasma, E_b_100, R_M_vac, C_loss)
        NWL = calculate_NWL(P_fusion, vessel_surface_area)
        Q = calculate_Q(P_fusion, P_NBI)

        # Print row
        print(f"{E_NBI_keV:6.0f} {n_20_target:6.2f} {beta_local:8.5f} {B_0:6.3f} {R_M_dmag:7.2f} "
              f"{a_0_abs:7.4f} {a_0_FLR:7.4f} {a_0_min:7.4f} {L_plasma:6.2f} "
              f"{V_plasma:7.3f} {C_loss:7.4f} {P_fusion:7.2f} {P_NBI:7.2f} "
              f"{NWL:6.3f} {Q:6.3f} {limiting_constraint:>6}")

    print("="*100 + "\n")


if __name__ == "__main__":
    print("Creating Beam-Target Fusion POPCON plots with frustum geometry...")
    print(f"Using B_max={B_max_default}T, B_central={B_central_default}T")
    print(f"Vacuum mirror ratio R_M_vac = {B_max_default/B_central_default:.2f}")

    print(f"Testing voltate calculation: BEAM voltage should be between 2 to 3")
    voltage_beam = calculate_voltage_closed_lines(1, 2.5, 0.3, 10, 12)
    voltage_beam = max(voltage_beam, calculate_voltage_field_reversal(1, 2.5, 0.3, 10, 12))
    print(f"BEAM voltage required: {voltage_beam}")

    # Test multiple design points
    print("\nTesting design points...")
    test_multiple_points()

    # Create main POPCON
    print("\nCreating main beam-target POPCON...")
    fig_single = create_full_popcon(B_max_default, B_central_default, beta_c_default)

    # Save figure
    output_path = figures_dir / 'POPCON_n20_Eb_Frustum.png'
    fig_single.savefig(output_path, dpi=figure_dpi, bbox_inches='tight')
    print(f"Saved: {output_path}")

    plt.show()
