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
    calculate_a0_DCLC,
    calculate_a0_adiabaticity,
    calculate_a0_end,
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
    calculate_grid_lifetime,
    calculate_capacity_factor_annual,
    calculate_average_fusion_power,
    calculate_isotope_revenue,
    calculate_revenue_per_volume,
)

from n20_Eb_inputs import (
    B_max_default,
    B_central_default,
    beta_c_default,
    T_i_coeff,
    T_e_coeff,
    E_b_min,
    E_b_max,
    n_grid_points,
    Q_levels,
    NWL_background,
    NWL_levels,
    P_fus_background,
    Rev_per_Vol_background,
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
    V_levels,
    a_w_levels,
    max_R_M_vortex_levels,
    voltage_levels,
    nu_levels,
    CF_levels,
    test_points_list,
    figures_dir,
    figure_dpi,
    figure_size,
    d_grid,
    t_replace,
    eta_duty,
    sigma_x_beam,
    sigma_y_beam,
    num_grids,
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
    a_0_DCLC = calculate_a0_DCLC(E_b100_grid, B_0_grid)  # DCLC stabilization (25*rho_i)
    a_0_adiabatic = calculate_a0_adiabaticity(E_b100_grid, B_0_grid, beta_local)  # Adiabaticity (50*rho_i*(1-sqrt(1-beta)))
    a_0_min = np.maximum(np.maximum(a_0_abs, a_0_DCLC), a_0_adiabatic)

    # Calculate a0 at mirror throat from flux conservation
    a_0_end = calculate_a0_end(a_0_min, B_0_grid, B_max)

    # Calculate plasma geometry using FRUSTUM model
    L_plasma = np.zeros_like(a_0_min)
    V_plasma = np.zeros_like(a_0_min)
    vessel_surface_area = np.zeros_like(a_0_min)

    for i in range(n_grid_points):
        for j in range(n_grid_points):
            L, V, A = calculate_plasma_geometry_frustum(
                a_0_min[i, j], a_0_end[i, j], E_b100_grid[i, j], B_0_grid[i, j]
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

    # Calculate capacity factor and time-averaged fusion power
    print(f"Calculating capacity factor for grid lifetime...")
    t_grid = calculate_grid_lifetime(
        E_b100_grid * 100,  # Convert to keV (not 100 keV units!)
        P_NBI_required,
        d_mm=d_grid,
        sigma_x_cm=sigma_x_beam,
        sigma_y_cm=sigma_y_beam,
        num_grids=num_grids
    )
    CF_annual = calculate_capacity_factor_annual(t_grid, t_replace_months=t_replace, eta_duty=eta_duty)
    P_fus_avg = calculate_average_fusion_power(P_fusion_beam_target, t_grid,
                                                t_replace_months=t_replace, eta_duty=eta_duty)

    # Calculate capacity factor adjusted fusion power density [MW/m³]
    P_fus_avg_density = P_fus_avg / V_plasma

    # Calculate Revenue/Volume using capacity factor adjusted fusion power
    Revenue = calculate_isotope_revenue(P_fus_avg)  # [$/yr] using <P_fus>
    Rev_per_Vol = Revenue / V_plasma  # [$/yr/m³]

    print(f"Capacity factor range: {np.nanmin(CF_annual):.3f} - {np.nanmax(CF_annual):.3f}")
    print(f"Grid lifetime range: {np.nanmin(t_grid):.1f} - {np.nanmax(t_grid):.1f} hours")
    print(f"⟨P_fus⟩/V range: {np.nanmin(P_fus_avg_density):.2f} - {np.nanmax(P_fus_avg_density):.2f} MW/m³")
    print(f"Revenue/Volume range: {np.nanmin(Rev_per_Vol)/1e6:.2f} - {np.nanmax(Rev_per_Vol)/1e6:.2f} $M/yr/m³")

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
    end_plate_voltage = np.maximum(voltage_cl, voltage_fr)

    # Calculate mirror ratio limit for vortex stabilization
    max_R_M_vortex = calculate_max_mirror_ratio_vortex(E_b100_grid, B_0_grid, a_0_min, L_plasma)
    print(f"Max Rm for vortex stabilization: {np.nanmin(max_R_M_vortex)}")

    # Calculate end plug magnetic field and heat flux
    Bw = calculate_Bw(E_b100_grid, B_0_grid, a_0_min)

    # BUG FIX: Use Q_beam_target instead of undefined Q
    q_w = calculate_heat_flux(P_NBI_required, Q_beam_target, a_0_min, B_0_grid, Bw)

    # Calculate end plug radius
    a_w = calculate_a_w(a_0_min, B_0_grid, Bw)

    # Create masks for different regions
    mask_beta = n_20_grid > n_20_beta_limit
    # mask_impractical = a_0_min > a0_limit  # REMOVED: No max a0 limit
    mask_min_a0 = a_0_min < min_a0  # Minimum radius constraint
    mask_heat_flux = q_w >= 5
    mask_low_NWL = NWL_beam_target < 0.0

    # NEW: Mask for invalid Bw region
    # Valid when Bw < B_max/74 (calculated end-wall field must be achievable)
    # Invalid when Bw > B_max/74 (required end-wall field too high)
    Bw_max_limit = B_max / 74.0  # Maximum allowable Bw
    mask_Bw_invalid = Bw > Bw_max_limit  # Invalid where Bw exceeds limit

    print(f"Bw range: {np.nanmin(Bw):.3f} - {np.nanmax(Bw):.3f} T")
    print(f"Bw_max_limit (B_max/74): {Bw_max_limit:.3f} T")
    print(f"Points with Bw > B_max/74 (invalid): {np.sum(mask_Bw_invalid)}")

    mask_gray = mask_beta | mask_min_a0 | mask_heat_flux
    mask_black = np.zeros_like(mask_gray, dtype=bool)
    mask_white = (~mask_gray) & mask_low_NWL

    # NEW: Add Bw invalid region as a separate hatched region
    mask_Bw_display = mask_Bw_invalid & (~mask_gray)  # Only show where not already gray

    # Fill regions
    ax.contourf(E_b100_grid, n_20_grid, mask_gray.astype(int),
                levels=[0.5, 1.5], colors=['lightgray'], alpha=0.8)

    # NEW: Fill Bw invalid region with hatching (different color to distinguish)
    ax.contourf(E_b100_grid, n_20_grid, mask_Bw_display.astype(int),
                levels=[0.5, 1.5], colors=['darkgray'], alpha=0.6, hatches=['//'])

    # ===========================================================================
    # CHANGED: Plot P_fus as background instead of Revenue/Volume
    # ===========================================================================
    P_fus_valid = P_fusion_beam_target.copy()
    P_fus_valid[mask_gray | mask_black | mask_white | mask_Bw_display] = np.nan

    # Create P_fus background levels with max at 10 MW
    max_P_fus_background = 25.0  # MW
    P_fus_background_levels = np.linspace(0, max_P_fus_background, 100)

    im = ax.contourf(E_b100_grid, n_20_grid, P_fus_valid,
                     levels=P_fus_background_levels, cmap='viridis', extend='max')

    # Also prepare NWL for contour lines (not background)
    NWL_valid = NWL_beam_target.copy()
    NWL_valid[mask_gray | mask_black | mask_white | mask_Bw_display] = np.nan

    # Beta limit line
    ax.plot(E_b100, n_20_beta_limit[0, :], 'purple', linewidth=3, zorder=5,
            label='Beta limit')

    # Minimum a0 boundary line
    min_a0_boundary = a_0_min - min_a0
    ax.contour(E_b100_grid, n_20_grid, min_a0_boundary,
               levels=[0], colors=['orange'], linewidths=2, linestyles='--', zorder=4)
    ax.plot([], [], color='orange', linewidth=2, linestyle='--',
            label=f"a0={min_a0:.2f}m min")

    # NEW: Bw = B_max/74 boundary line (valid below, invalid above)
    Bw_boundary = Bw - Bw_max_limit
    CS_Bw_boundary = ax.contour(E_b100_grid, n_20_grid, Bw_boundary,
               levels=[0], colors=['red'], linewidths=2, linestyles=':', zorder=4)
    ax.plot([], [], color='red', linewidth=2, linestyle=':',
            label=f"$B_w$=$B_m$/74 limit")

    # a₀ contours
    a_0_min_valid = a_0_min.copy()
    a_0_min_valid[mask_gray | mask_black | mask_white | mask_Bw_display] = np.nan

    CS = ax.contour(E_b100_grid, n_20_grid, a_0_min_valid,
                    levels=a0_levels, colors='pink', linewidths=1.5, alpha=0.9)

    for level in a0_levels:
        label = f'a₀={level:.2f}m'
        ax.clabel(CS, levels=[level], inline=True, fontsize=7, fmt=label)

    # Q contour lines
    Q_valid = Q_beam_target.copy()
    Q_valid[mask_gray | mask_black | mask_white | mask_Bw_display] = np.nan

    CS_Q = ax.contour(E_b100_grid, n_20_grid, Q_valid,
                      levels=Q_levels, colors='cyan', linewidths=1.5,
                      alpha=0.8, linestyles='-')
    ax.clabel(CS_Q, inline=True, fontsize=8, fmt='Q=%.2f')

    # ⟨P_fus⟩ contours (capacity factor adjusted fusion power)
    if len(P_fus_levels) > 0:
        P_fus_avg_valid = P_fus_avg.copy()
        P_fus_avg_valid[mask_gray | mask_black | mask_white | mask_Bw_display] = np.nan

        CS_Pfus = ax.contour(E_b100_grid, n_20_grid, P_fus_avg_valid,
                             levels=P_fus_levels, colors='magenta', linewidths=2.0,
                             alpha=0.9, linestyles='-')
        ax.clabel(CS_Pfus, inline=True, fontsize=9, fmt='⟨P_fus⟩=%.0f MW')

    # NWL contour lines
    CS_NWL = ax.contour(E_b100_grid, n_20_grid, NWL_valid,
                        levels=NWL_levels, colors='white', linewidths=1.0,
                        alpha=0.9, linestyles='-')
    ax.clabel(CS_NWL, inline=True, fontsize=8, fmt='%.1f MW/m²')

    # B₀ contours
    if len(B_0_levels) > 0:
        B_0_valid = B_0_grid.copy()
        B_0_valid[mask_gray | mask_black | mask_white | mask_Bw_display] = np.nan
        CS_B0 = ax.contour(E_b100_grid, n_20_grid, B_0_valid,
                           levels=B_0_levels, colors='orange', linewidths=1.5,
                           alpha=0.7, linestyles='-')
        ax.clabel(CS_B0, inline=True, fontsize=8, fmt='B₀=%.1f T')

    # P_NBI contours
    P_NBI_valid = P_NBI_required.copy()
    P_NBI_valid[mask_gray | mask_black | mask_white | mask_Bw_display] = np.nan
    CS_PNBI = ax.contour(E_b100_grid, n_20_grid, P_NBI_valid,
                         levels=P_NBI_levels, colors='red', linewidths=2.5,
                         alpha=1.0, linestyles='-')
    ax.clabel(CS_PNBI, inline=True, fontsize=8, fmt='P_NBI=%.0f MW')

    # Beta contours
    if len(beta_levels) > 0:
        beta_valid = beta_local.copy()
        beta_valid[mask_gray | mask_black | mask_white | mask_Bw_display] = np.nan
        CS_beta = ax.contour(E_b100_grid, n_20_grid, beta_valid,
                             levels=beta_levels, colors='orange', linewidths=1.5,
                             alpha=0.8, linestyles='-.')
        ax.clabel(CS_beta, inline=True, fontsize=8, fmt='β=%.2f')

    # C (Loss Coefficient) contours
    if len(C_levels) > 0:
        C_valid = C_loss.copy()
        C_valid[mask_gray | mask_black | mask_white | mask_Bw_display] = np.nan
        CS_C = ax.contour(E_b100_grid, n_20_grid, C_valid,
                          levels=C_levels, colors='brown', linewidths=1.5,
                          alpha=0.8, linestyles=':')
        ax.clabel(CS_C, inline=True, fontsize=8, fmt='C=%.2f s')

    # R_M (Mirror Ratio) contours - diamagnetic
    if len(R_M_levels) > 0:
        R_M_valid = R_M_dmag.copy()
        R_M_valid[mask_gray | mask_black | mask_white | mask_Bw_display] = np.nan
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
        Bw_valid[mask_gray | mask_black | mask_white | mask_Bw_display] = np.nan
        CS_BW = ax.contour(E_b100_grid, n_20_grid, Bw_valid,
                           levels=Bw_levels, colors='lime', linewidths=2,
                           alpha=1.0, linestyles='-')
        ax.clabel(CS_BW, inline=True, fontsize=10, fmt='$B_w$=%.2f')

    if len(a_w_levels) > 0:
        a_w_valid = a_w.copy()
        a_w_valid[mask_gray | mask_black | mask_white | mask_Bw_display] = np.nan
        CS_AW = ax.contour(E_b100_grid, n_20_grid, a_w_valid,
                           levels=a_w_levels, colors='magenta', linewidths=2,
                           alpha=1.0, linestyles='-')
        ax.clabel(CS_AW, inline=True, fontsize=10, fmt='$a_w$=%.2f')

    # Max R_M contours for vortex stabilization
    if len(max_R_M_vortex_levels) > 0:
        max_R_M_vortex_valid = max_R_M_vortex.copy()
        max_R_M_vortex_valid[mask_gray | mask_black | mask_white | mask_Bw_display] = np.nan
        CS_RM = ax.contour(E_b100_grid, n_20_grid, max_R_M_vortex_valid,
                           levels=max_R_M_vortex_levels, colors='magenta', linewidths=2,
                           alpha=0.8, linestyles='-')
        ax.clabel(CS_RM, inline=True, fontsize=10, fmt='R_M_max=%.0f')

    # end plate voltage contours
    if len(voltage_levels) > 0:
        voltage_valid = end_plate_voltage.copy()
        voltage_valid[mask_gray | mask_black | mask_white | mask_Bw_display] = np.nan
        CS_V = ax.contour(E_b100_grid, n_20_grid, voltage_valid,
                           levels=voltage_levels, colors='#a0a0a0', linewidths=3,
                           alpha=1.0, linestyles='-')
        ax.clabel(CS_V, inline=True, fontsize=10, fmt='$e\\phi/T_e$=%.3f')

    # Collisionality contours
    if len(nu_levels) > 0:
        nu_valid = collisionality.copy()
        nu_valid[mask_gray | mask_black | mask_white | mask_Bw_display] = np.nan
        CS_NU = ax.contour(E_b100_grid, n_20_grid, nu_valid,
                          levels=nu_levels, colors='tab:orange', linewidths=3,
                          alpha=0.8, linestyles='-')
        ax.clabel(CS_NU, inline=True, fontsize=8, fmt='$\\nu_{*}$=%.1e')

    # Capacity factor contours
    if len(CF_levels) > 0:
        CF_valid = CF_annual.copy()
        CF_valid[mask_gray | mask_black | mask_white | mask_Bw_display] = np.nan
        CS_CF = ax.contour(E_b100_grid, n_20_grid, CF_valid,
                          levels=CF_levels, colors='white', linewidths=1.0,
                          alpha=0.9, linestyles='-')
        ax.clabel(CS_CF, inline=True, fontsize=10, fmt='CF=%.1f')

    # Volume contours [m³]
    if len(V_levels) > 0:
        V_valid = V_plasma.copy()
        V_valid[mask_gray | mask_black | mask_white | mask_Bw_display] = np.nan
        CS_V = ax.contour(E_b100_grid, n_20_grid, V_valid,
                          levels=V_levels, colors='white', linewidths=1.5,
                          alpha=0.9, linestyles='-')
        ax.clabel(CS_V, inline=True, fontsize=9, fmt='V=%.1f m³')


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

    # ===========================================================================
    # CHANGED: Colorbar for P_fus (max 10 MW)
    # ===========================================================================
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r'$P_{fus}$ [MW]', fontsize=12)
    cbar_ticks = np.linspace(0, max_P_fus_background, 6)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f'{x:.1f}' for x in cbar_ticks])
    cbar.ax.tick_params(labelsize=10)

    # Grid
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)

    ax.set_title(f'($B_{{max}}$={B_max}T, $B_{{central}}$={B_central:.1f}T, '
                 f'$R_{{M,vac}}$={R_M_vac:.2f}, $\\beta_c$={beta_c})\n'
                 f'Frustum Geometry | Background: $P_{{fus}}$',
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
    # print(f"TESTING MULTIPLE DESIGN POINTS: B_max={B_max}T, B_central={B_central}T, R_M_vac={R_M_vac:.2f}")
    print("="*100)

    # Header
    print(f"\n{'E_b':>6} {'n_20':>6} {'β':>8} {'B_0':>6} {'R_dmag':>7} {'a0_abs':>7} {'a0_DCLC':>7} "
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
        a_0_DCLC = calculate_a0_DCLC(E_b_100, B_0)
        # BUG FIX: Use beta_local instead of undefined beta
        a_0_adiabatic = calculate_a0_adiabaticity(E_b_100, B_0, beta_local)
        a_0_min = max(a_0_abs, a_0_DCLC, a_0_adiabatic)

        # Determine limiting constraint
        if a_0_abs >= a_0_DCLC and a_0_abs >= a_0_adiabatic:
            limiting_constraint = "Abs"
        elif a_0_DCLC >= a_0_adiabatic:
            limiting_constraint = "DCLC"
        else:
            limiting_constraint = "Adiabatic"

        a_0_end = calculate_a0_end(a_0_min, B_0, B_max)

        L_plasma, V_plasma, vessel_surface_area = calculate_plasma_geometry_frustum(
            a_0_min, a_0_end, E_b_100, B_0
        )

        C_loss = calculate_loss_coefficient(E_b_100, R_M_vac)

        T_i = T_i_coeff * E_NBI_keV
        P_fusion = calculate_fusion_power(E_b_100, n_20_target, V_plasma, T_i)
        P_NBI = calculate_NBI_power(n_20_target, V_plasma, E_b_100, R_M_vac, C_loss)
        NWL = calculate_NWL(P_fusion, vessel_surface_area)
        Q = calculate_Q(P_fusion, P_NBI)

        # BUG FIX: Use a_0_DCLC instead of undefined a_0_FLR
        print(f"{E_NBI_keV:6.0f} {n_20_target:6.2f} {beta_local:8.5f} {B_0:6.3f} {R_M_dmag:7.2f} "
              f"{a_0_abs:7.4f} {a_0_DCLC:7.4f} {a_0_min:7.4f} {L_plasma:6.2f} "
              f"{V_plasma:7.3f} {C_loss:7.4f} {P_fusion:7.2f} {P_NBI:7.2f} "
              f"{NWL:6.3f} {Q:6.3f} {limiting_constraint:>6}")

    print("="*100 + "\n")


if __name__ == "__main__":
    print("Creating Beam-Target Fusion POPCON plots with frustum geometry...")
    print(f"Using B_max={B_max_default}T, B_central={B_central_default}T")
    print(f"Vacuum mirror ratio R_M_vac = {B_max_default/B_central_default:.2f}")

    print(f"Testing voltage calculation: BEAM voltage should be between 2 to 3")
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
