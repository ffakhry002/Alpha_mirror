"""
Identify the n20, Eb operating point that optimizes Rev/Vol
above a provided NWL and below a provided Pnbi for several Rm
at fixed Bm and injection angle
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patheffects as pe
import pandas as pd
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
    calculate_a0_cold_neutral_mfp,
    calculate_a0_end,
    calculate_plasma_geometry_frustum,
    calculate_collisionality,
    calculate_voltage_closed_lines,
    calculate_voltage_field_reversal,
    calculate_max_mirror_ratio_vortex,
    calculate_fusion_power,
    calculate_NBI_power,
    calculate_NBI_current,
    calculate_NWL,
    calculate_Q,
    calculate_max_n20_ecrh,
    calculate_Bw,
    calculate_a_w,
    calculate_heat_flux,
    calculate_ion_flux_on_target,
    calculate_target_erosion_rate,
    calculate_end_ring_thickness,
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
    n_20_min,
    n_grid_points,
    Q_levels,
    min_NWL,
    NWL_background,
    NWL_levels,
    P_fus_background,
    Rev_per_Vol_background,
    max_rev_per_vol,
    a0_levels,
    min_a0,
    P_fus_levels,
    P_fus_avg_levels,
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
    max_nbi_current,
    max_nbi_power_ftop,
)

def create_popcon(B_central, B_max=B_max_default, beta_c=beta_c_default):
    """
    Creates n20 vs Eb POPCON for B0, Bm and returns the optimal 
    Rev/Vol and corresponding values of n20 and Eb
    """
    print(f"Creating POPCON for B_central: {B_central:.2f}")
    # Calculate vacuum mirror ratio
    R_M_vac = B_max / B_central
    if R_M_vac < 4:
        print(f"R_M_vac < 4: R_M_Vac = {R_M_vac:.2f}. Returning NaNs")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Create grid using input E_b range
    E_b100 = np.linspace(E_b_min, E_b_max, n_grid_points)
    n_20_max = calculate_beta_limit(E_b_min, B_central, beta_c)
    n_20 = np.linspace(n_20_min, n_20_max, n_grid_points)

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
    a_0_cold_neutrals = calculate_a0_cold_neutral_mfp(n_20_grid)
    a_0_eng = min_a0 * np.ones_like(a_0_abs) # Practical engineering constraint
    # Stack all arrays along a new axis and find limiting constraint on a_0
    a_0_arrays = np.stack([a_0_abs, a_0_DCLC, a_0_cold_neutrals, a_0_adiabatic, a_0_eng], axis=0)
    a_0_limits = ['abs', 'DCLC', 'N MFP', 'adiab', 'eng']
    a_0_min = np.max(a_0_arrays, axis=0)
    a_0_min_which = np.argmax(a_0_arrays, axis=0)
    a_0_min_limit = np.array(a_0_limits)[a_0_min_which]

    # Calculate a0 at mirror throat from flux conservation
    a_0_end = calculate_a0_end(a_0_min, B_0_grid, B_max)

    # Calculate plasma geometry using FRUSTUM model
    L_plasma = np.zeros_like(a_0_min)
    V_plasma = np.zeros_like(a_0_min)
    V_fus = np.zeros_like(a_0_min)
    vessel_surface_area = np.zeros_like(a_0_min)

    for i in range(n_grid_points):
        for j in range(n_grid_points):
            L, Vp, Vf, A = calculate_plasma_geometry_frustum(
                a_0_min[i, j], a_0_end[i, j], E_b100_grid[i, j], B_0_grid[i, j]
            )
            L_plasma[i, j] = L
            V_plasma[i, j] = Vp
            V_fus[i,j] = Vf
            vessel_surface_area[i, j] = A

    # Calculate loss coefficient - use vacuum mirror ratio
    C_loss = calculate_loss_coefficient(E_b100_grid, R_M_vac)

    # Calculate required NBI power
    P_NBI_required = calculate_NBI_power(n_20_grid, V_plasma, E_b100_grid, R_M_vac, C_loss)
    I_NBI_required = calculate_NBI_current(P_NBI_required, E_b100_grid)

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
                Vf = V_fus[i, j]

                # Calculate fusion power
                P_fusion = calculate_fusion_power(E_b_100_point, n_20_point, Vf, T_i)

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
    P_fus_avg_density = P_fus_avg / V_fus

    # Calculate Revenue/Volume using capacity factor adjusted fusion power
    Revenue = calculate_isotope_revenue(P_fus_avg)  # [$/yr] using <P_fus>
    Rev_per_Vol = Revenue / V_plasma  # [$/yr/m³]
    # Calculate end plug magnetic field and heat flux
    Bw = calculate_Bw(E_b100_grid, B_0_grid, a_0_min)

    # BUG FIX: Use Q_beam_target instead of undefined Q
    q_w = calculate_heat_flux(P_NBI_required, Q_beam_target, a_0_min, B_0_grid, Bw)

    # Create masks for different regions
    mask_beta = n_20_grid > n_20_beta_limit
    mask_heat_flux = q_w >= 5
    mask_low_NWL = NWL_beam_target < min_NWL
    mask_nbi_limit = P_NBI_required > max_nbi_power_ftop
    # Mask for where density is too high for ECRH to heat center
    n_cutoff = calculate_max_n20_ecrh(B_central)
    mask_ecrh_cutoff = n_20_grid > n_cutoff
    mask_invalid = mask_beta | mask_heat_flux | mask_ecrh_cutoff | mask_low_NWL | mask_nbi_limit
    
    # Find max Rev per volume over the valid region by making invalid points -inf
    Rev_per_Vol_valid = np.where(~mask_invalid, Rev_per_Vol, -np.inf)
    max_rev = np.nanmax(Rev_per_Vol_valid)
    i, j = np.unravel_index(np.argmax(Rev_per_Vol_valid), Rev_per_Vol_valid.shape)
    n_20_opt = n_20_grid[i, j]
    E_b100_opt = E_b100_grid[i, j]
    # Get NWL, a_0, and limiting a_0 constraint
    nwl_opt = NWL_beam_target[i, j]
    a_0_opt = a_0_min[i, j]
    a_0_limit_opt = a_0_min_limit[i,j]
    return max_rev, n_20_opt, E_b100_opt, nwl_opt, a_0_opt, a_0_limit_opt

def popcon_scan(B_0_scan, B_max, bypass=False):
    fn = f'Rm_optimizaiton_Bm_{B_max:.0f}.csv'
    print(fn)
    if os.path.exists(fn) and not bypass:
        print(f"POPCON scan already exists.\nReading results from {fn}")
        df = pd.read_csv(fn)
        return df
    print(f"POPCON scan for $B_m = ${B_max:0f} T")
    max_rev_scan = np.zeros_like(B_0_scan)
    n_20_opt_scan = np.zeros_like(B_0_scan)
    E_b100_opt_scan = np.zeros_like(B_0_scan)
    nwl_opt_scan = np.zeros_like(B_0_scan)
    a_0_opt_scan = np.zeros_like(B_0_scan)
    a_0_limit_opt_scan = []
    for i, b0 in enumerate(B_0_scan):
        max_rev, n_20_opt, E_b100_opt, nwl_opt, a_0_opt, a_0_limit_opt = create_popcon(b0, B_max=B_max)
        max_rev_scan[i] = max_rev
        n_20_opt_scan[i] = n_20_opt
        E_b100_opt_scan[i] = E_b100_opt
        nwl_opt_scan[i] = nwl_opt
        a_0_opt_scan[i] = a_0_opt
        a_0_limit_opt_scan.append(a_0_limit_opt)
    df = pd.DataFrame({
        'B_0': B_0_scan,
        'Rev_per_vol_opt': max_rev_scan,
        'n_20_opt': n_20_opt_scan,
        'E_b100_opt': E_b100_opt_scan,
        'NWL_at_opt_Rev_per_vol': nwl_opt_scan,
        'a_0_opt': a_0_opt_scan,
        'a_0_limit_opt': np.array(a_0_limit_opt_scan, dtype=str),
    })
    df.to_csv(fn, index=False)
    return df


if __name__=="__main__":
    plt.rcParams['font.size'] = 11
    B_0_scan = np.arange(2.5, 7.25, 0.25)
    df_22 = popcon_scan(B_0_scan, B_max=22, bypass=False)
    df_25 = popcon_scan(B_0_scan, B_max=25, bypass=False)
    df_28 = popcon_scan(B_0_scan, B_max=28, bypass=False)
    dfs = [df_22, df_25, df_28]
    labels = ['$B_m = 22$ T', '$B_m = 25$ T', '$B_m = 28$ T']
    cmap = plt.get_cmap('Reds')
    Bm_colors = [cmap(x) for x in np.linspace(0.3, 0.9, 3)]
    for df, l, c in zip(dfs, labels, Bm_colors):
        plt.plot(df['B_0'], df['Rev_per_vol_opt']/1e6, marker='o', label=l, c=c)
    plt.xlabel(r'$B_0$ [T]', fontsize=14)
    plt.ylabel(r'$R/V_p$ [\$M/yr/m$^{3}$]', fontsize=14)
    plt.ylim(0, 6000)
    plt.xticks(np.arange(2.5, 7.5, 0.5))
    plt.xlim(2.5, 7)
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('rm_optimization.png')
    plt.show()
