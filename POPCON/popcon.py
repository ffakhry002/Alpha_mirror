"""
Class for a POPCON of n20 vs Eb. 
Inputs to the POPCON are specified in a config file and
B0, Bm, N_rho, can be additionally changed since we want to scan those
Injection angle of 45 degrees is assumed. 
"""

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

import POPCON.utils.equations as eqn
from POPCON.params import Params

def add_label_outline(clabels, linewidth=2, foreground='k'):
    """
    Add white outline around contour label text so that the text appears
    visible over a wide range of background colors.
    """
    for txt in clabels:
        txt.set_path_effects([
            pe.withStroke(linewidth=linewidth, foreground=foreground)
        ])

class Popcon():
    # TODO: Convert data on Eb, n20 grid to xarray

    def __init__(self, B_0_vac=None, B_m=None, N_rho=None):
        """
        Params
        - default_Params, PopconParams: dictionary of default parameters
        - B_0, float: override central magnetic field, [T]
        - B_m, float: override mirror magnetic field, [T]
        - N_rho, float: override N_rho = a_0/rho_i
        """
        # Popcon inputs that can be overridden bc we may want to scan them
        self.B_0_vac = Params.B_central_default if B_0_vac is None else B_0_vac
        self.B_m = Params.B_max_default if B_m is None else B_m
        self.N_rho = Params.N_rho_default if N_rho is None else N_rho
        # Input derived quantities
        self.R_M_vac = self.B_m / self.B_0_vac

        # Popcon outpus
        # Can later make this an xarray DataSet
        self.E_b100_grid = None    # [100keV]
        self.n_20_grid = None      # [1e20 m^-3]
        self.beta_local = None     # Beta with diamagnetic correction
        self.B_0_grid = None       # 2D grid of B_0 with diamagnetic contribution [T]
        self.R_M_dmag = None       # 2D grid of mirror ratio incl. diamagnetic effects
        self.a_0_dict = None       # Dictionary storing 2D arrays of each min constraint on a_0 [m]
        self.a_0_min = None        # 2D array of final limit on a_0 [m]
        self.a_0_min_limit = None  # 2D array of strings describing limiting constraint on a_0
        self.L_mirror = None       # 2D array of distances between mirror throats [m]
        self.V_plasma = None       # 2D array of plasma volumes from frustum model [m^-3]
        self.C_loss = None         # 2D array of particle confinement times [s]
        self.P_nbi = None         # 2D array of NBI powers required for flattop fueling [MW]
        self.Q_phy = None          # 2D array of fusion power gain (physics)
        self.NWL = None            # 2D array of Neutron Wall Loading [MW/m^2]
        self.P_fus = None          # 2D array of fusion power during flattop [MW]
        self.P_fus_avg = None      # 2D array of capacity factor averaged fusion power [MW]
        self.CF = None             # 2D array of Capacity Factors for VNS
        self.rev_per_vol = None    # 2D array of revenue per unit plasma volume [$/m^-3]
        self.collisionality = None # ... collsionality
        self.end_plate_voltage = None  # ... Required voltage of end plates for vortex stabilization [V]
        self.B_w = None            # Min limit of magnetic field at end plate [T]
        self.a_w = None            # Plasma radius at end rings
        self.max_R_M_vortex = None # Maximum mirror ratio for votex stabilization to work
        self.q_w = None            # Heat flux at end plates [MW/m^2]
        self.ion_flux_target = None # Ion flux on end rings [m^-2 s^-1]
        self.ero_rate_w = None      # Erosion rate of tungsten [mm/yr]
        self.invalid = None        # Mask of all points that violate constraints
        self.invalid_dict = None   # Dictionary containing masks for each constraint

        # Useful for plotting and test points, maybe rework plotting function later
        self.E_b100 = None
        self.n_20 = None
        self.n_20_beta_limit = None
        self.n_cutoff = None
        self.B_w_max_limit = None
    
    def create_popcon(self):
        """
        Calculates all the POPCON output quantities on a 2D grid of n_20 vs Eb
        Returns the popcon object
        """
        # Create grid using input E_b range
        self.E_b100 = np.linspace(Params.E_b_min, Params.E_b_max, Params.n_grid_points)
        n_20_max = eqn.calculate_beta_limit(Params.E_b_min, self.B_0_vac, Params.beta_c_default)
        self.n_20 = np.linspace(Params.n_20_min, n_20_max, Params.n_grid_points)

        self.E_b100_grid, self.n_20_grid = np.meshgrid(self.E_b100, self.n_20)

        # Calculate constraints with NEW beta formulation
        self.n_20_beta_limit = eqn.calculate_beta_limit(self.E_b100_grid, self.B_0_vac, Params.beta_c_default)

        # Calculate local beta and on-axis field (diamagnetically adjusted)
        self.beta_local = eqn.calculate_beta_local(self.n_20_grid, self.E_b100_grid, self.B_0_vac)
        self.B_0_grid = eqn.calculate_B0_with_diamagnetic(self.B_0_vac, self.beta_local)

        # Calculate diamagnetic mirror ratio
        self.R_M_dmag = self.B_m / self.B_0_grid

        # Calculate geometry constraints
        a_0_abs = eqn.calculate_a0_absorption(self.E_b100_grid, self.n_20_grid)
        a_0_DCLC = eqn.calculate_a0_DCLC(self.E_b100_grid, self.B_0_grid, N_rho=self.N_rho)  # DCLC stabilization
        a_0_adiabatic = eqn.calculate_a0_adiabaticity(self.E_b100_grid, self.B_0_grid, self.beta_local)  # Adiabaticity (50*rho_i*(1-sqrt(1-beta)))
        a_0_cold_neutrals = eqn.calculate_a0_cold_neutral_mfp(self.n_20_grid)
        a_0_eng = Params.min_a0 * np.ones_like(a_0_abs) # Practical engineering constraint
        # Stack all arrays along a new axis and find limiting constraint on a_0
        self.a_0_dict = {
            'abs':    a_0_abs,
            'DCLC':   a_0_DCLC,
            'N MFP':  a_0_cold_neutrals,
            'adiab':  a_0_adiabatic,
            'eng':    a_0_eng,
        }
        a_0_arrays  = np.stack(list(self.a_0_dict.values()), axis=0)
        self.a_0_min     = np.max(a_0_arrays, axis=0)
        self.a_0_min_limit = np.array(list(self.a_0_dict.keys()))[np.argmax(a_0_arrays, axis=0)]

        # Calculate a0 at mirror throat from flux conservation
        a_0_end = eqn.calculate_a0_end(self.a_0_min, self.B_0_grid, self.B_m)

        # Calculate plasma geometry using FRUSTUM model
        self.L_mirror = np.zeros_like(self.a_0_min)
        self.V_plasma = np.zeros_like(self.a_0_min)
        V_fus = np.zeros_like(self.a_0_min)
        vessel_surface_area = np.zeros_like(self.a_0_min)

        for i in range(Params.n_grid_points):
            for j in range(Params.n_grid_points):
                L, Vp, Vf, A = eqn.calculate_plasma_geometry_frustum(
                    self.a_0_min[i, j], a_0_end[i, j], self.E_b100_grid[i, j], self.B_0_grid[i, j]
                )
                self.L_mirror[i, j] = L
                self.V_plasma[i, j] = Vp
                V_fus[i,j] = Vf
                vessel_surface_area[i, j] = A

        # Calculate loss coefficient - use vacuum mirror ratio
        self.C_loss = eqn.calculate_loss_coefficient(self.E_b100_grid, self.R_M_vac)

        # Calculate required NBI power
        self.P_nbi = eqn.calculate_NBI_power(self.n_20_grid, self.V_plasma, self.E_b100_grid, self.R_M_vac, self.C_loss)

        # Calculate beam-target fusion for full grid
        print(f"Calculating beam-target physics for {Params.n_grid_points}×{Params.n_grid_points} grid points...")

        self.P_fus = np.zeros_like(self.E_b100_grid)
        self.Q_phy = np.zeros_like(self.E_b100_grid)

        # Calculate for each grid point
        for i in range(Params.n_grid_points):
            for j in range(Params.n_grid_points):
                E_b_100_point = self.E_b100_grid[i, j]
                n_20_point = self.n_20_grid[i, j]
                E_NBI_keV = E_b_100_point * 100

                try:
                    # Calculate temperature from Egedal scaling
                    T_i = Params.T_i_coeff * E_NBI_keV
                    Vf = V_fus[i, j]

                    # Calculate fusion power
                    Pf = eqn.calculate_fusion_power(E_b_100_point, n_20_point, Vf, T_i)

                    # Calculate Q
                    if self.P_nbi[i, j] > 0:
                        Q = eqn.calculate_Q(Pf, self.P_nbi[i, j])
                    else:
                        Q = 0

                    self.P_fus[i, j] = Pf
                    self.Q_phy[i, j] = Q

                except Exception as e:
                    self.P_fus[i, j] = 0
                    self.Q_phy[i, j] = 0

        # Calculate NWL
        self.NWL = eqn.calculate_NWL(self.P_fus, vessel_surface_area)

        # Calculate capacity factor and time-averaged fusion power
        print(f"Calculating capacity factor for grid lifetime...")
        t_grid = eqn.calculate_grid_lifetime(
            self.E_b100_grid * 100,  # Convert to keV (not 100 keV units!)
            self.P_nbi,
            d_mm=Params.d_grid,
            sigma_x_cm=Params.sigma_x_beam,
            sigma_y_cm=Params.sigma_y_beam,
            num_grids=Params.num_grids
        )
        self.CF = eqn.calculate_capacity_factor_annual(t_grid, t_replace_months=Params.t_replace, eta_duty=Params.eta_duty)
        self.P_fus_avg = eqn.calculate_average_fusion_power(self.P_fus, t_grid,
                                                    t_replace_months=Params.t_replace, eta_duty=Params.eta_duty)
        
        # Calculate capacity factor adjusted fusion power density [MW/m³]
        P_fus_avg_density = self.P_fus_avg / V_fus

        # Calculate Revenue/Volume using capacity factor adjusted fusion power
        revenue = eqn.calculate_isotope_revenue(self.P_fus_avg)  # [$/yr] using <P_fus>
        self.rev_per_vol = revenue / self.V_plasma  # [$/yr/m³]

        print(f"Capacity factor range: {np.nanmin(self.CF):.3f} - {np.nanmax(self.CF):.3f}")
        print(f"Grid lifetime range: {np.nanmin(t_grid):.1f} - {np.nanmax(t_grid):.1f} hours")
        print(f"⟨P_fus⟩/V range: {np.nanmin(P_fus_avg_density):.2f} - {np.nanmax(P_fus_avg_density):.2f} MW/m³")
        print(f"Revenue/Volume range: {np.nanmin(self.rev_per_vol)/1e6:.2f} - {np.nanmax(self.rev_per_vol)/1e6:.2f} $M/yr/m³")

        # Calculate collisionality for sanity check
        self.collisionality = eqn.calculate_collisionality(E_b_100keV=self.E_b100_grid, n_20=self.n_20_grid, L_plasma=self.L_mirror)
        print(f"Max collisionality: {np.nanmax(self.collisionality)}")
        print(f"Min collisionality: {np.nanmin(self.collisionality)}")

        # Calculate end-plate voltage bias for vortex stabilization
        voltage_cl = eqn.calculate_voltage_closed_lines(self.E_b100_grid, self.B_0_grid, self.a_0_min, self.L_mirror, self.R_M_dmag)
        print(f"Max Voltage for flow closure: {np.nanmax(voltage_cl)}")
        print(f"Min Voltage for flow closure: {np.nanmin(voltage_cl)}")
        voltage_fr = eqn.calculate_voltage_field_reversal(self.E_b100_grid, self.B_0_grid, self.a_0_min, self.L_mirror, self.R_M_dmag)
        print(f"Max Voltage for field reversal: {np.nanmax(voltage_fr)}")
        print(f"Min Voltage for field reversal: {np.nanmin(voltage_fr)}")
        self.end_plate_voltage = np.maximum(voltage_cl, voltage_fr)

        # Calculate mirror ratio limit for vortex stabilization
        self.max_R_M_vortex = eqn.calculate_max_mirror_ratio_vortex(self.E_b100_grid, self.B_0_grid, self.a_0_min, self.L_mirror)
        print(f"Max Rm for vortex stabilization: {np.nanmin(self.max_R_M_vortex)}")

        # Calculate end plate params
        self.B_w = eqn.calculate_Bw(self.E_b100_grid, self.B_0_grid, self.a_0_min)
        self.q_w = eqn.calculate_heat_flux(self.P_nbi, self.Q_phy, self.a_0_min, self.B_0_grid, self.B_w)
        self.a_w = eqn.calculate_a_w(self.a_0_min, self.B_0_grid, self.B_w)
        self.ion_flux_target = eqn.calculate_ion_flux_on_target(P_nbi=self.P_nbi, E_b_100keV=self.E_b100_grid, a_w=self.a_w)
        self.ero_rate_w = eqn.calculate_target_erosion_rate(P_nbi=self.P_nbi, E_b_100keV=self.E_b100_grid, a_w=self.a_w)

        # Create masks for different regions
        mask_high_beta = self.n_20_grid > self.n_20_beta_limit
        mask_high_heat_flux = self.q_w >= 5
        mask_low_NWL = self.NWL < Params.min_NWL
        #mask_nbi_current_limit = I_NBI_required > max_nbi_current

        # NEW: Mask for invalid Bw region
        # Valid when Bw < B_max/74 (eqn.calculated end-wall field must be achievable)
        # Invalid when Bw > B_max/74 (required end-wall field too high)
        self.B_w_max_limit = self.B_m / 74.0  # Maximum allowable self.B_w
        mask_Bw_invalid = self.B_w > self.B_w_max_limit  # Invalid where Bw exceeds limit

        # Mask for where density is too high for ECRH to heat center
        # TODO: Consider diamagnetic effects in cutoff density
        self.n_cutoff = eqn.calculate_max_n20_ecrh(self.B_0_vac)
        print(f"Cutoff density: {self.n_cutoff}")
        mask_ecrh_cutoff = self.n_20_grid > self.n_cutoff

        print(f"Bw range: {np.nanmin(self.B_w):.3f} - {np.nanmax(self.B_w):.3f} T")
        print(f"Bw_max_limit (B_max/74): {self.B_w_max_limit:.3f} T")
        print(f"Points with Bw > B_max/74 (invalid): {np.sum(mask_Bw_invalid)}")

        self.invalid = mask_high_beta | mask_high_heat_flux | mask_ecrh_cutoff | mask_Bw_invalid
        self.invalid_dict = {
            'high_beta': mask_high_beta,
            'high_heat_flux': mask_high_heat_flux,
            'ecrh_cutoff': mask_ecrh_cutoff,
            'Bw_invalid': mask_Bw_invalid,
        }
        return self


    def plot_popcon(self, save_fig=True):
        """
        Plots the POPCON with a 2D heatmap of Revenue vs Volume
        and contours showing other quantities of interest specified in self.params
        Returns the figure
        """
        # Fill regions
        fig, ax = plt.subplots(figsize=Params.figure_size)
        ax.contourf(self.E_b100_grid, self.n_20_grid, self.invalid.astype(int),
                    levels=[0.5, 1.5], colors=['lightgray'], alpha=0.8)

        # ===========================================================================
        # CHANGED: Plot P_fus as background instead of Revenue/Volume
        # CHANGED AGAIN: Plot Rev/Volume as background instead of Pfus
        # ===========================================================================
        P_fus_valid = self.P_fus.copy()
        P_fus_valid[self.invalid] = np.nan

        # Create P_fus background levels with max at 10 MW
        max_P_fus_background = 25.0  # MW
        P_fus_background_levels = np.linspace(0, max_P_fus_background, 100)


        im = ax.contourf(self.E_b100_grid, self.n_20_grid, self.rev_per_vol/1e6,
                        levels=Params.Rev_per_Vol_background/1e6, cmap='viridis', extend='max')
        
        # Handles and labaels for contours
        legend_handles = []
        legend_labels = []

        # Also prepare NWL for contour lines (not background)
        NWL_valid = self.NWL.copy()
        NWL_valid[self.invalid] = np.nan

        # Beta limit line
        ax.plot(self.E_b100, self.n_20_beta_limit[0, :], 'purple', linewidth=4, zorder=5,
                label='Beta limit')
        
        # Max density line for cutoff
        ax.axhline(self.n_cutoff, linestyle='-', linewidth=4, c='cyan', zorder=5)


        # NEW: Bw = B_max/74 boundary line (valid below, invalid above)
        Bw_boundary = self.B_w - self.B_w_max_limit
        CS_Bw_boundary = ax.contour(self.E_b100_grid, self.n_20_grid, Bw_boundary,
                levels=[0], colors=['red'], linewidths=2, linestyles=':', zorder=4)
        # ax.plot([], [], color='red', linewidth=2, linestyle=':',
        #         label=f"$B_w$=$B_m$/74 limit")

        # a₀ contours
        a_0_min_valid = self.a_0_min.copy()
        a_0_min_valid[self.invalid] = np.nan

        CS = ax.contour(self.E_b100_grid, self.n_20_grid, a_0_min_valid,
                        levels=Params.a0_levels, colors='pink', linewidths=1.5, alpha=0.9)

        for level in Params.a0_levels:
            label = f'a₀={level:.2f}m'
            ax.clabel(CS, levels=[level], inline=True, fontsize=7, fmt=label)

        # Q contour lines
        Q_valid = self.Q_phy.copy()
        Q_valid[self.invalid] = np.nan

        CS_Q = ax.contour(self.E_b100_grid, self.n_20_grid, Q_valid,
                        levels=Params.Q_levels, colors='cyan', linewidths=1.5,
                        alpha=0.8, linestyles='-')
        ax.clabel(CS_Q, inline=True, fontsize=8, fmt='Q=%.2f')

        # P_fus contours (no capacity factor adjustment)
        if len(Params.P_fus_levels) > 0:
            P_fus_valid = self.P_fus.copy()
            P_fus_valid[self.invalid] = np.nan
            CS_Pfus = ax.contour(self.E_b100_grid, self.n_20_grid, P_fus_valid,
                                levels=Params.P_fus_levels, colors='tan', linewidths=2.5,
                                alpha=0.9, linestyles='-')
            h, _ = CS_Pfus.legend_elements()
            add_label_outline(ax.clabel(CS_Pfus, inline=True, fontsize=12, fmt='%.0f'))
            legend_handles.append(h[0])
            legend_labels.append(r'$P_{f}$ [MW]')

        # ⟨P_fus⟩ contours (capacity factor adjusted fusion power)
        if len(Params.P_fus_avg_levels) > 0:
            P_fus_avg_valid = self.P_fus_avg.copy()
            P_fus_avg_valid[self.invalid] = np.nan

            CS_Pfus_avg = ax.contour(self.E_b100_grid, self.n_20_grid, P_fus_avg_valid,
                                levels=Params.P_fus_avg_levels, colors='cyan', linewidths=2.0,
                                alpha=0.9, linestyles='-')
            h, _ = CS_Pfus_avg.legend_elements()
            add_label_outline(ax.clabel(CS_Pfus_avg, inline=True, fontsize=12, fmt='%.0f'))
            legend_handles.append(h[0])
            legend_labels.append('⟨P_fus⟩ [MW]')

        # NWL contour lines
        if len(Params.NWL_levels) > 0:
            CS_NWL = ax.contour(self.E_b100_grid, self.n_20_grid, NWL_valid,
                                levels=Params.NWL_levels, colors='w', linewidths=2.5,
                                alpha=0.9, linestyles='-')
            h, _ = CS_NWL.legend_elements()
            add_label_outline(ax.clabel(CS_NWL, inline=True, fontsize=12, fmt='%.1f'), foreground='k')
            legend_handles.append(h[0])
            legend_labels.append('$P_n/S$ [MW/$m^2$]')

        # B₀ contours
        if len(Params.B_0_levels) > 0:
            B_0_valid = self.B_0_grid.copy()
            B_0_valid[self.invalid] = np.nan
            CS_B0 = ax.contour(self.E_b100_grid, self.n_20_grid, B_0_valid,
                            levels=Params.B_0_levels, colors='orange', linewidths=1.5,
                            alpha=0.7, linestyles='-')
            ax.clabel(CS_B0, inline=True, fontsize=8, fmt='B₀=%.1f T')

        # P_NBI contours
        if len(Params.P_NBI_levels) > 0:
            P_NBI_valid = self.P_nbi.copy()
            P_NBI_valid[self.invalid] = np.nan
            CS_PNBI = ax.contour(self.E_b100_grid, self.n_20_grid, P_NBI_valid,
                                levels=Params.P_NBI_levels, colors='red', linewidths=2.5,
                                alpha=1.0, linestyles='-')
            add_label_outline(ax.clabel(CS_PNBI, inline=True, fontsize=12, fmt='%.0f'), foreground='k')
            h, _ = CS_PNBI.legend_elements()
            legend_handles.append(h[0])
            legend_labels.append('$P_{NBI}$ [MW]')

        # Beta contours
        if len(Params.beta_levels) > 0:
            beta_valid = self.beta_local.copy()
            beta_valid[self.invalid] = np.nan
            CS_beta = ax.contour(self.E_b100_grid, self.n_20_grid, beta_valid,
                                levels=Params.beta_levels, colors='orange', linewidths=1.5,
                                alpha=0.8, linestyles='-.')
            ax.clabel(CS_beta, inline=True, fontsize=8, fmt='β=%.2f')

        # C (Loss Coefficient) contours
        if len(Params.C_levels) > 0:
            C_valid = self.C_loss.copy()
            C_valid[self.invalid] = np.nan
            CS_C = ax.contour(self.E_b100_grid, self.n_20_grid, C_valid,
                            levels=Params.C_levels, colors='brown', linewidths=1.5,
                            alpha=0.8, linestyles=':')
            ax.clabel(CS_C, inline=True, fontsize=8, fmt='C=%.2f s')

        # R_M (Mirror Ratio) contours - diamagnetic
        if len(Params.R_M_levels) > 0:
            R_M_valid = self.R_M_dmag.copy()
            R_M_valid[self.invalid] = np.nan
            CS_RM = ax.contour(self.E_b100_grid, self.n_20_grid, R_M_valid,
                            levels=Params.R_M_levels, colors='lime', linewidths=2,
                            alpha=0.8, linestyles='--')
            ax.clabel(CS_RM, inline=True, fontsize=9, fmt='R_M_dmag=%.0f')

        # Heat flux limit contour
        # TODO: Plot other contours like this so that we don't need extra instance variables
        q_w_valid = self.q_w.copy()
        q_w_valid[self.invalid_dict['high_beta']] = np.nan
        ax.contour(self.E_b100_grid, self.n_20_grid, q_w_valid,
                levels=[Params.qw_limit], colors=['tab:orange'], linewidths=5, linestyles='-', zorder=4)
        ax.plot([], [], color='tab:orange', linewidth=3, linestyle='-',
                label=f"$q_w$={Params.qw_limit} MW/m^2 limit")

        # Heat flux contours
        if len(Params.q_w_levels) > 0:
            CS_qw = ax.contour(self.E_b100_grid, self.n_20_grid, q_w_valid,
                            levels=Params.q_w_levels, colors='tab:orange', linewidths=2,
                            alpha=1.0, linestyles='-', label='$q_w = 5$ MW/m$^2$ limit')
            ax.clabel(CS_qw, inline=True, fontsize=12, fmt='$q_w$=%.1f')

        # End-plug magnetic field levels
        if len(Params.Bw_levels) > 0:
            Bw_valid = self.B_w.copy()
            Bw_valid[self.invalid] = np.nan
            CS_BW = ax.contour(self.E_b100_grid, self.n_20_grid, Bw_valid,
                            levels=Params.Bw_levels, colors='lime', linewidths=2,
                            alpha=1.0, linestyles='-')
            ax.clabel(CS_BW, inline=True, fontsize=10, fmt='$B_w$=%.2f')

        if len(Params.a_w_levels) > 0:
            a_w_valid = self.a_w.copy()
            a_w_valid[self.invalid] = np.nan
            CS_AW = ax.contour(self.E_b100_grid, self.n_20_grid, a_w_valid,
                            levels=Params.a_w_levels, colors='magenta', linewidths=2,
                            alpha=1.0, linestyles='-')
            ax.clabel(CS_AW, inline=True, fontsize=10, fmt='$a_w$=%.2f')

        # Max R_M contours for vortex stabilization
        if len(Params.max_R_M_vortex_levels) > 0:
            max_R_M_vortex_valid = self.max_R_M_vortex.copy()
            max_R_M_vortex_valid[self.invalid] = np.nan
            CS_RM = ax.contour(self.E_b100_grid, self.n_20_grid, max_R_M_vortex_valid,
                            levels=Params.max_R_M_vortex_levels, colors='magenta', linewidths=2,
                            alpha=0.8, linestyles='-')
            ax.clabel(CS_RM, inline=True, fontsize=10, fmt='R_M_max=%.0f')

        # end plate voltage contours
        if len(Params.voltage_levels) > 0:
            voltage_valid = self.end_plate_voltage.copy()
            voltage_valid[self.invalid] = np.nan
            CS_V = ax.contour(self.E_b100_grid, self.n_20_grid, voltage_valid,
                            levels=Params.voltage_levels, colors='#a0a0a0', linewidths=3,
                            alpha=1.0, linestyles='-')
            ax.clabel(CS_V, inline=True, fontsize=10, fmt='$e\\phi/T_e$=%.3f')

        # Collisionality contours
        if len(Params.nu_levels) > 0:
            nu_valid = self.collisionality.copy()
            nu_valid[self.invalid] = np.nan
            CS_NU = ax.contour(self.E_b100_grid, self.n_20_grid, nu_valid,
                            levels=Params.nu_levels, colors='tab:orange', linewidths=3,
                            alpha=0.8, linestyles='-')
            ax.clabel(CS_NU, inline=True, fontsize=8, fmt='$\\nu_{*}$=%.1e')

        # Capacity factor contours
        if len(Params.CF_levels) > 0:
            CF_valid = self.CF.copy()
            CF_valid[self.invalid] = np.nan
            CS_CF = ax.contour(self.E_b100_grid, self.n_20_grid, CF_valid,
                            levels=Params.CF_levels, colors='white', linewidths=1.0,
                            alpha=0.9, linestyles='-')
            ax.clabel(CS_CF, inline=True, fontsize=10, fmt='CF=%.1f')

        # Volume contours [m³]
        if len(Params.V_levels) > 0:
            V_valid = self.V_plasma.copy()
            V_valid[self.invalid] = np.nan
            CS_V = ax.contour(self.E_b100_grid, self.n_20_grid, V_valid,
                            levels=Params.V_levels, colors='magenta', linewidths=1.5,
                            alpha=0.9, linestyles='-')
            ax.clabel(CS_V, inline=True, fontsize=9, fmt='V=%.1f m³')

        # Gray out hard limits
        ax.contourf(self.E_b100_grid, self.n_20_grid, self.invalid.astype(int),
            levels=[0.5, 1.5], colors=['lightgray'], alpha=1.0)
        
        # Text for hard limits
        ax.text(1.0, 2.93, 'Beta Limit', fontsize=18, c='purple', rotation=-40, zorder=10)
        ax.text(0.55, 3.7, 'Heat Flux Limit', fontsize=18, c='tab:orange', rotation=6, zorder=10)

        # Test point:
        for Eb, n20 in Params.test_points_list:
            star = ax.scatter(Eb, n20, marker='*', s=400, color='magenta', 
                            edgecolors='w', zorder=10)
        if len(Params.test_points_list) > 0:
            legend_handles.append(star)
            legend_labels.append('Design Point')

        # Formatting
        ax.set_xlabel(r'$E_{b}$ [100 keV]', fontsize=18)
        ax.set_ylabel(r'$n_{20}$ [$10^{20}$ m$^{-3}$]', fontsize=18)
        ax.set_xlim([Params.E_b_min, Params.E_b_max])
        ax.set_ylim([Params.n_20_min, 4])

        # Force linear tick formatting
        ax.ticklabel_format(style='plain', axis='x')
        ax.ticklabel_format(style='plain', axis='y')
        x_ticks = np.arange(Params.E_b_min, Params.E_b_max + 0.1, 0.2)
        ax.set_xticks(x_ticks)

        # Legend
        ax.legend(legend_handles, legend_labels, loc='upper right', 
                fontsize=14, facecolor='dimgray', labelcolor='white', edgecolor='white')

        # ===========================================================================
        # CHANGED: Colorbar for P_fus (max 10 MW)-- Change back to Rev/Vol
        # ===========================================================================
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(r'$R/V_p$ [\$M/yr/$m^3$]', fontsize=18)
        cbar_ticks = np.linspace(0, Params.max_rev_per_vol/1e6, 6)
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels([f'{x:.0f}' for x in cbar_ticks])
        cbar.ax.tick_params(labelsize=14)

        # Grid
        ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)

        # ax.set_title(f'($B_{{max}}$={B_max}T, $B_{{central}}$={B_central:.1f}T, '
        #              f'$R_{{M,vac}}$={self.R_M_vac:.2f}, $\\beta_c$={beta_c})\n'
        #              f'Frustum Geometry | Git Hash: {get_git_hash()}',
        #              fontsize=12, weight='bold')
        ax.set_title(rf'$B_m = {self.B_m:.0f}$ T, $B_0 = {self.B_0_vac:.2f}$ T', fontsize=18)
        plt.tight_layout()
        if save_fig:
            output_path = Params.figures_dir / 'POPCON_n20_Eb_Frustum.png'
            fig.savefig(output_path, dpi=Params.figure_dpi, bbox_inches='tight')
            print(f"Saved: {output_path}")
        return fig

    def print_test_points(self, test_points=Params.test_points_list):
        print("\n" + "="*100)
        # print(f"TESTING MULTIPLE DESIGN POINTS: B_max={B_max}T, B_central={B_central}T, R_M_vac={R_M_vac:.2f}")
        print("="*100)

        # Header
        print(f"\n{'E_b':>6} {'n_20':>6} {'Rev/V':>8} {'CF':>6} {'β':>8} {'B_0':>6}"
            f"{'R_dmag':>7} {'a0_abs':>7} {'a0_DCLC':>7} {'a0_nmfp':>7}"
            f"{'a0_min':>7} {'L':>6} {'V':>7} {'C':>7} {'P_fus':>7} {'P_NBI':>7} "
            f"{'NWL':>6} {'Q':>6} {'Limit':>6} {'q_w':>6} {'a_w':>6} {'B_w':>6} "
            f"{'ion_flux_w':>12} {'ero_rate_w':>8}")
        print(f"{'[keV]':>6} {'[e20]':>6} {'[$M/yr/m^3]':>8} {'':>6} {'':>6} {'[T]':>6} {'':>6} {'[m]':>7} {'[m]':>7} {'[m]':>7}"
            f"{'[m]':>7} {'[m]':>6} {'[m³]':>7} {'[s]':>7} {'[MW]':>7} {'[MW]':>7} "
            f"{'[MW/m²]':>6} {'':>5} {'':>6} {'[MW/m^2]':>6} {'[m]':>6} {'[T]':>6} {"[1e20/m^2*s]":>12} {'[mm/yr]':>12}")
        print("-"*100)

        for E_b100_target, n_20_target in test_points:
            # Get j then i due to how np.meshgrid orients the array
            j = np.argmin(np.abs(self.E_b100 - E_b100_target))
            i = np.argmin(np.abs(self.n_20 - n_20_target))

            print(f"{E_b100_target/100:6.0f} {n_20_target:6.2f} {self.rev_per_vol[i,j]/1e6:9.0f}" 
                  f"{self.CF[i,j]:6.3f} {self.beta_local[i,j]:6.3f} {self.B_0_grid[i,j]:6.3f} "
                  f"{self.R_M_dmag[i,j]:7.2f} {self.a_0_dict['abs'][i,j]:7.4f} {self.a_0_dict['DCLC'][i,j]:7.4f}"
                  f"{self.a_0_dict['N MFP'][i,j]:7.4f} {self.a_0_min[i,j]:7.4f} {self.L_mirror[i,j]:6.2f} "
                  f"{self.V_plasma[i,j]:7.3f} {self.C_loss[i,j]:7.4f} {self.P_fus[i,j]:7.2f} {self.P_nbi[i,j]:7.2f} "
                  f"{self.NWL[i,j]:6.3f} {self.Q_phy[i,j]:6.3f} {self.a_0_min_limit[i,j]:>6} {self.q_w[i,j]:6.1f}"
                  f"{self.a_w[i,j]:6.3} {self.B_w[i,j]:6.3} {self.ion_flux_target[i,j]/1e20:12.2f}"
                  f"{self.ero_rate_w[i,j]:7.4f}") 
        return

if __name__ == "__main__":
    print("Creating Beam-Target Fusion POPCON plots with frustum geometry...")
    print(f"Using B_max={Params.B_max_default}T, B_central={Params.B_central_default}T")
    print(f"Vacuum mirror ratio R_M_vac = {Params.B_max_default/Params.B_central_default:.2f}")

    print(f"Testing voltage calculation: BEAM voltage should be between 2 to 3")
    voltage_beam = eqn.calculate_voltage_closed_lines(1, 2.5, 0.3, 10, 12)
    voltage_beam = max(voltage_beam, eqn.calculate_voltage_field_reversal(1, 2.5, 0.3, 10, 12))
    print(f"BEAM voltage required: {voltage_beam}")

    # Test multiple design points
    print("\nTesting design points...")

    # Create main POPCON with default parameters
    plt.rcParams['font.size'] = 14
    print("\nCreating main beam-target POPCON...")
    popcon = Popcon()
    popcon.create_popcon()
    popcon.print_test_points()
    fig_single = popcon.plot_popcon()
    plt.show()