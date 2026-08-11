"""
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.constants as const
from scipy.integrate import trapezoid

from equilibrium.axial_profiles import AxialProfiles
from POPCON.equations import load_dt_reactivity_data, calculate_a0_absorption, calculate_a0_FLR, calculate_loss_coefficient


def read_vacuum_field(csv: str) -> pd.DataFrame:
    """
    Read the vaccuum magnetic field half-profile from Mirror-optimization
    Inputs:
    - fn: str, the .csv filename storing the magnetic field profile
    Returns: a pandas DataFrame storing the magnetic field profile
    """
    vacuum_B = pd.read_csv(csv)
    idx_throat = np.argmax(vacuum_B['B_z'])
    z_throat = np.abs(vacuum_B['z'][idx_throat])
    return vacuum_B[(vacuum_B['z'] > 0) & (vacuum_B['z'] < z_throat)][['z', 'B_z']]

def get_effective_Rm_and_theta_NBI(B_profile, theta_NBI) -> tuple[float, float]:
    """
    """
    min_B = np.min(B_profile['B_z'])
    B_0 = B_profile['B_z'].to_numpy()[0]
    theta_NBI_at_min_B = np.arcsin(np.sin(theta_NBI)*np.sqrt(min_B / B_0))
    R_m_at_min_B = np.max(B_profile['B_z']) / min_B
    return R_m_at_min_B, theta_NBI_at_min_B

def normalize_profiles(profiles: pd.DataFrame, n0: float, E_NBI_keV: float) -> pd.DataFrame:
    """
    """
    T_e = 0.1*E_NBI_keV * 1e3*const.e #[J]
    T_i = 2/3*E_NBI_keV * 1e3*const.e # [J]
    p0 = n0 * (T_e + T_i)
    profiles = profiles.sort_values(by='z')
    profiles['n'] = profiles['n'] / profiles['n'].to_numpy()[0] * n0
    profiles['p_perp'] = profiles['p_perp'] / profiles['p_perp'].to_numpy()[0] * p0
    profiles['p_par'] = profiles['p_par'] / profiles['p_par'].to_numpy()[0] * p0
    return profiles

def normalize_profiles_new(profiles: pd.DataFrame, n0: float) -> pd.DataFrame:
    """
    """
    vol = 2*trapezoid(np.pi*profiles['a']**2, profiles['z'])
    n_vol_avg = trapezoid(profiles['n']*np.pi*profiles['a']**2, profiles['z']) / vol
    norm_factor = n0 / n_vol_avg
    print(norm_factor)
    profiles['n'] = norm_factor * profiles['n']
    profiles['p_perp'] = norm_factor * profiles['p_perp']
    profiles['p_par'] = norm_factor * profiles['p_par']
    return profiles

def get_kinetic_profiles(B_profile: pd.DataFrame, theta_NBI: float, E_NBI_keV: float, cache_dir: str) -> pd.DataFrame:
    """
    """
    # Calculate kinetic profiles from the Egedal22 Sec. 2 distribution function
    R_m_at_min_B, theta_NBI_at_min_B = get_effective_Rm_and_theta_NBI(B_profile, theta_NBI)
    print(f"Getting axial profiles from Sam's code, R_m = {R_m_at_min_B}")
    ap = AxialProfiles(R_m=R_m_at_min_B, theta_NBI=theta_NBI_at_min_B, E_NBI=E_NBI_keV,
                       cache_dir=cache_dir)
    egedal_profiles = ap.get_profiles()
    ap.save_cache()
    print("Saved kinetic profile cache")
    egedal_profiles = egedal_profiles.sort_values(by='b')
    # Return kinetic + magnetic profiles as functions of z, interpolated onto higher sampled magnetic basis
    profiles = B_profile.copy()
    profiles['b'] = profiles['B_z'] / np.min(profiles['B_z'].to_numpy())
    profiles = profiles.sort_values(by='b')
    profiles['n'] = np.interp(profiles['b'].to_numpy(), egedal_profiles['b'].to_numpy(), egedal_profiles['n'].to_numpy())
    profiles['p_perp'] = np.interp(profiles['b'].to_numpy(), egedal_profiles['b'].to_numpy(), egedal_profiles['p_perp'].to_numpy())
    profiles['p_par'] = np.interp(profiles['b'].to_numpy(), egedal_profiles['b'].to_numpy(), egedal_profiles['p_par'].to_numpy())
    profiles = profiles.sort_values(by='z')
    return profiles

def add_plasma_beta_profile(profiles: pd.DataFrame, vacuum_profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the local plasma beta as a function of z as a column in the profiles DataFrame
    """
    # TODO: Use vacuum field
    profiles['beta'] = 2 * 4*np.pi*1e-7 * profiles['p_perp'] / vacuum_profiles['B_z']**2
    return profiles

def add_plasma_radius_profile(profiles: pd.DataFrame, E_NBI_keV, n0) -> pd.DataFrame:
    """
    Adds a column corresponding to the plasma radius from flux conservation
    """
    # Get plasma minor radius at mirror center
    a0_absorp = calculate_a0_absorption(E_b_100keV=E_NBI_keV/100, n_20=n0/1e20)
    a0_FLR = calculate_a0_FLR(E_b_100keV=E_NBI_keV/100, B_0=profiles['B_z'].to_numpy()[0])
    a0 = max(a0_absorp, a0_FLR)
    # Calculate a(z) from flux conservation
    B0 = profiles['B_z'].to_numpy()[0]
    profiles['a'] = a0 * np.sqrt(B0 / profiles['B_z'])
    return profiles

def add_fusion_power_density_profile(profiles: pd.DataFrame, E_NBI_keV) -> pd.DataFrame:
    """
    Adds a column corresponding to the fusion power density [MW/m^3]
    """
    Ti_keV = 2/3 * E_NBI_keV
    E_fus_MJ = 17.6 * const.e # [MeV] -> [MJ]
    dt_reactivity_interp = load_dt_reactivity_data()
    dt_reactivity = dt_reactivity_interp(Ti_keV)
    profiles['S_fus'] = 0.25*profiles['n']**2 * dt_reactivity * E_fus_MJ
    return profiles

def get_nwl_profile(profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a column corresponding to the NWL axial profile.
    Use assumption NWL = P_n / 2pi*r_wall where r_wall is the local
    radius of the wall
    """
    S_n = profiles['S_fus'].to_numpy() * 14.1/17.6
    r_wall = profiles['a'] + 0.1*profiles['a'].to_numpy()[0]
    profiles['nwl'] = S_n * profiles['a']**2 / (2*r_wall)
    return profiles

def get_fusion_power(profiles: pd.DataFrame) -> float:
    """
    Calculate the total fusion power in the mirror
    """
    # Factor of 2 because profiles stores the half-profiles z > 0
    return 2*trapezoid(profiles['S_fus']*np.pi*profiles['a']**2, profiles['z'])

def get_number_ions(profiles: pd.DataFrame) -> float:
    """Calculates the total number of DT ions in the plasma"""
    return 2*trapezoid(profiles['n']*np.pi*profiles['a']**2, profiles['z'])

def get_required_nbi_power(profiles: pd.DataFrame, Rm_vac: float, E_NBI_keV: float, n0: float) -> float:
    """
    Calculate the required NBI power [MW] to supply the density profile
    """
    num_ions = get_number_ions(profiles)
    loss_coef = calculate_loss_coefficient(E_b_100keV=E_NBI_keV/100, Rm_vac=Rm_vac)
    tau_p = loss_coef * (E_NBI_keV/100)**(3/2) * np.log10(Rm_vac)/(n0/1e20) # [s]
    E_NBI_MJ = E_NBI_keV/1e3 * const.e
    # Fudge factor of 2, divide by 0.9 to account for absorption loss
    return 2* E_NBI_MJ * num_ions / tau_p / 0.9

def get_tritium_mass_in_plasma(profiles: pd.DataFrame) -> float:
    """
    Returns the tritium mass [g] in the plasma at a single time
    """
    num_tritium_ions = 0.5 * get_number_ions(profiles)
    return get_number_ions(profiles) * 3*const.atomic_mass * 1e3

def get_tritium_mass_current(pnbi: float, Enbi_keV) -> float:
    """
    Returns the mass flow of tritium through the NBI ducts in [g/s]
    Pnbi in [MW]
    """
    E_NBI_MJ = E_NBI_keV/1e3 * const.e
    tritium_current = 0.5 * pnbi / E_NBI_MJ 
    print(f"tritium_current: {tritium_current}")
    return tritium_current * 3*const.atomic_mass*1e3

def get_plasma_volume(profiles: pd.DataFrame) -> float:
    return 2*trapezoid(np.pi*profiles['a']**2, profiles['z'])

def get_plasma_surface_area(profiles: pd.DataFrame) -> float:
    return 2*trapezoid(2*np.pi*profiles['a'], profiles['z'])

def get_wall_surface_area(profiles: pd.DataFrame) -> float:
    r_wall = profiles['a'] + 0.1*profiles['a'].to_numpy()[0]
    return 2*trapezoid(2*np.pi*r_wall, profiles['z'])

if __name__=='__main__':
    # physical constants
    m_p = 1.6726e-27 #[kg]
    m_e = 9.1093e-31 #[kg]
    eCharge = 1.60217e-19 #[C]

    # parameters
    # R_m = 4.0 
    mu_i = 2.5
    theta_NBI = np.pi/4.0 #[rad]
    E_NBI_keV = 47.4 #[keV]
    n0 = 2.1e20 # Volume averaged density [m^-3]
    max_iterations = 2
    vacuum_field_fn = 'Mirror-optimization/B_tot_vs_z_3m_final_v2.csv'
    cache_dir = 'equilibrium/cache'

    #  1) Read in vacuum magnetic equilibrium
    vacuum_B = read_vacuum_field(csv=vacuum_field_fn)
    print(vacuum_B)
    # plt.plot(vacuum_B['z'], vacuum_B['B_z'])
    # idx_wall = np.argmin(np.abs(vacuum_B['B_z'] - 0.0418))
    # plt.axvline(vacuum_B['z'].to_numpy()[idx_wall])
    # plt.axhline(vacuum_B['B_z'].to_numpy()[idx_wall])
    # plt.show()
    Rm_vac = np.max(vacuum_B['B_z'].to_numpy())/vacuum_B['B_z'].to_numpy()[0]
    print(f"Vacuum Rm = {Rm_vac}")

    B_tot = vacuum_B.copy()
    each_iteration_Btot = [] # For visualization of convergence
    for i in range(max_iterations):
        #  2) Find effective mirror ratio based on min B
        each_iteration_Btot.append(B_tot['B_z'].to_numpy())

        #  3) Pass parameters to axial profiles
        profiles = get_kinetic_profiles(B_profile=B_tot, theta_NBI=theta_NBI, 
                                         E_NBI_keV=E_NBI_keV, cache_dir=cache_dir)
        print(profiles)
        
        # # Get plasma radius as profile
        profiles = add_plasma_radius_profile(profiles, E_NBI_keV=E_NBI_keV, n0=n0)
        print(profiles)

        # # TODO: Add <sigma v> as profile

        # 4) Normalize profiles based on n0. Later normalize based on Pnbi
        #profiles = normalize_profiles_new(profiles=profiles, n0 = n0)
        profiles = normalize_profiles(profiles=profiles, n0=n0, E_NBI_keV=E_NBI_keV)
        #4) normalize profiles via normalization factor n20/<n>

        # 5) Find diamagnetic field
        profiles = add_plasma_beta_profile(profiles, vacuum_B)
        #profiles['beta_forest'] = 3* profiles['n']/1e20 * E_NBI_keV/100 / vacuum_B['B_z']**2
        B_tot['B_z'] = vacuum_B['B_z'] * np.sqrt(1 - profiles['beta'])
    each_iteration_Btot.append(B_tot['B_z'].to_numpy())

    
    # Get plasma radius
    profiles = add_plasma_radius_profile(profiles, E_NBI_keV=E_NBI_keV, n0=n0)
    profiles = add_fusion_power_density_profile(profiles, E_NBI_keV=E_NBI_keV)
    profiles = get_nwl_profile(profiles)
    profiles['B_z_vac'] = vacuum_B['B_z']

    # Constant columns for replicating data
    profiles['E_b_keV'] = E_NBI_keV
    profiles['n0'] = n0
    #profiles.to_csv('equilibrium/axial_profiles_final_v2.csv', index=False)

    # Plot to show convergence
    # for i, btot in enumerate(each_iteration_Btot):
    #     plt.plot(profiles['z'], btot, label=f'Iter {i}')
    # # plt.plot(profiles['z'], profiles['B_z'] - profiles['B_z']*np.sqrt(1-profiles['beta']), c='tab:orange', label='B dia')
    # # plt.plot(profiles['z'], profiles['B_z']*np.sqrt(1-profiles['beta']), c='r', label='B tot')
    # plt.xlabel('Axial Position z [m]')
    # plt.ylabel('$B_{z,tot}$ [T]')
    # plt.ylim(0, 30)
    # plt.legend()
    # plt.title('1D Equilibrium Solution for each iteration')
    # plt.show()

    # Calculate Fusion power

    print(f"n0 = {profiles['n'].to_numpy()[0]}")
    print(f"B0 = {profiles['B_z'].to_numpy()[0]:.1} T")
    print(f"a0 = {profiles['a'].to_numpy()[0]:.3} m")
    print(f"S_fus,0 = {profiles['S_fus'].to_numpy()[0]:.2f} MW/m^3")

    print("\nVolume integrated quantities:")
    print(f"P_fus = {get_fusion_power(profiles):.2f} MW")
    P_nbi = get_required_nbi_power(profiles, Rm_vac=Rm_vac, E_NBI_keV=E_NBI_keV, n0=n0)
    print(f"P_nbi = {P_nbi:.2f} MW")
    print(f"Mass of tritium in plasma = {get_tritium_mass_in_plasma(profiles):.2e} g")
    print(f"Mass current of tritium through NBI: {get_tritium_mass_current(pnbi=P_nbi, Enbi_keV=E_NBI_keV)} g/s")
    print(f'Plasma Volume = {get_plasma_volume(profiles)} m^3')
    print(f'Plasma surface area = {get_plasma_surface_area(profiles)} m^2')
    print(f'Wall surface area = {get_wall_surface_area(profiles)} m^2')

    # Tritium current:

    profiles = pd.read_csv('/Users/henrycw/projects/alpha-mirror/equilibrium/axial_profiles_final_nwl.csv')
    profiles_left = profiles.copy()
    profiles_left['z'] = -1*profiles_left['z']
    profiles = pd.concat([profiles, profiles_left]).sort_values(by='z')

    # Plot of profiles
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(6,5))
    for ax in axs:
        ax.tick_params(axis='both', labelsize=13)
    axs[0].plot(profiles['z'], profiles['B_z'], c='k')
    axs[0].set_ylim(0, 30)
    axs[0].set_ylabel('$B_z$ [T]', fontsize=14)
    axs[0].set_yticks(np.arange(0, 40, 10))
    axs[0].set_title('Net Axial Magnetic Field', fontsize=16)
    # axs[1].plot(profiles['z'], profiles['a'], c='k')
    # axs[1].set_ylim(0, 1.2*np.max(profiles['a']))
    # axs[1].set_yticks(np.arange(0, 0.16, 0.04))
    # axs[1].set_ylabel('a [m]', fontsize=14)
    # axs[1].set_title('Plasma Radius', fontsize=16)
    axs[1].plot(profiles['z'], profiles['S_fus'], c='b')
    axs[1].set_ylabel('$S_{fus}$ [MW/m$^3$]', fontsize=14)
    axs[1].set_ylim(0, 1.2*np.max(profiles['S_fus']))
    axs[1].set_yticks(np.arange(0, 60, 20))
    axs[1].set_title('Fusion Power Density', fontsize=16)
    axs[2].plot(profiles['z'], profiles['nwl'], c='r')
    axs[2].set_title('Neutron Wall Loading (OpenMC)', fontsize=16)
    axs[2].set_ylabel("MW/m$^2$", fontsize=14)
    axs[2].set_yticks(np.arange(0, 2.0, 0.5))
    axs[2].axvspan(-1.2, -0.4, color='tab:orange', alpha=0.2, zorder=0)
    axs[2].axvspan(-0.4, 1.2, color='tab:purple', alpha=0.2, zorder=0)
    axs[-1].set_xlabel('Z [m]', fontsize=14)
    axs[0].set_xlim(-1.5, 1.5)
    plt.tight_layout()
    plt.savefig('equilibrium_profiles.png')
    plt.show()




