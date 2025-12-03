"""equations.py - All physics equations for mirror fusion analysis
Updated to use energy-dependent loss coefficient with double interpolation
"""

import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
import scipy.constants as const
import pandas as pd

# ============================================================================
# PHYSICS CONSTANTS
# ============================================================================

# Neutron energy fraction (14.1 MeV out of 17.6 MeV total from D-T fusion)
NEUTRON_FRACTION = 14.1 / 17.6

# NBI efficiency (absorption × charge exchange)
# TODO: These should be functions of beam energy
ETA_ABS = 0.9   # Absorption efficiency
ETA_HEAT = 0.9 # NBI heating efficiency (just absorption)
NBI_EFFICIENCY = ETA_ABS

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_dt_reactivity_data():
    """Load D-T fusion reactivity ⟨σv⟩(T) from FusionReactivities.dat"""
    data_file = Path(__file__).parent / 'FusionReactivities.dat'

    temperatures = []  # keV
    dt_reactivities = []  # m³/s

    try:
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    try:
                        temp = float(parts[0])
                        dt_reactivity = float(parts[1])
                        temperatures.append(temp)
                        dt_reactivities.append(dt_reactivity)
                    except ValueError:
                        continue

        temperatures = np.array(temperatures)
        dt_reactivities = np.array(dt_reactivities)

        print(f"✓ Loaded {len(temperatures)} D-T reactivity data points")
        print(f"  Temperature range: {temperatures.min():.1f} - {temperatures.max():.1f} keV")

        interpolator = interp1d(temperatures, dt_reactivities,
                               kind='linear', bounds_error=False,
                               fill_value='extrapolate')

        return interpolator

    except Exception as e:
        print(f"⚠ Could not load D-T reactivity data: {e}")
        return None

dt_reactivity_interp = load_dt_reactivity_data()

# ============================================================================
# FUSION REACTIVITY
# ============================================================================

def get_dt_reactivity(T_keV):
    """Get ⟨σv⟩(T_keV) [m³/s]"""
    if dt_reactivity_interp is None:
        raise ValueError("D-T reactivity data not loaded")

    result = dt_reactivity_interp(T_keV)

    if np.isscalar(T_keV):
        result = float(result)

    return result


# ============================================================================
# LOSS COEFFICIENT - DOUBLE INTERPOLATION METHOD
# ============================================================================

def calculate_loss_coefficient(E_b_100keV, Rm_vac):

    E_b_keV = E_b_100keV * 100  # Convert to keV

    # C = c0_constant + c1_E_b * E_b_keV + c2_log10_Rm * np.log10(Rm_vac) + \
    #     c3_E_b_squared * (E_b_keV**2) + c4_log10_Rm_squared * (np.log10(Rm_vac)**2) + \
    #     c5_E_b_times_log10_Rm * (E_b_keV * np.log10(Rm_vac))
    C_45deg = (
        0.06873579466429273
        + 0.00020090892589799344 * E_b_keV
        + 0.09308427339368963 * np.log10(Rm_vac)
        - 1.1689907326070077e-06 * (E_b_keV ** 2)
        - 0.028641049003165112 * (np.log10(Rm_vac) ** 2)
        + 0.000180737335795075 * (E_b_keV * np.log10(Rm_vac))
    )
    return C_45deg


# ============================================================================
# BETA AND MAGNETIC FIELD (CORRECT DIAMAGNETIC FORMULATION)
# ============================================================================

def calculate_beta_local(n_20, E_b_100keV, B_central):
    """
    β = 1/2 - 1/2 √(1 - 12n₂₀E_b/B_central²)

    Quadratic solution from LaTeX document that properly accounts for diamagnetism.
    Diamagnetic currents REDUCE the on-axis field: B₀ = B_central √(1-β)

    Returns None if the discriminant is negative (physically impossible)
    """
    discriminant = 1 - (12 * n_20 * E_b_100keV) / (B_central**2)

    # Check if physically possible
    if np.any(discriminant < 0):
        # If array, set negative discriminants to NaN
        if isinstance(discriminant, np.ndarray):
            beta = 0.5 - 0.5 * np.sqrt(np.maximum(discriminant, 0))
            beta[discriminant < 0] = np.nan
            return beta
        else:
            return None

    beta = 0.5 - 0.5 * np.sqrt(discriminant)
    return beta


def calculate_B0_with_diamagnetic(B_central, beta_local):
    """
    B₀ = B_central × √(1 - β)

    CRITICAL: Diamagnetic effect REDUCES the on-axis field!
    The plasma diamagnetic current weakens the central field.
    """
    B_0 = B_central * np.sqrt(1 - beta_local)
    return B_0


def calculate_beta_limit(E_b_100keV, B_central, beta_c):
    """
    n₂₀,max = (B_central²/12E_b) × [1 - (1-2β_c)²]

    Maximum density from beta limit at β = β_c
    """
    n_20_max = (B_central**2 / (12 * E_b_100keV)) * (1 - (1 - 2*beta_c)**2)
    return n_20_max


# ============================================================================
# GEOMETRY CONSTRAINTS
# ============================================================================

def calculate_a0_absorption(E_b_100keV, n_20):
    """
    a₀,abs = 0.3√E_b / n₂₀

    Minimum radius for 90% beam absorption (from BEAM Figure 4)
    """
    a_0_abs = 0.3 * np.sqrt(E_b_100keV) / n_20
    return a_0_abs


def calculate_ion_larmor_radius(E_b_100keV, B_0):
    """
    ρᵢ = 0.04√E_b / B₀

    Ion Larmor radius for NBI-fueled mirror with proper velocity distribution.
    Prefactor 0.04 accounts for 45° injection and thermalization (Forest et al.)
    """
    rho_i = 0.04 * np.sqrt(E_b_100keV) / B_0
    return rho_i


def calculate_a0_DCLC(E_b_100keV, B_0):
    """
    a₀,DCLC = 25ρᵢ

    Minimum radius for DCLC (Drift-Cyclotron Loss Cone) stabilization.
    Conservative limit ensures kinetic stability even with sloshing ions.
    """
    rho_i = calculate_ion_larmor_radius(E_b_100keV, B_0)
    return 25 * rho_i

def calculate_a0_adiabaticity(E_b_100keV, B_0, beta):
    """
    a₀,adiabatic = 50ρᵢ(1 - √(1-β))

    Minimum radius from fast ion adiabaticity constraint (L_B/ρᵢ > 10).
    Note: Never limiting since at β_c=0.5, factor ≈ 14.65 < 25 (DCLC dominates).
    """
    rho_i = calculate_ion_larmor_radius(E_b_100keV, B_0)
    return 50 * rho_i * (1 - np.sqrt(1 - beta))


def calculate_a0_end(a_0_center, B_0, B_mirror):
    """
    a₀,end = a₀,center × √(B₀/B_mirror)

    Mirror throat radius from flux conservation.
    """
    a_0_end = a_0_center * np.sqrt(B_0 / B_mirror)
    return a_0_end


def calculate_plasma_geometry_frustum(a_0_min, a_0_end, E_b_100keV, B_0):
    """
    Three-segment geometry: frustum-cylinder-frustum
    Constant standoff: 0.1 × a_0_min absolute gap at all axial positions

    Length constraint from FLR stability: L ≥ a²/ρᵢ
    """
    # Total length from FLR stability: L = a² / rho_i
    rho_i = calculate_ion_larmor_radius(E_b_100keV, B_0)
    L_plasma = a_0_min**2 / rho_i

    # Each segment is L/3
    L_segment = L_plasma / 3

    # Volume of one frustum (plasma only, no standoff in volume)
    V_frustum = (1/3) * np.pi * L_segment * (
        a_0_min**2 + a_0_end**2 + a_0_min * a_0_end
    )

    # Volume of cylinder (plasma only)
    V_cylinder = np.pi * a_0_min**2 * L_segment

    # Total plasma volume
    V_plasma = V_cylinder + 2 * V_frustum

    # Surface area - CONSTANT absolute standoff of 0.1 × a_0_min everywhere
    standoff = 0.1 * a_0_min  # Constant absolute gap
    a0_center_vessel = a_0_min + standoff  # = 1.1 × a_0_min
    a0_end_vessel = a_0_end + standoff     # NOT proportional to a_0_end

    # Frustum lateral surface: π(R+r)√[h² + (R-r)²]
    slant_height = np.sqrt(L_segment**2 + (a0_center_vessel - a0_end_vessel)**2)
    A_frustum = np.pi * (a0_center_vessel + a0_end_vessel) * slant_height

    # Cylinder lateral surface: 2πRh
    A_cylinder = 2 * np.pi * a0_center_vessel * L_segment

    # Total vessel surface area
    vessel_surface_area = 2 * A_frustum + A_cylinder

    return L_plasma, V_plasma, vessel_surface_area

# ============================================================================
# COLLISIONAL QUANTITIES
# ============================================================================

def calculate_coulomb_logarithm(E_b_100keV, n_20):
    """
    Calculates the Coulomb logarithm from equation 9.36 in Freidberg
    Fusion Energy. Maxwellian approximation for ions and electrons from Egedal22
    """
    # Electron temperature is roughly 1/10th the injected beam energy
    Te_keV = 10 * E_b_100keV
    return np.log(4.9e7 * Te_keV**(3/2) / np.sqrt(n_20))

def calculate_electron_ion_collision_freq(E_b_100keV, n_20):
    """
    Calculates the electron ion collion frequency from Eq. 9.51
    in Freidberg Fusion Energy
    """
    # Use handy formula for electron plasma frequency (Chen 4.26)
    omega_pe = 1.8e11*np.pi * np.sqrt(n_20)
    log_lambda = calculate_coulomb_logarithm(E_b_100keV, n_20)
    return omega_pe*log_lambda / (np.sqrt(3)*np.exp(log_lambda))

def calculate_collisionality(E_b_100keV, n_20, L_plasma):
    """
    The collisionality nu_* = nu_ii * L / v_ti is the ratio of the
    time it takes an ion to travel along the mirror machine vs the time it takes
    to undergo a net 90 degree collision, evaluated at the effective beam temperature.
    We use the ion thermal speed, not the sound speed, since Ti >> Te.
    This also assumes the pitch-angle scattering frequency is similar to the collision frequency
    Sources:
    - Egedal et al, Nucl. Fusion, 2022, Sec 3.1
    - Schwartz et al, J. Plasma Phys., 2024, Sec 2.3
    """
    # Get triutium ion thermal velocity along magnetic field
    Ti_joule = 2/3 * E_b_100keV * 1e5 * const.e
    mass_T = 3.016 * const.atomic_mass # [kg]
    v_ti = np.sqrt(Ti_joule / mass_T) # [m/s]
    # Get ion-ion collision frequency using 9.52 in Friedberg and Te/Ti = 0.15
    nu_ii = 1.2*np.sqrt(const.m_e / mass_T)*(0.15)**(3/2) * calculate_electron_ion_collision_freq(E_b_100keV, n_20)
    return nu_ii * L_plasma / v_ti

# ============================================================================
# VORTEX STABILIZATION QUANTITIES
# ============================================================================

def calculate_ion_sound_gyroradius(E_b_100keV, B0):
    """
    Returns the ion Larmor radius, using the sound speed instead of
    the thermal speed. Units in meters
    """
    ion_mass_eff = (2.014+3.016)/2 * const.atomic_mass # [kg]
    Te = 0.1*E_b_100keV*1e5*const.e # [J]
    return np.sqrt(ion_mass_eff * Te) / (const.e * B0)

def calculate_vortex_length_scales(L_plasma):
    """
    L_ρ = L/2,  L_κ = L/4

    Vortex stabilization length scales from LaTeX document.
    L_ρ: plasma half length from density considerations
    L_κ: curvature length scale (L_κ = L_ρ/2)

    Returns (L_rho, L_kappa)
    """
    L_rho = L_plasma / 2
    L_kappa = L_plasma / 4
    return L_rho, L_kappa

def calculate_voltage_field_reversal(E_b_100keV, B0, a_0_min, L_plasma, Rm_diamag):
    """
    (e*φ/Te) > 7*R_eff * (ρ*/a)² * (L_ρ/L_κ) * √(Ti/Te + 1)

    Minimum bias potential for vortex stabilization (field reversal requirement).
    With L_ρ = L/2 and L_κ = L/4, the ratio L_ρ/L_κ = 2.
    Note: R_eff = tau_p * c_s / L_rho for classical; using Rm_diamag (gas dynamic).
    Sources: Beklemishev 2010 Eq. 23, Endrizzi 2023 Eq. 3.9
    """
    # Length scales from LaTeX
    L_rho, L_kappa = calculate_vortex_length_scales(L_plasma)

    # Ion sound gyroradius
    sound_gyrorad = calculate_ion_sound_gyroradius(E_b_100keV, B0)

    # Temperature ratio from Egedal
    ti_te_ratio = 20/3  # Ti/Te = (2/3*Eb)/(0.1*Eb) = 20/3

    # Field reversal voltage requirement
    voltage = 7 * Rm_diamag * (sound_gyrorad / a_0_min)**2
    voltage *= (L_rho / L_kappa) * np.sqrt(ti_te_ratio + 1)

    return voltage

def calculate_voltage_closed_lines(E_b_100keV, B0, a_0_min, L_plasma, Rm_diamag):
    """
    (e*φ/Te) > 4*R_eff² * (ρ*³ * L_ρ²)/(a² * L_κ³) * (Ti/Te + 1)^(3/2)

    Bias potential for vortex stabilization (closed flow lines requirement).
    With L_ρ = L/2 and L_κ = L/4. GDT found e*phi/Te ~ 1 works best in practice.
    Note: R_eff = tau_p * c_s / L_rho for classical; using Rm_diamag (gas dynamic).
    Sources: Beklemishev 2010 Eq. 20, Endrizzi 2023 Eq. 3.8
    """
    # Length scales from LaTeX
    L_rho, L_kappa = calculate_vortex_length_scales(L_plasma)

    # Temperature ratio from Egedal
    ti_te_ratio = 20/3  # Ti/Te = (2/3*Eb)/(0.1*Eb) = 20/3

    # Ion sound gyroradius
    sound_gyrorad = calculate_ion_sound_gyroradius(E_b_100keV, B0)

    # Closed flow lines voltage requirement
    voltage = 4 * Rm_diamag**2 * (ti_te_ratio + 1)**(3/2)
    voltage *= sound_gyrorad**3 * L_rho**2 / (a_0_min**2 * L_kappa**3)

    return voltage

def calculate_max_mirror_ratio_vortex(E_b_100keV, B0, a_0_min, L_plasma):
    """
    R_m,max = 0.7 * (a/ρ*)² * (L_κ/L_ρ) / √(Ti/Te + 1)

    Maximum mirror ratio for vortex stabilization (vortex flow radius < plasma radius).
    With L_ρ = L/2 and L_κ = L/4, the ratio L_κ/L_ρ = 0.5.
    Sources: Beklemishev 2010 Eq. 22, Endrizzi 2023 Eq. 3.9
    """
    # Length scales from LaTeX
    L_rho, L_kappa = calculate_vortex_length_scales(L_plasma)

    # Temperature ratio from Egedal
    ti_te_ratio = 20/3  # Ti/Te = (2/3*Eb)/(0.1*Eb) = 20/3

    # Ion sound gyroradius
    sound_gyrorad = calculate_ion_sound_gyroradius(E_b_100keV, B0)

    # Maximum mirror ratio for vortex stabilization (Eq. 3.9 in Endrizzi)
    R_m_max = 0.7 * (a_0_min / sound_gyrorad)**2 * (L_kappa / L_rho) / np.sqrt(ti_te_ratio + 1)

    return R_m_max

# ============================================================================
# POWER CALCULATION
# ============================================================================

def calculate_fusion_power(E_b_100keV, n_20, V_plasma, T_i_keV):
    """
    P_fus = 7.04×10²¹ n₂₀² V ⟨σv⟩ [MW]

    From LaTeX document Section 1.5
    """
    sigma_v = get_dt_reactivity(T_i_keV)
    P_fusion = 7.04e21 * n_20**2 * V_plasma * sigma_v

    if np.isscalar(E_b_100keV) and np.isscalar(n_20) and np.isscalar(V_plasma) and np.isscalar(T_i_keV):
        P_fusion = float(P_fusion)

    return P_fusion


def calculate_NBI_power(n_20, V_plasma, E_b_100keV, Rm_vac, C_loss=None, efficiency=NBI_EFFICIENCY):
    """
    P_NBI = (1/η) × 1.6 n₂₀² V / (C√E_b log₁₀Rm_vac) [MW]

    From LaTeX document Section 1.6
    """
    if C_loss is None:
        C_loss = calculate_loss_coefficient(E_b_100keV, Rm_vac)

    P_NBI = 2*(1/efficiency) * 1.6 * n_20**2 * V_plasma / (C_loss * np.sqrt(E_b_100keV) * np.log10(Rm_vac))

    if np.isscalar(n_20) and np.isscalar(V_plasma) and np.isscalar(E_b_100keV):
        P_NBI = float(P_NBI)

    return P_NBI


def calculate_NWL(P_fusion, vessel_surface_area, neutron_fraction=NEUTRON_FRACTION):
    """
    NWL = P_fus × (14.1/17.6) / A_vessel [MW/m²]

    From LaTeX document Section 1.8
    """
    NWL = P_fusion * neutron_fraction / vessel_surface_area

    if np.isscalar(P_fusion) and np.isscalar(vessel_surface_area):
        NWL = float(NWL)

    return NWL


def calculate_Q(P_fusion, P_NBI):
    """
    Q = P_fus / P_NBI

    From LaTeX document Section 1.7
    """
    Q = P_fusion / P_NBI

    if np.isscalar(P_fusion) and np.isscalar(P_NBI):
        Q = float(Q)

    return Q

# ============================================================================
# CAPACITY FACTOR AND GRID LIFETIME
# ============================================================================

def calculate_sputtering_yield_Mo(E_b_keV):
    """
    Sputtering yield for molybdenum grids from D-T ions

    Y_Mo(E_b) = 0.1596 × E_b^(-0.915)

    Parameters
    ----------
    E_b_keV : float or array
        Beam energy [keV] (NOT 100 keV units)

    Returns
    -------
    Y : float or array
        Sputtering yield [atoms/ion]
    """
    return 0.1596 * E_b_keV**(-0.915)


def calculate_grid_lifetime(E_b_keV, P_NBI_MW, d_mm=3.0, sigma_x_cm=4.3,
                           sigma_y_cm=10.77, num_grids=8, A_over_rho=9.4):
    """
    Calculate grid lifetime from sputtering erosion

    t_grid = d / V_e

    where erosion velocity:
    V_e = 10^-11 × j_0 × Y(E_b) × (A/ρ)  [m/s]

    and current density at beam center:
    j_0 = I / (2π σ_x σ_y) = (P_NBI/num_grids) / (E_b × 2π σ_x σ_y)  [A/m²]

    Parameters
    ----------
    E_b_keV : float or array
        Beam energy [keV] (NOT 100 keV units)
    P_NBI_MW : float or array
        NBI power [MW]
    d_mm : float
        Grid thickness erosion limit [mm] (default: 3.0)
    sigma_x_cm : float
        Horizontal Gaussian beam width [cm] (default: 4.3)
    sigma_y_cm : float
        Vertical Gaussian beam width [cm] (default: 10.77)
    num_grids : int
        Number of grids to spread power over (default: 8)
    A_over_rho : float
        Atomic mass / density for target material [cm³/mol]
        For Mo: A=96 g/mol, ρ=10.2 g/cm³ → A/ρ=9.4 (default)

    Returns
    -------
    t_grid_hrs : float or array
        Grid lifetime [hours]
    """
    # Unit conversions
    d_m = d_mm * 1e-3                    # mm → m
    sigma_x_m = sigma_x_cm * 1e-2        # cm → m
    sigma_y_m = sigma_y_cm * 1e-2        # cm → m
    P_NBI_W = P_NBI_MW * 1e6             # MW → W
    E_b_V = E_b_keV * 1e3                # keV → V

    # Power per grid
    P_per_grid = P_NBI_W / num_grids     # W

    # Current per grid (I = P/V)
    I_per_grid = P_per_grid / E_b_V      # A

    # Peak current density (Gaussian beam, elliptical)
    j0 = I_per_grid / (2 * np.pi * sigma_x_m * sigma_y_m)  # A/m²

    # Sputtering yield
    Y = calculate_sputtering_yield_Mo(E_b_keV)  # atoms/ion

    # Erosion velocity (Whyte formula)
    # V_e [m/s] = 10^-11 × j [A/m²] × Y × (A/ρ) [cm³/mol]
    V_e = 1e-11 * j0 * Y * A_over_rho    # m/s

    # Grid lifetime
    t_grid_s = d_m / V_e                 # seconds
    t_grid_hrs = t_grid_s / 3600         # hours

    if np.isscalar(E_b_keV) and np.isscalar(P_NBI_MW):
        t_grid_hrs = float(t_grid_hrs)

    return t_grid_hrs


def calculate_capacity_factor_grid(t_grid_hrs):
    """
    Grid lifetime capacity factor (fraction of year before replacement)

    CF_grid = t_grid / 8760 hrs

    Parameters
    ----------
    t_grid_hrs : float or array
        Grid lifetime [hours]

    Returns
    -------
    CF_grid : float or array
        Grid lifetime capacity factor [dimensionless]
    """
    hours_per_year = 8760
    CF_grid = t_grid_hrs / hours_per_year

    if np.isscalar(t_grid_hrs):
        CF_grid = float(CF_grid)

    return CF_grid


def calculate_capacity_factor_annual(t_grid_hrs, t_replace_months=2.5,
                                    eta_duty=1.0):
    """
    Annual capacity factor accounting for replacement downtime

    For continuous operation (eta_duty = 1.0):
        CF_annual = t_grid / (t_grid + t_replace)

    For weekly duty cycle (e.g., eta_duty = 6/7):
        CF_annual = t_grid / (t_grid/eta_duty + t_replace)

    Parameters
    ----------
    t_grid_hrs : float or array
        Grid lifetime [hours]
    t_replace_months : float
        Replacement downtime [months] (default: 2.5)
    eta_duty : float
        Duty cycle fraction (default: 1.0 for continuous, 6/7 for 6 days/week)

    Returns
    -------
    CF_annual : float or array
        Annual capacity factor [dimensionless]
    """
    # Convert replacement time to hours
    t_replace_hrs = t_replace_months * (8760 / 12)  # months → hours

    # Calculate annual capacity factor
    if eta_duty >= 0.999:  # Treat as continuous
        CF_annual = t_grid_hrs / (t_grid_hrs + t_replace_hrs)
    else:  # Weekly duty cycle
        CF_annual = t_grid_hrs / (t_grid_hrs / eta_duty + t_replace_hrs)

    if np.isscalar(t_grid_hrs):
        CF_annual = float(CF_annual)

    return CF_annual


def calculate_average_fusion_power(P_fusion_MW, t_grid_hrs,
                                  t_replace_months=2.5, eta_duty=1.0):
    """
    Time-averaged fusion power accounting for downtime

    ⟨P_fus⟩ = P_fus × CF_annual

    Parameters
    ----------
    P_fusion_MW : float or array
        Instantaneous fusion power [MW]
    t_grid_hrs : float or array
        Grid lifetime [hours]
    t_replace_months : float
        Replacement downtime [months] (default: 2.5)
    eta_duty : float
        Duty cycle fraction (default: 1.0 for continuous, 6/7 for 6 days/week)

    Returns
    -------
    P_fus_avg : float or array
        Time-averaged fusion power [MW]
    """
    CF_annual = calculate_capacity_factor_annual(t_grid_hrs, t_replace_months, eta_duty)
    P_fus_avg = P_fusion_MW * CF_annual

    if np.isscalar(P_fusion_MW) and np.isscalar(t_grid_hrs):
        P_fus_avg = float(P_fus_avg)

    return P_fus_avg


# ============================================================================
# END PLUG HEAT FLUX QUANTITIES
# ============================================================================

def calculate_Bw(E_b_100keV, B0, a_0_min, Nwall=1):
    """
    Returns the magnetic field strength [T] at the end-plug wall based on
    flux expansion and constraints on adiabadicity
    """
    return 7.3e-3 * Nwall**2 * E_b_100keV / (a_0_min**2 * B0)

def calculate_a_w(a_0_min, B0, Bw):
    """
    Calculate the radius (and length) of the end-plug [m]
    """
    return a_0_min * np.sqrt(B0/Bw)


def calculate_heat_flux(P_nbi, Q, a_0_min, B0, Bw):
    """
    Returns the heat flux at each end cell in MW/m^2
    """
    power_in = ETA_HEAT * P_nbi * (1 + Q/5)
    wetted_area = 2*np.pi * a_0_min**2 * B0 / Bw
    return power_in / wetted_area


# ============================================================================
# TAPE REQUIREMENTS (SIMPLE 3-COIL MODEL)
# ============================================================================

def tape_req_simple(a_0_min, a_0_FLR_mirror, B_central, B_max,
                    r_baker=0.88, r_shield=0.6):
    """
    Calculate HTS tape requirements for a simple 3-coil model:
    - 2 end coils (at mirror field)
    - 1 central coil

    All coils are circular rings.

    Parameters
    ----------
    a_0_min : float
        Minimum plasma radius at center [m]
    a_0_FLR_mirror : float
        Plasma radius at mirror (from FLR constraint) [m]
    B_central : float
        Magnetic field at center [T]
    B_max : float
        Maximum magnetic field at mirror [T]
    r_baker : float, optional
        Breeder + shield thickness for central coil [m] (default: 0.88)
    r_shield : float, optional
        Shield thickness for end coils [m] (default: 0.6)

    Returns
    -------
    dict
        Dictionary containing:
        - 'kAm_central': Central coil tape requirement [kA-m]
        - 'kAm_end_single': Single end coil tape requirement [kA-m]
        - 'kAm_end_total': Total for both end coils [kA-m]
        - 'kAm_total': Total tape requirement [kA-m]
        - 'R_central': Central coil radius [m]
        - 'R_end': End coil radius [m]
    """
    mu_0 = 4 * np.pi * 1e-7  # H/m

    # Central coil: uses a_0_min and B_central
    # Radius = 1.1 * plasma_radius + baker + shield
    R_central = 1.1 * a_0_min + r_baker
    Am_central = (4 * np.pi * R_central**2 * B_central) / mu_0
    kAm_central = Am_central / 1000

    # End coils: use a_0_FLR_mirror and B_max
    # Radius = 1.1 * plasma_radius + shield
    R_end = 1.1 * a_0_FLR_mirror + r_shield
    Am_end_single = (4 * np.pi * R_end**2 * B_max) / mu_0
    kAm_end_single = Am_end_single / 1000
    kAm_end_total = 2 * kAm_end_single

    # Total tape requirement
    kAm_total = kAm_central + kAm_end_total

    return {
        'kAm_central': float(kAm_central),
        'kAm_end_single': float(kAm_end_single),
        'kAm_end_total': float(kAm_end_total),
        'kAm_total': float(kAm_total),
        'R_central': float(R_central),
        'R_end': float(R_end)
    }


# ============================================================================
# ISOTOPE REVENUE WATERFALL
# ============================================================================

# Constants for isotope production
AVOGADRO = 6.022e23  # atoms/mol
SECONDS_PER_YEAR = 3600 * 24 * 365  # s/yr
NEUTRONS_PER_MW_PER_S = 1e6 / (17.6 * 1.6e-13)  # ~3.55e17 n/s/MW fusion

# Isotope data: sorted by $/MW/yr (most valuable first)
# Values from corrected financial model spreadsheet
ISOTOPES = [
    {"name": "Cu-64",  "mass": 64,  "atoms_per_n": 0.0065,      "value_per_kg": 40e12,  "max_demand": 20e6},
    {"name": "Cu-67",  "mass": 67,  "atoms_per_n": 0.000756825, "value_per_kg": 75e12,  "max_demand": 100e6},
    {"name": "Lu-177", "mass": 177, "atoms_per_n": 1.4e-05,     "value_per_kg": 3e12,   "max_demand": 200e6},
    {"name": "Mo-99",  "mass": 99,  "atoms_per_n": 5.92e-06,    "value_per_kg": 50e9,   "max_demand": 500e6},
]


def _init_isotope_economics():
    """Initialize isotope economic parameters (called once at import)"""
    neutrons_per_MW_per_yr = NEUTRONS_PER_MW_PER_S * SECONDS_PER_YEAR
    for iso in ISOTOPES:
        atoms_per_MW_per_yr = iso["atoms_per_n"] * neutrons_per_MW_per_yr
        kg_per_MW_per_yr = (atoms_per_MW_per_yr / AVOGADRO) * (iso["mass"] / 1000)
        iso["kg_per_MW_yr"] = kg_per_MW_per_yr
        iso["usd_per_MW_yr"] = kg_per_MW_per_yr * iso["value_per_kg"]
        iso["MW_to_saturate"] = iso["max_demand"] / iso["usd_per_MW_yr"]
    # Sort by $/MW/yr descending
    ISOTOPES.sort(key=lambda x: x["usd_per_MW_yr"], reverse=True)


_init_isotope_economics()


def calculate_isotope_revenue(P_fus_MW):
    """
    Calculate annual isotope revenue using waterfall allocation.

    Allocates fusion power to isotopes in order of $/MW/yr (most valuable first).
    Each isotope is filled until its market is saturated, then moves to next.

    Parameters
    ----------
    P_fus_MW : float or array
        Fusion power [MW] - should be capacity factor adjusted <P_fus>

    Returns
    -------
    revenue : float or array
        Annual revenue [$/yr]
    """
    scalar_input = np.isscalar(P_fus_MW)
    P_fus_MW = np.atleast_1d(P_fus_MW).astype(float)
    revenue = np.zeros_like(P_fus_MW, dtype=float)

    for i, P_fus in enumerate(P_fus_MW.flat):
        if np.isnan(P_fus) or P_fus <= 0:
            revenue.flat[i] = np.nan
            continue

        MW_remaining = P_fus
        total_rev = 0.0

        for iso in ISOTOPES:
            if MW_remaining <= 0:
                break

            # Allocate MW to this isotope (up to saturation)
            MW_for_this = min(MW_remaining, iso["MW_to_saturate"])

            # Revenue from this isotope (capped at market size)
            rev_from_this = min(MW_for_this * iso["usd_per_MW_yr"], iso["max_demand"])
            total_rev += rev_from_this

            # Subtract MW used
            MW_remaining -= iso["MW_to_saturate"]

        revenue.flat[i] = total_rev

    if scalar_input:
        return float(revenue.flat[0])
    return revenue.reshape(P_fus_MW.shape)


def calculate_revenue_per_volume(P_fus_MW, V_plasma_m3):
    """
    Calculate revenue per unit volume (economic figure of merit).

    Parameters
    ----------
    P_fus_MW : float or array
        Fusion power [MW] - should be capacity factor adjusted <P_fus>
    V_plasma_m3 : float or array
        Plasma volume [m³]

    Returns
    -------
    rev_per_vol : float or array
        Revenue per volume [$/yr/m³]
    """
    revenue = calculate_isotope_revenue(P_fus_MW)
    rev_per_vol = revenue / V_plasma_m3
    return rev_per_vol
