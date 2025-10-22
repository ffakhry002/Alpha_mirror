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
ETA_ABS = 0.9   # Absorption efficiency
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


def calculate_a0_FLR(E_b_100keV, B_0, N_25=1.0):
    """
    a₀,FLR = N₂₅√E_b / B₀

    Minimum radius for adiabaticity and DCLC suppression (BEAM Eq. 2.4)
    """
    a_0_FLR = N_25 * np.sqrt(E_b_100keV) / B_0
    return a_0_FLR


def calculate_a0_FLR_at_mirror(E_b_100keV, B_mirror, N_25=1.0):
    """
    a₀,FLR at mirror field (for frustum geometry)
    """
    a_0_FLR_mirror = N_25 * np.sqrt(E_b_100keV) / B_mirror
    return a_0_FLR_mirror


def calculate_plasma_geometry_frustum(a_0_min, a_0_FLR_mirror, N_rho=25.0):
    """
    Three-segment geometry: frustum-cylinder-frustum
    10% standoff: vessel radius = 1.1 × plasma radius
    """
    # Total length
    L_plasma = N_rho * a_0_min

    # Each segment is L/3
    L_segment = L_plasma / 3

    # Volume of one frustum (plasma only, no standoff in volume)
    V_frustum = (1/3) * np.pi * L_segment * (
        a_0_min**2 + a_0_FLR_mirror**2 + a_0_min * a_0_FLR_mirror
    )

    # Volume of cylinder (plasma only)
    V_cylinder = np.pi * a_0_min**2 * L_segment

    # Total plasma volume
    V_plasma = V_cylinder + 2 * V_frustum

    # Surface area - 10% standoff applied to radii
    a0_min_vessel = 1.1 * a_0_min
    a0_FLR_vessel = 1.1 * a_0_FLR_mirror

    slant_height = np.sqrt(L_segment**2 + (a0_min_vessel - a0_FLR_vessel)**2)
    A_frustum = np.pi * (a0_min_vessel + a0_FLR_vessel) * slant_height
    A_cylinder = 2 * np.pi * a0_min_vessel * L_segment
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
    # Get ion-ion collision frequency
    nu_ii = np.sqrt(const.m_e / mass_T) * calculate_electron_ion_collision_freq(E_b_100keV, n_20)
    return nu_ii * L_plasma / v_ti



# ============================================================================
# POWER CALCULATIONS
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

    P_NBI = (1/efficiency) * 1.6 * n_20**2 * V_plasma / (C_loss * np.sqrt(E_b_100keV) * np.log10(Rm_vac))

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
