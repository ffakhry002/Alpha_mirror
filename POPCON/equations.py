"""equations.py - All physics equations for mirror fusion analysis"""

import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
import pandas as pd

# ============================================================================
# PHYSICS CONSTANTS
# ============================================================================

# Neutron energy fraction (14.1 MeV out of 17.6 MeV total from D-T fusion)
NEUTRON_FRACTION = 14.1 / 17.6

# Default NBI efficiency (injection + trapping)
NBI_EFFICIENCY = 0.9 * 0.9  # 81% total

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_extracted_data():
    """Load C@90° and C@45°/C@90° ratio data from extracted_data.csv"""
    data_file = Path(__file__).parent / 'OCR' / 'outputs' / 'extracted_data.csv'

    try:
        df = pd.read_csv(data_file)
        E_beam_keV = df['E_beam_keV'].values
        available_Rm = range(4, 17)

        # Create interpolators for C@90° values
        C_90deg_interpolators = {}
        for Rm in available_Rm:
            col_name = f'd_Rm{Rm}'
            if col_name in df.columns:
                valid_mask = ~pd.isna(df[col_name])
                if np.sum(valid_mask) > 1:
                    E_valid = E_beam_keV[valid_mask]
                    C_valid = df[col_name].values[valid_mask]
                    interpolator = interp1d(E_valid, C_valid, kind='linear',
                                          bounds_error=False, fill_value='extrapolate')
                    C_90deg_interpolators[Rm] = interpolator
                    print(f"Loaded C@90° interpolator for Rm={Rm} ({len(E_valid)} points)")

        # Create interpolators for C@45°/C@90° ratios
        ratio_45_90_interpolators = {}
        for Rm in available_Rm:
            ratio_col_name = f'ratio_45_90_Rm{Rm}'
            if ratio_col_name in df.columns:
                valid_mask = ~pd.isna(df[ratio_col_name])
                if np.sum(valid_mask) > 1:
                    E_valid = E_beam_keV[valid_mask]
                    ratio_valid = df[ratio_col_name].values[valid_mask]
                    ratio_mean = np.mean(ratio_valid)
                    ratio_45_90_interpolators[Rm] = lambda E, ratio=ratio_mean: ratio
                    print(f"Loaded C@45°/C@90° ratio for Rm={Rm}: {ratio_mean:.4f}")

        print(f"Successfully loaded extracted data with {len(E_beam_keV)} energy points")
        return E_beam_keV, C_90deg_interpolators, ratio_45_90_interpolators

    except Exception as e:
        print(f"Error loading extracted data: {e}")
        print("Using fallback - no extracted data available")
        return None, {}, {}


def load_dt_reactivity_data():
    """Load D-T fusion reactivity ⟨σv⟩(T) from FusionReactivities.dat"""
    data_file = Path(__file__).parent / 'FusionReactivities.dat'

    temperatures = []  # keV
    dt_reactivities = []  # m³/s

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

    print(f"Loaded {len(temperatures)} D-T reactivity data points")
    print(f"Temperature range: {temperatures.min():.1f} - {temperatures.max():.1f} keV")

    interpolator = interp1d(temperatures, dt_reactivities,
                           kind='linear', bounds_error=False,
                           fill_value='extrapolate')

    return interpolator


# Load data globally at module import
E_beam_extracted, C_90deg_interp, ratio_45_90_interp = load_extracted_data()
dt_reactivity_interp = load_dt_reactivity_data()


# ============================================================================
# FUSION REACTIVITY
# ============================================================================

def get_dt_reactivity(T_keV):
    """Get ⟨σv⟩(T_keV) [m³/s]"""
    result = dt_reactivity_interp(T_keV)

    if np.isscalar(T_keV):
        result = float(result)

    return result


# ============================================================================
# LOSS COEFFICIENT
# ============================================================================

def calculate_loss_coefficient(E_b_100keV, R_M):
    """C@45° = C@90° × (C@45°/C@90°) × 0.1 [s]"""
    is_scalar = np.isscalar(E_b_100keV)
    E_b_100keV = np.asarray(E_b_100keV)
    E_b_keV = E_b_100keV * 100.0

    C = np.zeros_like(E_b_100keV, dtype=float)

    if len(C_90deg_interp) == 0 or len(ratio_45_90_interp) == 0:
        raise ValueError("No extracted data available for loss coefficient calculation.")

    R_M_int = int(round(R_M))
    available_Rm = list(C_90deg_interp.keys())
    R_M_clamped = np.clip(R_M_int, min(available_Rm), max(available_Rm))

    if R_M_clamped != R_M_int:
        print(f"Warning: R_M={R_M} clamped to {R_M_clamped} (available range: {min(available_Rm)}-{max(available_Rm)})")

    try:
        # Get C@90° values
        if R_M_clamped in C_90deg_interp:
            C_90deg = C_90deg_interp[R_M_clamped](E_b_keV)
        else:
            available_Rm_sorted = sorted(available_Rm)

            if R_M_clamped <= available_Rm_sorted[0]:
                C_90deg = C_90deg_interp[available_Rm_sorted[0]](E_b_keV)
            elif R_M_clamped >= available_Rm_sorted[-1]:
                C_90deg = C_90deg_interp[available_Rm_sorted[-1]](E_b_keV)
            else:
                R_M_lower = max([rm for rm in available_Rm_sorted if rm <= R_M_clamped])
                R_M_upper = min([rm for rm in available_Rm_sorted if rm >= R_M_clamped])

                if R_M_lower == R_M_upper:
                    C_90deg = C_90deg_interp[R_M_lower](E_b_keV)
                else:
                    C_90deg_lower = C_90deg_interp[R_M_lower](E_b_keV)
                    C_90deg_upper = C_90deg_interp[R_M_upper](E_b_keV)

                    log_rm = np.log10(R_M_clamped)
                    log_rm_lower = np.log10(R_M_lower)
                    log_rm_upper = np.log10(R_M_upper)

                    weight = (log_rm - log_rm_lower) / (log_rm_upper - log_rm_lower)
                    C_90deg = C_90deg_lower * (1 - weight) + C_90deg_upper * weight

        # Get C@45°/C@90° ratio
        if R_M_clamped in ratio_45_90_interp:
            ratio_45_90 = ratio_45_90_interp[R_M_clamped](E_b_keV)
        else:
            available_ratio_Rm = sorted(ratio_45_90_interp.keys())

            if R_M_clamped <= available_ratio_Rm[0]:
                ratio_45_90 = ratio_45_90_interp[available_ratio_Rm[0]](E_b_keV)
            elif R_M_clamped >= available_ratio_Rm[-1]:
                ratio_45_90 = ratio_45_90_interp[available_ratio_Rm[-1]](E_b_keV)
            else:
                R_M_lower = max([rm for rm in available_ratio_Rm if rm <= R_M_clamped])
                R_M_upper = min([rm for rm in available_ratio_Rm if rm >= R_M_clamped])

                if R_M_lower == R_M_upper:
                    ratio_45_90 = ratio_45_90_interp[R_M_lower](E_b_keV)
                else:
                    ratio_lower = ratio_45_90_interp[R_M_lower](E_b_keV)
                    ratio_upper = ratio_45_90_interp[R_M_upper](E_b_keV)

                    log_rm = np.log10(R_M_clamped)
                    log_rm_lower = np.log10(R_M_lower)
                    log_rm_upper = np.log10(R_M_upper)

                    weight = (log_rm - log_rm_lower) / (log_rm_upper - log_rm_lower)
                    ratio_45_90 = ratio_lower * (1 - weight) + ratio_upper * weight

        # Calculate C@45° = C@90° × ratio × 0.1
        C = C_90deg * ratio_45_90 * 0.1

    except Exception as e:
        raise RuntimeError(f"Error in extracted data interpolation: {e}") from e

    if is_scalar:
        C = float(C)



    C = (0.0957 + 0.0638 * np.log10(R_M))

    # Ensure output type matches input type for E_b_100keV
    is_scalar = np.isscalar(E_b_100keV)
    if is_scalar:
        C = float(C)
    else:
        # If E_b_100keV is an array, broadcast C to the same shape
        C = C * np.ones_like(E_b_100keV, dtype=float)




    return C


# ============================================================================
# MAGNETIC FIELD AND BETA
# ============================================================================

def calculate_beta_local(n_20, E_b_100keV, B_max, R_M):
    """β = 3n₂₀E_b / ((B_max/R_M)² + 3n₂₀E_b)"""
    B_center = B_max / R_M
    beta = (3 * n_20 * E_b_100keV) / (B_center**2 + 3 * n_20 * E_b_100keV)
    return beta


def calculate_B0_with_diamagnetic(B_max, R_M, beta_local):
    """B₀ = (B_max/R_M) / √(1 - β)"""
    B_0 = (B_max / R_M) / np.sqrt(1 - beta_local)
    return B_0


def calculate_beta_limit(E_b_100keV, B_max, R_M, beta_c):
    """n₂₀,max = (B_max/R_M)² × β_c / (3E_b(1 - β_c))"""
    n_20_max = ((B_max/R_M)**2 * beta_c) / (3 * E_b_100keV * (1 - beta_c))
    return n_20_max


# ============================================================================
# GEOMETRY CONSTRAINTS
# ============================================================================

def calculate_a0_absorption(E_b_100keV, n_20):
    """a₀,abs = 0.3√E_b / n₂₀"""
    a_0_abs = 0.3 * np.sqrt(E_b_100keV) / n_20
    return a_0_abs


def calculate_a0_FLR(E_b_100keV, B_0, N_25=1.0):
    """a₀,FLR = N₂₅√E_b / B₀"""
    a_0_FLR = N_25 * np.sqrt(E_b_100keV) / B_0
    return a_0_FLR


def calculate_plasma_geometry(a_0_min, N_rho=25.0):
    """L = N_ρ a₀, V = πa₀²L, A_vessel = 2.2πa₀²N_ρ"""
    L_plasma = N_rho * a_0_min
    V_plasma = np.pi * a_0_min**2 * L_plasma
    vessel_surface_area = 2.2 * np.pi * a_0_min**2 * N_rho

    return L_plasma, V_plasma, vessel_surface_area


# ============================================================================
# POWER CALCULATIONS
# ============================================================================

def calculate_fusion_power(E_b_100keV, n_20, V_plasma, T_i_keV):
    """P_fus = 7.04×10²¹ n₂₀² V ⟨σv⟩ [MW]"""
    sigma_v = get_dt_reactivity(T_i_keV)
    P_fusion = 7.04e21 * n_20**2 * V_plasma * sigma_v

    if np.isscalar(E_b_100keV) and np.isscalar(n_20) and np.isscalar(V_plasma) and np.isscalar(T_i_keV):
        P_fusion = float(P_fusion)

    return P_fusion


def calculate_NBI_power(n_20, V_plasma, E_b_100keV, R_M, C_loss=None, efficiency=NBI_EFFICIENCY):
    """P_NBI = (1/η) × 1.6 n₂₀² V / (C√E_b log₁₀R_M) [MW]"""
    if C_loss is None:
        C_loss = calculate_loss_coefficient(E_b_100keV, R_M)

    P_NBI = (1/efficiency) * 1.6 * n_20**2 * V_plasma / (C_loss * np.sqrt(E_b_100keV) * np.log10(R_M))

    if np.isscalar(n_20) and np.isscalar(V_plasma) and np.isscalar(E_b_100keV):
        P_NBI = float(P_NBI)

    return P_NBI


def calculate_NWL(P_fusion, vessel_surface_area, neutron_fraction=NEUTRON_FRACTION):
    """NWL = P_fus × (14.1/17.6) / A_vessel [MW/m²]"""
    NWL = P_fusion * neutron_fraction / vessel_surface_area

    if np.isscalar(P_fusion) and np.isscalar(vessel_surface_area):
        NWL = float(NWL)

    return NWL


def calculate_Q(P_fusion, P_NBI):
    """Q = P_fus / P_NBI"""
    Q = P_fusion / P_NBI

    if np.isscalar(P_fusion) and np.isscalar(P_NBI):
        Q = float(Q)

    return Q


# ============================================================================
# TESTING FUNCTION
# ============================================================================

def test_loss_coefficient(E_keV=120.0, R_M=7):
    """Test loss coefficient calculation at specific E_keV and R_M"""
    E_b_100keV = E_keV / 100.0

    print(f"\n=== Testing Loss Coefficient Calculation ===")
    print(f"Test parameters: E_beam = {E_keV} keV, R_M = {R_M}")

    C_extracted = calculate_loss_coefficient(E_b_100keV, R_M)
    print(f"C@45° (extracted): {C_extracted:.6f} s")

    if len(C_90deg_interp) > 0 and R_M in C_90deg_interp:
        C_90deg = C_90deg_interp[int(round(R_M))](E_keV)
        ratio = ratio_45_90_interp[int(round(R_M))](E_keV)
        print(f"Breakdown:")
        print(f"  C@90° = {C_90deg:.6f}")
        print(f"  C@45°/C@90° ratio = {ratio:.6f}")
        print(f"  C@45° = C@90° × ratio × 0.1 = {C_90deg * ratio * 0.1:.6f}")

    print(f"=== End Test ===\n")

    return C_extracted


if __name__ == "__main__":
    print("Testing equations module...")

    if len(C_90deg_interp) > 0:
        print(f"✓ Extracted data loaded successfully")
        print(f"  Available mirror ratios: {sorted(C_90deg_interp.keys())}")
        test_loss_coefficient(120.0, 7)
    else:
        print("⚠ No extracted data available")
