##### Faris POPCON #####

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
import pandas as pd

# ============================================================================
# EXTRACTED DATA LOADER
# ============================================================================

def load_extracted_data():
    """
    Load extracted curve data from graph extraction

    Returns:
    --------
    E_beam_keV : array
        Beam energies in keV
    C_90deg_interpolators : dict
        Interpolation functions for C@90° for each Rm
    ratio_45_90_interpolators : dict
        Interpolation functions for C@45°/C@90° ratios for each Rm
    """
    # Load the extracted data CSV
    data_file = Path(__file__).parent / 'OCR' / 'outputs' / 'extracted_data.csv'

    if not data_file.exists():
        print(f"Warning: Extracted data file not found at {data_file}")
        print("Using fallback loss coefficient calculation")
        return None, {}, {}

    try:
        df = pd.read_csv(data_file)

        # Get beam energies
        E_beam_keV = df['E_beam_keV'].values

        # Available mirror ratios (from 4 to 16)
        available_Rm = range(4, 17)

        # Create interpolators for C@90° values
        C_90deg_interpolators = {}
        for Rm in available_Rm:
            col_name = f'd_Rm{Rm}'
            if col_name in df.columns:
                # Remove NaN values for interpolation
                valid_mask = ~pd.isna(df[col_name])
                if np.sum(valid_mask) > 1:  # Need at least 2 points
                    E_valid = E_beam_keV[valid_mask]
                    C_valid = df[col_name].values[valid_mask]

                    # Create interpolator with extrapolation
                    interpolator = interp1d(E_valid, C_valid, kind='linear',
                                          bounds_error=False, fill_value='extrapolate')
                    C_90deg_interpolators[Rm] = interpolator
                    print(f"Loaded C@90° interpolator for Rm={Rm} ({len(E_valid)} points)")

        # Create interpolators for C@45°/C@90° ratios
        ratio_45_90_interpolators = {}
        for Rm in available_Rm:
            ratio_col_name = f'ratio_45_90_Rm{Rm}'
            if ratio_col_name in df.columns:
                # Remove NaN values for interpolation
                valid_mask = ~pd.isna(df[ratio_col_name])
                if np.sum(valid_mask) > 1:  # Need at least 2 points
                    E_valid = E_beam_keV[valid_mask]
                    ratio_valid = df[ratio_col_name].values[valid_mask]

                    # For ratios, they should be constant across energy, so just take the mean
                    ratio_mean = np.mean(ratio_valid)
                    # Create constant interpolator
                    ratio_45_90_interpolators[Rm] = lambda E, ratio=ratio_mean: ratio
                    print(f"Loaded C@45°/C@90° ratio for Rm={Rm}: {ratio_mean:.4f}")

        print(f"Successfully loaded extracted data with {len(E_beam_keV)} energy points")
        return E_beam_keV, C_90deg_interpolators, ratio_45_90_interpolators

    except Exception as e:
        print(f"Error loading extracted data: {e}")
        print("Using fallback loss coefficient calculation")
        return None, {}, {}

# Load extracted data globally
E_beam_extracted, C_90deg_interp, ratio_45_90_interp = load_extracted_data()

# ============================================================================
# FUSION REACTIVITY DATA READER
# ============================================================================

def load_dt_reactivity_data():
    """
    Load D-T fusion reactivity data from FusionReactivities.dat

    Returns:
    --------
    interpolator : function
        Interpolation function for D-T reactivity vs temperature
    """
    # Get path to reactivity data file
    data_file = Path(__file__).parent / 'FusionReactivities.dat'

    # Read the data
    temperatures = []  # keV
    dt_reactivities = []  # m³/s

    with open(data_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line.startswith('#') or not line:
                continue

            # Parse data line
            parts = line.split()
            if len(parts) >= 2:
                try:
                    temp = float(parts[0])  # Temperature in keV
                    dt_reactivity = float(parts[1])  # D-T reactivity in m³/s
                    temperatures.append(temp)
                    dt_reactivities.append(dt_reactivity)
                except ValueError:
                    continue

    # Convert to numpy arrays
    temperatures = np.array(temperatures)
    dt_reactivities = np.array(dt_reactivities)

    print(f"Loaded {len(temperatures)} D-T reactivity data points")
    print(f"Temperature range: {temperatures.min():.1f} - {temperatures.max():.1f} keV")

    # Create interpolation function with extrapolation
    interpolator = interp1d(temperatures, dt_reactivities,
                           kind='linear', bounds_error=False,
                           fill_value='extrapolate')

    return interpolator

# Global interpolator for D-T reactivity
dt_reactivity_interp = load_dt_reactivity_data()

def get_dt_reactivity(T_keV):
    """
    Get D-T fusion reactivity at given temperature

    Parameters:
    -----------
    T_keV : float or array
        Temperature in keV

    Returns:
    --------
    reactivity : float or array
        D-T fusion reactivity in m³/s
    """
    result = dt_reactivity_interp(T_keV)

    # Convert to scalar if input was scalar
    if np.isscalar(T_keV):
        result = float(result)

    return result

# ============================================================================
# PHYSICS CONSTANTS AND HELPER FUNCTIONS
# ============================================================================

# Loss coefficient from Egedal et al. (Section from paper)
C_LOSS_COEFF = 0.17  # [s] for E_b=100keV, theta=45°, R_M=10

def calculate_loss_coefficient(E_b_100keV, R_M):
    """
    Calculate loss coefficient C(E_b, theta=45°, R_M) using extracted graph data

    This function now uses the extracted C@90° curves and C@45°/C@90° ratios
    to calculate C@45° = C@90° × ratio

    Parameters:
    -----------
    E_b_100keV : float or array
        Beam energy in 100 keV units
    R_M : float
        Mirror ratio

    Returns:
    --------
    C : float or array
        Loss coefficient C@45° in [s]
    """
    # Handle scalar input case
    is_scalar = np.isscalar(E_b_100keV)
    E_b_100keV = np.asarray(E_b_100keV)

    # Convert to keV
    E_b_keV = E_b_100keV * 100.0

    # Initialize output array
    C = np.zeros_like(E_b_100keV, dtype=float)

    # Check if extracted data is available
    if len(C_90deg_interp) == 0 or len(ratio_45_90_interp) == 0:
        print("Warning: Using fallback loss coefficient calculation")
        return calculate_loss_coefficient_fallback(E_b_100keV, R_M)

    # Round R_M to nearest integer for lookup
    R_M_int = int(round(R_M))

    # Clamp R_M to available range
    available_Rm = list(C_90deg_interp.keys())
    R_M_clamped = np.clip(R_M_int, min(available_Rm), max(available_Rm))

    if R_M_clamped != R_M_int:
        print(f"Warning: R_M={R_M} clamped to {R_M_clamped} (available range: {min(available_Rm)}-{max(available_Rm)})")

    try:
        # Get C@90° values for this mirror ratio
        if R_M_clamped in C_90deg_interp:
            C_90deg = C_90deg_interp[R_M_clamped](E_b_keV)
        else:
            # Interpolate between nearby mirror ratios
            available_Rm_sorted = sorted(available_Rm)

            # Find bounding Rm values
            if R_M_clamped <= available_Rm_sorted[0]:
                C_90deg = C_90deg_interp[available_Rm_sorted[0]](E_b_keV)
            elif R_M_clamped >= available_Rm_sorted[-1]:
                C_90deg = C_90deg_interp[available_Rm_sorted[-1]](E_b_keV)
            else:
                # Linear interpolation in log(Rm) space
                R_M_lower = max([rm for rm in available_Rm_sorted if rm <= R_M_clamped])
                R_M_upper = min([rm for rm in available_Rm_sorted if rm >= R_M_clamped])

                if R_M_lower == R_M_upper:
                    C_90deg = C_90deg_interp[R_M_lower](E_b_keV)
                else:
                    C_90deg_lower = C_90deg_interp[R_M_lower](E_b_keV)
                    C_90deg_upper = C_90deg_interp[R_M_upper](E_b_keV)

                    # Interpolate in log space
                    log_rm = np.log10(R_M_clamped)
                    log_rm_lower = np.log10(R_M_lower)
                    log_rm_upper = np.log10(R_M_upper)

                    weight = (log_rm - log_rm_lower) / (log_rm_upper - log_rm_lower)
                    C_90deg = C_90deg_lower * (1 - weight) + C_90deg_upper * weight

        # Get C@45°/C@90° ratio for this mirror ratio
        if R_M_clamped in ratio_45_90_interp:
            ratio_45_90 = ratio_45_90_interp[R_M_clamped](E_b_keV)
        else:
            # Interpolate ratio between nearby mirror ratios
            available_ratio_Rm = sorted(ratio_45_90_interp.keys())

            if R_M_clamped <= available_ratio_Rm[0]:
                ratio_45_90 = ratio_45_90_interp[available_ratio_Rm[0]](E_b_keV)
            elif R_M_clamped >= available_ratio_Rm[-1]:
                ratio_45_90 = ratio_45_90_interp[available_ratio_Rm[-1]](E_b_keV)
            else:
                # Linear interpolation in log(Rm) space for ratios too
                R_M_lower = max([rm for rm in available_ratio_Rm if rm <= R_M_clamped])
                R_M_upper = min([rm for rm in available_ratio_Rm if rm >= R_M_clamped])

                if R_M_lower == R_M_upper:
                    ratio_45_90 = ratio_45_90_interp[R_M_lower](E_b_keV)
                else:
                    ratio_lower = ratio_45_90_interp[R_M_lower](E_b_keV)
                    ratio_upper = ratio_45_90_interp[R_M_upper](E_b_keV)

                    # Interpolate in log space
                    log_rm = np.log10(R_M_clamped)
                    log_rm_lower = np.log10(R_M_lower)
                    log_rm_upper = np.log10(R_M_upper)

                    weight = (log_rm - log_rm_lower) / (log_rm_upper - log_rm_lower)
                    ratio_45_90 = ratio_lower * (1 - weight) + ratio_upper * weight

        # Calculate C@45° = C@90° × ratio, 0.1 for units
        C = C_90deg * ratio_45_90 * 0.1

    except Exception as e:
        print(f"Error in extracted data interpolation: {e}")
        print("Falling back to original calculation")
        return calculate_loss_coefficient_fallback(E_b_100keV, R_M)

    # Return scalar if input was scalar
    if is_scalar:
        C = float(C)

    return C

def calculate_loss_coefficient_fallback(E_b_100keV, R_M):
    """
    Fallback loss coefficient calculation (original approximation)
    """
    # Base value for 100 keV, 45°, R_M=10
    C_base = C_LOSS_COEFF

    # Handle scalar input case
    is_scalar = np.isscalar(E_b_100keV)
    E_b_100keV = np.asarray(E_b_100keV)

    # Initialize output array
    C = np.zeros_like(E_b_100keV, dtype=float)

    # Approximate scaling from paper - apply element-wise
    low_energy_mask = E_b_100keV <= 1.05
    high_energy_mask = E_b_100keV > 1.05

    C[low_energy_mask] = C_base - 0.04375 * (1 - E_b_100keV[low_energy_mask])
    C[high_energy_mask] = 1.1 * C_base

    # Return scalar if input was scalar
    if is_scalar:
        C = float(C)

    return C

def calculate_fusion_power_paper(E_b_100keV, n_20, V_plasma, T_i_keV):
    """
    Calculate fusion power using paper formula: P_fus = 7.04×10²¹ n₂₀²V⟨σv⟩ [MW]

    Parameters:
    -----------
    E_b_100keV : float or array
        Beam energy in 100 keV units
    n_20 : float or array
        Density in 10^20 m^-3
    V_plasma : float or array
        Plasma volume in m³
    T_i_keV : float or array
        Ion temperature in keV

    Returns:
    --------
    P_fusion : float or array
        Fusion power in MW
    """
    # Get D-T reactivity at ion temperature
    sigma_v = get_dt_reactivity(T_i_keV)  # m³/s

    # Paper formula: P_fus = 7.04×10²¹ n₂₀²V⟨σv⟩ [MW]
    P_fusion = 7.04e21 * n_20**2 * V_plasma * sigma_v  # MW

    # Ensure scalar output for scalar inputs
    if np.isscalar(E_b_100keV) and np.isscalar(n_20) and np.isscalar(V_plasma) and np.isscalar(T_i_keV):
        P_fusion = float(P_fusion)

    return P_fusion

def calculate_Q_simple(P_fusion_MW, P_NBI_MW):
    """
    Calculate Q as simple ratio: Q = P_fusion / P_NBI

    Parameters:
    -----------
    P_fusion_MW : float or array
        Fusion power in MW
    P_NBI_MW : float or array
        Required NBI power in MW

    Returns:
    --------
    Q : float or array
        Q factor (dimensionless)
    """
    Q = P_fusion_MW / P_NBI_MW

    # Ensure scalar output for scalar inputs
    if np.isscalar(P_fusion_MW) and np.isscalar(P_NBI_MW):
        Q = float(Q)

    return Q

# Create figures directory if it doesn't exist
figures_dir = Path(__file__).parent / 'figures'
figures_dir.mkdir(exist_ok=True)

# ============================================================================
# INPUT PARAMETERS - MODIFY THESE
# ============================================================================

# Fixed physics parameters for single plots and when fixed in grids
B_max_default = 26.0      # Maximum mirror field [T]
R_M_default = 5       # Mirror ratio
beta_c_default = 0.3    # MHD stability limit

# Fixed values to use when a parameter is held constant in grid plots
fixed_B_max = 22        # Use this when B_max is fixed
fixed_R_M = 10        # Use this when R_M is fixed
fixed_beta_c = 0.3        # Use this when beta_c is fixed

# Energy range - THIS APPLIES TO ALL PLOTS
E_b_min = 0.4            # Minimum beam energy [100 keV units]
E_b_max = 1.2            # Maximum beam energy [100 keV units]

# Temperature options (matching beam_target.py system)
use_temperature_override = False  # If True, use explicit temperatures; if False, use beam coefficients [TOGGLE]

# Option 1: Beam energy coefficients (used when use_temperature_override=False)
T_i_coeff = 2/3        # Ti = T_i_coeff * E_b [keV]  (paper default: 2/3)
T_e_coeff = 0.1          # Te = T_e_coeff * E_b [keV]  (paper default: 0.1)

# Option 2: Fixed explicit temperatures (used when use_temperature_override=True)
T_i_fixed = 50.0         # Fixed ion temperature [keV]
T_e_fixed = 10.0         # Fixed electron temperature [keV]

# Practical engineering limit - CONFIGURABLE SIZE CONSTRAINT SYSTEM
size_limit_type = 'a0'      # Size constraint type: 'a0' (radius), 'L' (length), 'V' (volume) [TOGGLE]
size_limit_value = 1.5      # Size constraint value: a0 [m], L [m], or V [m³] [MODIFY BASED ON TYPE]

# Plotting parameters
a0_min = 0.0             # Minimum radius for contours [m]
a0_step = 0.1            # Step size for radius contours [m]
a0_max = 1.0             # Maximum radius for contours [m]

# Grid resolution
n_grid_points = 500      # Number of grid points (higher = smoother)
n_grid_points_small = 500  # For grid plots (faster)

# Grid plot options
create_bmax_rm_grid = False     # Create B_max vs R_M forward analysis grid [TOGGLE]

# B_max vs R_M grid parameters
B_max_grid_values = [18, 22, 26]        # B_max values for grid [T]
R_M_grid_values = [7, 10, 13]          # R_M values for grid
beta_c_grid_fixed = 0.3                 # Fixed beta_c for B_max vs R_M grid

# Contour levels for POPCON plots
Q_levels = np.array([0.4, 0.5, 0.6, 0.7, 0.8 , 1.0, 1.2, 1.4, 5.0, 10.0])
# Background color levels (fine resolution for smooth colors)
max_NWL = 8
NWL_background = np.linspace(0, max_NWL, 25)                   # MW/m²
# Contour line levels (clean decimal points for labels)
NWL_levels = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 7.5])      # MW/m²
a0_levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])  # Minor radius [m]
P_fus_levels = np.array([])  # Fusion power [MW]
P_NBI_levels = np.array([10, 20, 30, 40, 50, 100, 200, 500, 1000])  # NBI power [MW]
B_0_levels = np.array([])  # On-axis field [T]

# ============================================================================
# MAIN POPCON FUNCTION
# ============================================================================

def create_single_popcon(ax, B_max, R_M, beta_c, E_b_min, E_b_max,
                         a0_min, a0_step, a0_max,
                         n_grid_points=200, show_legend=False, simplified_contours=False):
    """Generate a single POPCON plot with beam-target fusion physics

    Parameters:
    -----------
    simplified_contours : bool
        If True, only show NWL, Length, and Q contours (for grid plots)
        If False, show all contours (for main single plot)
    """

    # Calculate derived parameters
    B_0 = B_max / R_M

    # Create grid
    E_b100 = np.linspace(E_b_min, E_b_max, n_grid_points)
    # Updated beta limit with diamagnetic effects from paper
    n_20_max = ((B_max/R_M)**2 * beta_c) / (3 * E_b_min * (1 - beta_c))
    n_20 = np.linspace(0.01, n_20_max, n_grid_points)

    E_b100_grid, n_20_grid = np.meshgrid(E_b100, n_20)

    # Calculate constraints with diamagnetic effects
    n_20_beta_limit = ((B_max/R_M)**2 * beta_c) / (3 * E_b100_grid * (1 - beta_c))

    # NEW: Implement N_25 from paper Section 2.3.1
    N_25 = np.ones_like(E_b100_grid)  # Fixed at 1

    # Calculate local beta with diamagnetic effects
    beta_local = (3 * n_20_grid * E_b100_grid) / ((B_max/R_M)**2 + 3 * n_20_grid * E_b100_grid)

    # Calculate on-axis field B0 with diamagnetic effects: B0 = B_max/(R_M * sqrt(1-beta))
    B_0_grid = (B_max / R_M) / np.sqrt(1 - beta_local)

    a_0_abs = 0.3 * np.sqrt(E_b100_grid) / n_20_grid
    a_0_FLR = N_25 * np.sqrt(E_b100_grid) / B_0  # Use original B_0 for this constraint
    a_0_min_calc = np.maximum(a_0_abs, a_0_FLR)
    # Calculate vessel surface area for NWL (following paper Eq. vessel_area)
    # Use effective N_rho = 25 * N_25 to account for varying N_25
    N_rho = 25.0 * N_25
    L_plasma = N_rho * a_0_min_calc  # L = N_ρ * a0
    # Paper: A_vessel = 1.1 × 2π a₀ L = 2.2π a₀² N_ρ (10% gap, cylindrical wall only)
    vessel_surface_area = 2.2 * np.pi * a_0_min_calc**2 * N_rho

    # Calculate volume for contours: V = π * a₀² * L = π * N_ρ * a₀³
    V_plasma = np.pi * N_rho * a_0_min_calc**3

    # Calculate required NBI power from paper: P_B = 1.6n₂₀²V/(C(E_b,θ,R_M)√E_b,100keV log R_M)
    # Now using extracted graph data for loss coefficient calculation
    C_loss = calculate_loss_coefficient(E_b100_grid, R_M)
    P_NBI_required = 1.6 * n_20_grid**2 * V_plasma / (C_loss * np.sqrt(E_b100_grid) * np.log10(R_M))  # MW

    # Calculate beam-target fusion properly with n² dependence at each grid point
    print(f"Calculating beam-target physics for {n_grid_points}x{n_grid_points} grid points...")

    # Initialize arrays for full grid calculation
    P_fusion_beam_target = np.zeros_like(E_b100_grid)
    Q_beam_target = np.zeros_like(E_b100_grid)

    # Calculate for each grid point (density and energy dependent)
    for i in range(n_grid_points):
        for j in range(n_grid_points):
            E_b_100_point = E_b100_grid[i, j]  # Energy in 100 keV units
            n_20_point = n_20_grid[i, j]       # Density in 10^20 m^-3
            E_NBI_keV = E_b_100_point * 100    # Convert to keV

            try:
                # Calculate temperatures using new system
                T_i, T_e = calculate_temperatures(E_NBI_keV)

                # Use the SAME volume that was already calculated for the grid
                V = V_plasma[i, j]  # Use the grid volume, not recalculating!

                # Calculate fusion power using paper formula - WITH CORRECT DENSITY AND VOLUME
                P_fusion = calculate_fusion_power_paper(E_b_100_point, n_20_point, V, T_i)

                # Q calculation using consistent volumes (P_NBI already calculated with same V)
                if P_NBI_required[i, j] > 0:
                    Q = P_fusion / P_NBI_required[i, j]
                else:
                    Q = 0

                P_fusion_beam_target[i, j] = P_fusion
                Q_beam_target[i, j] = Q

            except Exception as e:
                print(f"Error at E_b={E_NBI_keV:.1f} keV, n_20={n_20_point:.2f}: {e}")
                P_fusion_beam_target[i, j] = 0
                Q_beam_target[i, j] = 0

    # NWL scales with fusion power and inversely with vessel surface area
    # NWL should only include neutron energy: 14.1 MeV out of 17.6 MeV total
    neutron_fraction = 14.1 / 17.6  # Fraction of fusion energy carried by neutrons

    # Calculate NWL using the corrected fusion power (already calculated for each grid point)
    NWL_beam_target = P_fusion_beam_target * neutron_fraction / vessel_surface_area

    # Calculate effective a0 limit based on size_limit_type and N_rho
    if size_limit_type == 'a0':
        a0_eff_limit = size_limit_value  # Direct radius limit
    elif size_limit_type == 'L':
        # L = N_rho * a0, so a0 = L / N_rho
        a0_eff_limit = size_limit_value / N_rho
    elif size_limit_type == 'V':
        # V = π * N_rho * a0³, so a0 = (V / (π * N_rho))^(1/3)
        a0_eff_limit = (size_limit_value / (np.pi * N_rho))**(1/3)
    else:
        a0_eff_limit = 1.5  # Fallback

    # Create masks for different regions
    mask_beta = n_20_grid > n_20_beta_limit                    # Above beta limit
    mask_impractical = a_0_min_calc > a0_eff_limit            # Device too large (using dynamic limit)
    mask_low_NWL = NWL_beam_target < 0.0                      # NWL too low

    # Region priorities (in order):
    # 1. Gray: Invalid physics/engineering (beta limit or impractical size)
    # 2. White: Low NWL performance
    # 3. Colored: Good operating regions

    mask_gray = mask_beta | mask_impractical
    mask_black = np.zeros_like(mask_gray, dtype=bool)  # No black regions
    mask_white = (~mask_gray) & mask_low_NWL

    # Fill regions in order
    # 1. Gray regions (invalid)
    ax.contourf(E_b100_grid, n_20_grid, mask_gray.astype(int),
                levels=[0.5, 1.5], colors=['lightgray'], alpha=0.8)

    # 2. Black regions (low required power)
    ax.contourf(E_b100_grid, n_20_grid, mask_black.astype(int),
                levels=[0.5, 1.5], colors=['black'], alpha=0.8)

    # 3. White regions remain white (matplotlib default background)

    # Plot NWL contours (main plot) - using beam-target physics
    NWL_valid = NWL_beam_target.copy()
    NWL_valid[mask_gray | mask_black | mask_white] = np.nan  # Mask all non-contour regions

    im = ax.contourf(E_b100_grid, n_20_grid, NWL_valid,
                     levels=NWL_background, cmap='viridis', extend='max')

    # Beta limit line
    ax.plot(E_b100, n_20_beta_limit[0,:], 'purple', linewidth=2, zorder=5,
            label='β limit' if show_legend else '')

    # Size limit boundary line - show where a_0_min_calc = a0_eff_limit
    size_limit_boundary = a_0_min_calc - a0_eff_limit  # Zero where a_0_min_calc = a0_eff_limit
    CS_size = ax.contour(E_b100_grid, n_20_grid, size_limit_boundary,
                        levels=[0], colors=['darkred'], linewidths=2, linestyles='--', zorder=4)

    # Label based on size constraint type (only if showing legend)
    if show_legend:
        if size_limit_type == 'a0':
            size_label = f'a₀={size_limit_value:.1f}m limit'
        elif size_limit_type == 'L':
            size_label = f'L={size_limit_value:.0f}m limit'
        elif size_limit_type == 'V':
            size_label = f'V={size_limit_value:.0f}m³ limit'
        else:
            size_label = 'Size limit'

        # Add to legend manually
        ax.plot([], [], color='darkred', linewidth=2, linestyle='--', label=size_label)

    # For small plots: show both a₀ and P_NBI contours (same as big plots)
    # For big plots: show a₀ contours
    if simplified_contours:
        # a₀ contours for small plots (same style as big plot)
        a_0_min_valid = a_0_min_calc.copy()
        a_0_min_valid[mask_gray | mask_black | mask_white] = np.nan

        CS_a0 = ax.contour(E_b100_grid, n_20_grid, a_0_min_valid,
                          levels=a0_levels, colors='pink', linewidths=1.5, alpha=0.9, linestyles='-')

        # Label all a₀ contours (every level, not skipping)
        for level in a0_levels:
            label = f'a₀={level:.1f}m'
            ax.clabel(CS_a0, levels=[level], inline=True, fontsize=5, fmt=label)

        # P_NBI contours for small plots (same style as big plot)
        P_NBI_valid = P_NBI_required.copy()
        P_NBI_valid[mask_gray | mask_black | mask_white] = np.nan
        CS_PNBI = ax.contour(E_b100_grid, n_20_grid, P_NBI_valid,
                           levels=P_NBI_levels, colors='red', linewidths=2.5, alpha=1.0, linestyles='-')
        ax.clabel(CS_PNBI, inline=True, fontsize=5, fmt='P_NBI=%.0f MW')
    else:
        # a₀ contours for detailed plots (needed for forward analysis)
        a_0_min_valid = a_0_min_calc.copy()
        a_0_min_valid[mask_gray | mask_black | mask_white] = np.nan # Mask all non-contour regions

        CS = ax.contour(E_b100_grid, n_20_grid, a_0_min_valid,
                        levels=a0_levels, colors='pink', linewidths=2, alpha=0.8, linestyles='--')

        # Label all a₀ contours (every level)
        for level in a0_levels:
            label = f'a₀={level:.1f}m'
            ax.clabel(CS, levels=[level], inline=True, fontsize=5, fmt=label)

    # Q contour lines - using beam-target physics (density-invariant)
    Q_valid = Q_beam_target.copy()
    Q_valid[mask_gray | mask_black | mask_white] = np.nan # Mask all non-contour regions

    CS_Q = ax.contour(E_b100_grid, n_20_grid, Q_valid,
                      levels=Q_levels, colors='cyan', linewidths=2.0,
                      alpha=0.9, linestyles='-')
    ax.clabel(CS_Q, inline=True, fontsize=6, fmt='Q=%.2f')

    # P_fusion contour lines - using beam-target physics (density-invariant)
    P_fusion_valid = P_fusion_beam_target.copy()
    P_fusion_valid[mask_gray | mask_black | mask_white] = np.nan

    CS_Pfus = ax.contour(E_b100_grid, n_20_grid, P_fusion_valid,
                         levels=P_fus_levels, colors='magenta', linewidths=1.5,
                         alpha=0.8, linestyles='-')
    ax.clabel(CS_Pfus, inline=True, fontsize=6, fmt='P_fus=%.0f MW')

    # NWL contour lines - using beam-target physics
    CS_NWL = ax.contour(E_b100_grid, n_20_grid, NWL_valid,
                        levels=NWL_levels, colors='white', linewidths=2.5,
                        alpha=0.9, linestyles='-')
    ax.clabel(CS_NWL, inline=True, fontsize=6, fmt='%.1f MW/m²')

    # B₀ and P_NBI contours - only for detailed plots
    if not simplified_contours:
        # B₀ contours with diamagnetic effects
        B_0_valid = B_0_grid.copy()
        B_0_valid[mask_gray | mask_black | mask_white] = np.nan
        CS_B0 = ax.contour(E_b100_grid, n_20_grid, B_0_valid,
                           levels=B_0_levels, colors='orange', linewidths=1.5,
                           alpha=0.7, linestyles='-')
        ax.clabel(CS_B0, inline=True, fontsize=6, fmt='B₀=%.1f T')

        # P_NBI contours - solid lines with strong color
        P_NBI_valid = P_NBI_required.copy()
        P_NBI_valid[mask_gray | mask_black | mask_white] = np.nan
        CS_PNBI = ax.contour(E_b100_grid, n_20_grid, P_NBI_valid,
                             levels=P_NBI_levels, colors='red', linewidths=2.5,
                             alpha=1.0, linestyles='-')
        ax.clabel(CS_PNBI, inline=True, fontsize=8, fmt='P_NBI=%.0f MW')

    # Beta contours: β = 3*n₂₀*E_b/B₀²
    if not simplified_contours:
        beta_valid = beta_local.copy()
        beta_valid[mask_gray | mask_black | mask_white] = np.nan
        beta_levels = np.array([0.05, 0.15, 0.25])  # Beta levels
        CS_beta = ax.contour(E_b100_grid, n_20_grid, beta_valid,
                            levels=beta_levels, colors='orange', linewidths=2.0,
                            alpha=0.9, linestyles='-')
        ax.clabel(CS_beta, inline=True, fontsize=6, fmt='β=%.2f')

        # C (Loss Coefficient) contours
        C_valid = C_loss.copy()
        C_valid[mask_gray | mask_black | mask_white] = np.nan
        C_levels = np.array([])  # Loss coefficient levels [s]
        CS_C = ax.contour(E_b100_grid, n_20_grid, C_valid,
                         levels=C_levels, colors='brown', linewidths=1.5,
                         alpha=0.8, linestyles=':')
        ax.clabel(CS_C, inline=True, fontsize=6, fmt='C=%.2f s')

    # Formatting
    ax.set_xlim([E_b_min, E_b_max])
    ax.set_ylim([0, n_20_max])
    ax.grid(True, alpha=0.15, linestyle=':', linewidth=0.3)

    # Force linear tick formatting with clean decimal values
    ax.ticklabel_format(style='plain', axis='x')
    ax.ticklabel_format(style='plain', axis='y')
    # Set explicit x-axis ticks for clean decimal display
    x_ticks = np.arange(E_b_min, E_b_max, 0.2)  # [0.4, 0.6, 0.8, 1.0]
    ax.set_xticks(x_ticks)

    # Title with temperature scaling info
    if use_temperature_override:
        temp_info = f"$T_i={T_i_fixed:.0f}$keV, $T_e={T_e_fixed:.0f}$keV (fixed)"
    else:
        temp_info = f"$T_i={T_i_coeff:.2f}E_{{NBI}}$, $T_e={T_e_coeff:.2f}E_{{NBI}}$"

    # Size constraint info for single popcon
    if size_limit_type == 'a0':
        size_info_single = f'a₀≤{size_limit_value:.1f}m'
    elif size_limit_type == 'L':
        size_info_single = f'L≤{size_limit_value:.0f}m'
    elif size_limit_type == 'V':
        size_info_single = f'V≤{size_limit_value:.0f}m³'
    else:
        size_info_single = ''

    ax.set_title(f'$B_{{max}}$={B_max:.1f}T, $R_M$={R_M:.0f}, $\\beta_c$={beta_c:.2f}, {size_info_single}\n'
                f'{temp_info}',
                 fontsize=9, weight='bold')

    # Add small legend only if requested
    if show_legend:
        ax.legend(loc='lower left', fontsize=5, framealpha=0.7)

    return im



# ============================================================================
# B_MAX VS R_M FORWARD ANALYSIS GRID
# ============================================================================

def create_rm_scan_plot():
    """Create R_M scan: 3 horizontal subplots with increasing R_M at default B_max"""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'R_M Scan at B_max={B_max_default}T, β_c={beta_c_grid_fixed}\n'
                 f'Contours: Q (cyan), NWL (white), a₀ (black), P_NBI (red)',
                 fontsize=14, weight='bold')

    for i, R_M in enumerate(R_M_grid_values):
        ax = axes[i]

        # Create simplified POPCON with only Q, NWL, and a0 contours
        im = create_single_popcon(ax, B_max_default, R_M, beta_c_grid_fixed,
                                E_b_min, E_b_max,
                                a0_min, a0_step, a0_max,
                                n_grid_points_small,
                                show_legend=(i==0),
                                simplified_contours=True)

        # Labels
        ax.set_xlabel('$E_{NBI}$ [100 keV]', fontsize=12)
        if i == 0:
            ax.set_ylabel('$n_{20}$', fontsize=12)

        # Subplot title
        ax.set_title(f'$R_M$={R_M}', fontsize=13, weight='bold')
        ax.tick_params(labelsize=10)

    plt.tight_layout()

    # Add shared colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label(r'NWL [MW/m²]', fontsize=12)
    cbar.set_ticks(NWL_levels)  # Use clean decimal points for ticks
    cbar.ax.tick_params(labelsize=10)

    return fig

def create_bmax_scan_plot():
    """Create B_max scan: 3 horizontal subplots with increasing B_max at default R_M"""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'B_max Scan at R_M={R_M_default}, β_c={beta_c_grid_fixed}\n'
                 f'Contours: Q (cyan), NWL (white), a₀ (black), P_NBI (red)',
                 fontsize=14, weight='bold')

    for i, B_max in enumerate(B_max_grid_values):
        ax = axes[i]

        # Create simplified POPCON with only Q, NWL, and a0 contours
        im = create_single_popcon(ax, B_max, R_M_default, beta_c_grid_fixed,
                                E_b_min, E_b_max,
                                a0_min, a0_step, a0_max,
                                n_grid_points_small,
                                show_legend=(i==0),
                                simplified_contours=True)

        # Labels
        ax.set_xlabel('$E_{NBI}$ [100 keV]', fontsize=12)
        if i == 0:
            ax.set_ylabel('$n_{20}$', fontsize=12)

        # Subplot title
        ax.set_title(f'$B_{{max}}$={B_max}T', fontsize=13, weight='bold')
        ax.tick_params(labelsize=10)

    plt.tight_layout()

    # Add shared colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label(r'NWL [MW/m²]', fontsize=12)
    cbar.set_ticks(NWL_levels)  # Use clean decimal points for ticks
    cbar.ax.tick_params(labelsize=10)

    return fig

# ============================================================================
# SINGLE FULL-SIZE POPCON PLOT
# ============================================================================

def create_full_popcon(B_max=B_max_default, R_M=R_M_default, beta_c=beta_c_default):
    """Create a full-size detailed POPCON plot with beam-target fusion physics"""

    fig, ax = plt.subplots(figsize=(11, 8))

    # Calculate derived parameters
    B_0 = B_max / R_M

    # Create grid using input E_b range
    E_b100 = np.linspace(E_b_min, E_b_max, n_grid_points)
    # Updated beta limit with diamagnetic effects from paper
    n_20_max = ((B_max/R_M)**2 * beta_c) / (3 * E_b_min * (1 - beta_c))
    n_20 = np.linspace(0.01, n_20_max, n_grid_points)

    E_b100_grid, n_20_grid = np.meshgrid(E_b100, n_20)

    # Calculate constraints with diamagnetic effects
    n_20_beta_limit = ((B_max/R_M)**2 * beta_c) / (3 * E_b100_grid * (1 - beta_c))

    # Paper-based constraints with N₂₅ = 1 (fixed)
    N_25 = np.ones_like(E_b100_grid)  # Fixed at 1

    # Calculate local beta with diamagnetic effects: β = 3n₂₀E_b/((B_max/R_m)² + 3n₂₀E_b)
    beta_local = (3 * n_20_grid * E_b100_grid) / ((B_max/R_M)**2 + 3 * n_20_grid * E_b100_grid)

    # Calculate on-axis field B0 with diamagnetic effects: B0 = B_max/(R_M * sqrt(1-beta))
    B_0_grid = (B_max / R_M) / np.sqrt(1 - beta_local)

    # a₀ constraints from paper
    a_0_abs = 0.3 * np.sqrt(E_b100_grid) / n_20_grid  # Absorption constraint
    a_0_FLR = N_25 * np.sqrt(E_b100_grid) / B_0       # FLR constraint (N₂₅=1)
    a_0_min = np.maximum(a_0_abs, a_0_FLR)             # Take maximum

    # Calculate vessel surface area for NWL (following paper Eq. vessel_area)
    # Use N_rho = 25 since N₂₅ = 1 (from paper)
    N_rho = 25.0 * N_25  # = 25 since N₂₅ = 1
    L_plasma = N_rho * a_0_min  # L = N_ρ * a0
    # Paper: A_vessel = 1.1 × 2π a₀ L = 2.2π a₀² N_ρ (10% gap, cylindrical wall only)
    vessel_surface_area = 2.2 * np.pi * a_0_min**2 * N_rho

    # Calculate required NBI power from paper: P_B = 1.6n₂₀²V/(C(E_b,θ,R_M)√E_b,100keV log R_M)
    # Now using extracted graph data for loss coefficient calculation
    V_plasma = np.pi * a_0_min**2 * L_plasma  # Plasma volume = π * a₀² * L
    C_loss = calculate_loss_coefficient(E_b100_grid, R_M)
    P_NBI_required = 1.6 * n_20_grid**2 * V_plasma / (C_loss * np.sqrt(E_b100_grid) * np.log10(R_M))  # MW

    # Calculate beam-target fusion properly with n² dependence at each grid point
    print(f"Calculating beam-target physics for {n_grid_points}x{n_grid_points} grid points...")

    # Initialize arrays for full grid calculation
    P_fusion_beam_target = np.zeros_like(E_b100_grid)
    Q_beam_target = np.zeros_like(E_b100_grid)

    # Calculate for each grid point (density and energy dependent)
    for i in range(n_grid_points):
        for j in range(n_grid_points):
            E_b_100_point = E_b100_grid[i, j]  # Energy in 100 keV units
            n_20_point = n_20_grid[i, j]       # Density in 10^20 m^-3
            E_NBI_keV = E_b_100_point * 100    # Convert to keV

            try:
                # Calculate temperatures using new system
                T_i, T_e = calculate_temperatures(E_NBI_keV)

                # Use the SAME volume that was already calculated for the grid
                V = V_plasma[i, j]  # Use the grid volume, not recalculating!

                # Calculate fusion power using paper formula - WITH CORRECT DENSITY AND VOLUME
                P_fusion = calculate_fusion_power_paper(E_b_100_point, n_20_point, V, T_i)

                # Q calculation using consistent volumes (P_NBI already calculated with same V)
                if P_NBI_required[i, j] > 0:
                    Q = P_fusion / P_NBI_required[i, j]
                else:
                    Q = 0

                P_fusion_beam_target[i, j] = P_fusion
                Q_beam_target[i, j] = Q

            except Exception as e:
                print(f"Error at E_b={E_NBI_keV:.1f} keV, n_20={n_20_point:.2f}: {e}")
                P_fusion_beam_target[i, j] = 0
                Q_beam_target[i, j] = 0

    # NWL scales with fusion power and inversely with vessel surface area
    # NWL should only include neutron energy: 14.1 MeV out of 17.6 MeV total
    neutron_fraction = 14.1 / 17.6  # Fraction of fusion energy carried by neutrons

    # Calculate NWL using the corrected fusion power (already calculated for each grid point)
    NWL_beam_target = P_fusion_beam_target * neutron_fraction / vessel_surface_area

    # Calculate effective a0 limit based on size_limit_type and N_rho
    if size_limit_type == 'a0':
        a0_eff_limit = size_limit_value  # Direct radius limit
    elif size_limit_type == 'L':
        # L = N_rho * a0, so a0 = L / N_rho
        a0_eff_limit = size_limit_value / N_rho
    elif size_limit_type == 'V':
        # V = π * N_rho * a0³, so a0 = (V / (π * N_rho))^(1/3)
        a0_eff_limit = (size_limit_value / (np.pi * N_rho))**(1/3)
    else:
        a0_eff_limit = 1.5  # Fallback

    # Create masks for different regions
    mask_beta = n_20_grid > n_20_beta_limit                    # Above beta limit
    mask_impractical = a_0_min > a0_eff_limit                 # Device too large (using dynamic limit)
    mask_low_NWL = NWL_beam_target < 0.0                     # NWL too low

    # Region priorities (in order):
    # 1. Gray: Invalid physics/engineering (beta limit or impractical size)
    # 2. White: Low NWL performance
    # 3. Colored: Good operating regions

    mask_gray = mask_beta | mask_impractical
    mask_black = np.zeros_like(mask_gray, dtype=bool)  # No black regions
    mask_white = (~mask_gray) & mask_low_NWL

    # Fill regions in order
    # 1. Gray regions (invalid)
    ax.contourf(E_b100_grid, n_20_grid, mask_gray.astype(int),
                levels=[0.5, 1.5], colors=['lightgray'], alpha=0.8)

    # 2. Black regions (low required power)
    ax.contourf(E_b100_grid, n_20_grid, mask_black.astype(int),
                levels=[0.5, 1.5], colors=['black'], alpha=0.8)

    # 3. White regions remain white (matplotlib default background)

    # Plot NWL contours - using beam-target physics
    NWL_valid = NWL_beam_target.copy()
    NWL_valid[mask_gray | mask_black | mask_white] = np.nan  # Mask all non-contour regions

    im = ax.contourf(E_b100_grid, n_20_grid, NWL_valid,
                     levels=NWL_background, cmap='viridis', extend='max')

    # Beta limit line
    ax.plot(E_b100, n_20_beta_limit[0,:], 'purple', linewidth=3, zorder=5,
            label='Beta limit')

    # Size limit boundary line - show where a_0_min = a0_eff_limit
    size_limit_boundary = a_0_min - a0_eff_limit  # Zero where a_0_min = a0_eff_limit
    CS_size = ax.contour(E_b100_grid, n_20_grid, size_limit_boundary,
                        levels=[0], colors=['darkred'], linewidths=2, linestyles='--', zorder=4)

    # Label based on size constraint type
    if size_limit_type == 'a0':
        size_label = f'a₀={size_limit_value:.1f}m limit'
    elif size_limit_type == 'L':
        size_label = f'L={size_limit_value:.0f}m limit'
    elif size_limit_type == 'V':
        size_label = f'V={size_limit_value:.0f}m³ limit'
    else:
        size_label = 'Size limit'

    # Add to legend manually
    ax.plot([], [], color='darkred', linewidth=2, linestyle='--', label=size_label)

    # a₀ contours (same as original)
    a_0_min_valid = a_0_min.copy()
    a_0_min_valid[mask_gray | mask_black | mask_white] = np.nan # Mask all non-contour regions

    a0_levels = np.arange(a0_min, a0_max, a0_step)
    CS = ax.contour(E_b100_grid, n_20_grid, a_0_min_valid,
                    levels=a0_levels, colors='pink', linewidths=1.5, alpha=0.9)

    for level in a0_levels:
        # Calculate L and V for this a₀ level
        L_val = 25.0 * level  # L = N_ρ * a₀ (N_ρ = 25)
        V_val = np.pi * 25.0 * level**3  # V = π * N_ρ * a₀³
        label = f'a₀={level:.1f}m'
        ax.clabel(CS, levels=[level], inline=True, fontsize=7, fmt=label)

    # Q contour lines - using beam-target physics (density-invariant)
    Q_valid = Q_beam_target.copy()
    Q_valid[mask_gray | mask_black | mask_white] = np.nan # Mask all non-contour regions

    CS_Q = ax.contour(E_b100_grid, n_20_grid, Q_valid,
                      levels=Q_levels, colors='cyan', linewidths=1.5,
                      alpha=0.8, linestyles='-')
    ax.clabel(CS_Q, inline=True, fontsize=8, fmt='Q=%.2f')

    # P_fusion contour lines - using beam-target physics (density-invariant)
    P_fusion_valid = P_fusion_beam_target.copy()
    P_fusion_valid[mask_gray | mask_black | mask_white] = np.nan

    CS_Pfus = ax.contour(E_b100_grid, n_20_grid, P_fusion_valid,
                         levels=P_fus_levels, colors='magenta', linewidths=1.5,
                         alpha=0.8, linestyles='-')
    ax.clabel(CS_Pfus, inline=True, fontsize=8, fmt='P_fus=%.0f MW')

    # NWL contour lines - using beam-target physics
    CS_NWL = ax.contour(E_b100_grid, n_20_grid, NWL_valid,
                        levels=NWL_levels, colors='white', linewidths=1.0,
                        alpha=0.9, linestyles='-')
    ax.clabel(CS_NWL, inline=True, fontsize=8, fmt='%.1f MW/m²')

    # B₀ contours with diamagnetic effects
    B_0_valid = B_0_grid.copy()
    B_0_valid[mask_gray | mask_black | mask_white] = np.nan
    CS_B0 = ax.contour(E_b100_grid, n_20_grid, B_0_valid,
                       levels=B_0_levels, colors='orange', linewidths=1.5,
                       alpha=0.7, linestyles='-')
    ax.clabel(CS_B0, inline=True, fontsize=8, fmt='B₀=%.1f T')

    # P_NBI contours - solid lines with strong color
    P_NBI_valid = P_NBI_required.copy()
    P_NBI_valid[mask_gray | mask_black | mask_white] = np.nan
    CS_PNBI = ax.contour(E_b100_grid, n_20_grid, P_NBI_valid,
                         levels=P_NBI_levels, colors='red', linewidths=2.5,
                         alpha=1.0, linestyles='-')
    ax.clabel(CS_PNBI, inline=True, fontsize=8, fmt='P_NBI=%.0f MW')

    # Beta contours: β = 3*n₂₀*E_b/B₀²
    beta_valid = beta_local.copy()
    beta_valid[mask_gray | mask_black | mask_white] = np.nan
    beta_levels = np.array([0.05, 0.15, 0.25])  # Beta levels
    CS_beta = ax.contour(E_b100_grid, n_20_grid, beta_valid,
                        levels=beta_levels, colors='orange', linewidths=1.5,
                        alpha=0.8, linestyles='-.')
    ax.clabel(CS_beta, inline=True, fontsize=8, fmt='β=%.2f')

    # C (Loss Coefficient) contours
    C_valid = C_loss.copy()
    C_valid[mask_gray | mask_black | mask_white] = np.nan
    C_levels = np.array([])  # Loss coefficient levels [s]
    CS_C = ax.contour(E_b100_grid, n_20_grid, C_valid,
                     levels=C_levels, colors='brown', linewidths=1.5,
                     alpha=0.8, linestyles=':')
    ax.clabel(CS_C, inline=True, fontsize=8, fmt='C=%.2f s')

    # Formatting
    ax.set_xlabel(r'$E_{NBI}$ [100 keV]', fontsize=14)
    ax.set_ylabel(r'$\langle n_{20} \rangle$ [$10^{20}$ m$^{-3}$]', fontsize=14)
    ax.set_xlim([E_b_min, E_b_max])
    ax.set_ylim([0, n_20_max])

    # Force linear tick formatting (no scientific notation)
    ax.ticklabel_format(style='plain', axis='x')
    ax.ticklabel_format(style='plain', axis='y')
    # Set explicit x-axis ticks for clean decimal display
    x_ticks = np.arange(E_b_min, E_b_max, 0.2)  # [0.4, 0.6, 0.8, 1.0, 1.2]
    ax.set_xticks(x_ticks)

    # Legend
    ax.legend(loc='lower left', fontsize=10)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r'NWL [MW/m²] (Beam-Target Fusion)', fontsize=12)
    cbar.set_ticks(NWL_levels)  # Use clean decimal points for ticks
    cbar.ax.tick_params(labelsize=10)

    # Grid
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)

    # Title with temperature scaling info and size constraint
    if use_temperature_override:
        temp_info = f"$T_i={T_i_fixed:.0f}$keV, $T_e={T_e_fixed:.0f}$keV (fixed)"
    else:
        temp_info = f"$T_i={T_i_coeff:.2f}E_{{NBI}}$, $T_e={T_e_coeff:.2f}E_{{NBI}}$ (beam-dependent)"

    # Size constraint info
    if size_limit_type == 'a0':
        size_info = f'a₀≤{size_limit_value:.1f}m'
    elif size_limit_type == 'L':
        size_info = f'L≤{size_limit_value:.0f}m'
    elif size_limit_type == 'V':
        size_info = f'V≤{size_limit_value:.0f}m³'
    else:
        size_info = ''

    # Title
    ax.set_title(f'($B_{{max}}$={B_max}T, $R_M$={int(R_M)}, '
                 f'$B_0$={B_0:.1f}T, $\\beta_c$={beta_c}, {size_info})\n'
                 f'{temp_info}',
                 fontsize=14, weight='bold')

    plt.tight_layout()
    target_E_b100 = 0.8  # Beam energy in 100 keV units
    target_n_20 = 0.5    # Density in 10^20 m^-3

        # Find closest indices
    E_idx = np.argmin(np.abs(E_b100 - target_E_b100))
    n_idx = np.argmin(np.abs(n_20 - target_n_20))

    # Get NWL value at target point for verification
    nwl_target = NWL_beam_target[n_idx, E_idx]
    print(f"NWL at target point ({target_E_b100:.1f}, {target_n_20:.1f}): {nwl_target:.3f} MW/m²")

    return fig

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def test_loss_coefficient(E_keV=120.0, R_M=7):
    """
    Test function to verify loss coefficient calculation with extracted data

    Parameters:
    -----------
    E_keV : float
        Beam energy in keV (default: 120 keV)
    R_M : float
        Mirror ratio (default: 7)
    """
    E_b_100keV = E_keV / 100.0

    print(f"\n=== Testing Loss Coefficient Calculation ===")
    print(f"Test parameters: E_beam = {E_keV} keV, R_M = {R_M}")

    # Calculate using extracted data
    C_extracted = calculate_loss_coefficient(E_b_100keV, R_M)
    print(f"C@45° (extracted): {C_extracted:.6f} s")

    # Calculate using fallback method for comparison
    C_fallback = calculate_loss_coefficient_fallback(E_b_100keV, R_M)
    print(f"C@45° (fallback): {C_fallback:.6f} s")

    # Show breakdown if extracted data is available
    if len(C_90deg_interp) > 0 and R_M in C_90deg_interp:
        C_90deg = C_90deg_interp[int(round(R_M))](E_keV)
        ratio = ratio_45_90_interp[int(round(R_M))](E_keV)
        print(f"Breakdown:")
        print(f"  C@90° = {C_90deg:.6f}")
        print(f"  C@45°/C@90° ratio = {ratio:.6f}")
        print(f"  C@45° = C@90° × ratio = {C_90deg * ratio:.6f}")

    print(f"=== End Test ===\n")

    return C_extracted

def calculate_temperatures(E_NBI_keV):
    """
    Calculate plasma temperatures using the new beam_target.py system.

    Parameters:
    -----------
    E_NBI_keV : float or array, Beam energy in keV

    Returns:
    --------
    T_i_keV, T_e_keV : Temperature values in keV
    """

    if use_temperature_override:
        # Use fixed explicit temperatures
        T_i_keV = T_i_fixed
        T_e_keV = T_e_fixed
    else:
        # Use beam energy coefficients (paper approach)
        T_i_keV = T_i_coeff * E_NBI_keV
        T_e_keV = T_e_coeff * E_NBI_keV

    return T_i_keV, T_e_keV

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Creating Beam-Target Fusion POPCON plots...")

    # Print extracted data status
    if len(C_90deg_interp) > 0:
        print(f"✓ Using extracted graph data for loss coefficient calculation")
        print(f"  Available mirror ratios: {sorted(C_90deg_interp.keys())}")
        print(f"  Available C@45°/C@90° ratios: {sorted(ratio_45_90_interp.keys())}")

        # Test the loss coefficient calculation
        test_loss_coefficient(120.0, R_M_default)
    else:
        print("⚠ Using fallback loss coefficient calculation (no extracted data)")

    # 1. Create main POPCON (forward analysis)
    print("1. Creating main beam-target POPCON...")
    fig_single = create_full_popcon(B_max_default, R_M_default, beta_c_default)
    fig_single.savefig(figures_dir / 'POPCON_Full.png', dpi=300, bbox_inches='tight')

    # 2. Create forward analysis scans
    if create_bmax_rm_grid:
        print("2a. Creating R_M scan plot...")
        fig_rm_scan = create_rm_scan_plot()
        fig_rm_scan.savefig(figures_dir / 'forward_analysis_RM_scan.png', dpi=300, bbox_inches='tight')

        print("2b. Creating B_max scan plot...")
        fig_bmax_scan = create_bmax_scan_plot()
        fig_bmax_scan.savefig(figures_dir / 'forward_analysis_Bmax_scan.png', dpi=300, bbox_inches='tight')



    print("\n=== All Beam-Target POPCON plots completed! ===")
    print("Files saved:")
    print("  1. beam_target_popcon_full.png (Single detailed POPCON)")
    if create_bmax_rm_grid:
        print(f"  2a. forward_analysis_RM_scan.png (R_M scan: {R_M_grid_values})")
        print(f"  2b. forward_analysis_Bmax_scan.png (B_max scan: {B_max_grid_values})")

    plt.show()
