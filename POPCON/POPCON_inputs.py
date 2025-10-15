"""POPCON_inputs.py - All configurable parameters"""

import numpy as np
from pathlib import Path

# ============================================================================
# PHYSICS PARAMETERS
# ============================================================================

# Magnetic field parameters
B_max_default = 26.0      # Maximum mirror field [T]
R_M_default = 5.93        # Mirror ratio
beta_c_default = 0.3      # MHD stability limit

# Temperature scaling coefficients
T_i_coeff = 2/3          # Ti = T_i_coeff * E_b [keV]  (paper default: 2/3)
T_e_coeff = 0.1          # Te = T_e_coeff * E_b [keV]  (paper default: 0.1)

# Normalized parameters
N_25 = 1.0               # Normalized FLR parameter (fixed at 1)
N_rho = 25*N_25            # Normalized length parameter (N_ρ = 25 since N₂₅ = 1)

# NBI efficiency
NBI_EFFICIENCY = 0.9 * 0.9  # Injection efficiency × trapping efficiency = 81%

# ============================================================================
# ENERGY RANGE - APPLIES TO ALL PLOTS
# ============================================================================

E_b_min = 0.8            # Minimum beam energy [100 keV units]
E_b_max = 1.2            # Maximum beam energy [100 keV units]

# ============================================================================
# SIZE CONSTRAINT SYSTEM
# ============================================================================

# Practical engineering limit - configurable size constraint
size_limit_type = 'a0'      # Options: 'a0' (radius), 'L' (length), 'V' (volume)
size_limit_value = 1.5      # Constraint value: a0 [m], L [m], or V [m³]

# ============================================================================
# PLOTTING PARAMETERS
# ============================================================================

# Grid resolution
n_grid_points = 500      # Number of grid points (higher = smoother but slower)

# ============================================================================
# CONTOUR LEVELS FOR POPCON PLOTS
# ============================================================================

# Q factor contour levels
Q_levels = np.array([
    0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3,
    0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
    1.0, 1.2, 1.4, 5.0, 10.0
])

# Neutron wall loading (NWL) levels
max_NWL = 4
NWL_background = np.linspace(0, max_NWL, 25)  # Fine resolution for smooth background
NWL_levels = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])  # Clean contour lines

# Minor radius contour levels [m]
a0_levels = np.array([
    0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3
])

# Fusion power contour levels [MW]
P_fus_levels = np.array([])  # Empty = no contours shown

# NBI power contour levels [MW]
P_NBI_levels = np.array([10, 20, 25.52, 30, 40, 50, 100, 200, 500, 1000])

# On-axis field contour levels [T]
B_0_levels = np.array([4,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9,5.0,5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8,5.9,6.0])  # Empty = no contours shown

# Beta contour levels
beta_levels = np.array([0.3])  # Empty = no contours shown

# Loss coefficient contour levels [s]
C_levels = np.array([])  # Empty = no contours shown

# ============================================================================
# TEST POINT ANALYSIS
# ============================================================================

# Specific design point to analyze in detail
test_point_E_b100 = 1.0   # Beam energy [100 keV units]
test_point_n_20 = 1.45    # Density [10^20 m^-3]

# Multiple test points (optional - set to empty list to disable)
# Each tuple is (E_b100, n_20)
test_points_list = [
    (1.0, 1.45),
    (0.8, 0.5),
    # Add more test points here as needed
]

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

# Create figures directory
figures_dir = Path(__file__).parent / 'figures'
figures_dir.mkdir(exist_ok=True)

# Figure settings
figure_dpi = 300
figure_size = (11, 8)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_size_limit_label():
    """Format label for size constraint"""
    if size_limit_type == 'a0':
        return f'a₀≤{size_limit_value:.1f}m'
    elif size_limit_type == 'L':
        return f'L≤{size_limit_value:.0f}m'
    elif size_limit_type == 'V':
        return f'V≤{size_limit_value:.0f}m³'
    else:
        return 'Size limit'


def calculate_effective_a0_limit(N_rho=N_rho):
    """Convert size_limit to effective a0 limit based on type (a0, L, or V)"""
    if size_limit_type == 'a0':
        return size_limit_value
    elif size_limit_type == 'L':
        # L = N_rho * a0, so a0 = L / N_rho
        return size_limit_value / N_rho
    elif size_limit_type == 'V':
        # V = π * N_rho * a0³, so a0 = (V / (π * N_rho))^(1/3)
        return (size_limit_value / (np.pi * N_rho))**(1/3)
    else:
        return 1.5  # Fallback


def get_temperature_info_string():
    """Format temperature scaling: T_i = (2/3)E_NBI, T_e = 0.1E_NBI"""
    return f"$T_i={T_i_coeff:.2f}E_{{NBI}}$, $T_e={T_e_coeff:.2f}E_{{NBI}}$ (beam-dependent)"


def print_configuration():
    """Print current configuration"""
    print("="*70)
    print("POPCON CONFIGURATION")
    print("="*70)
    print(f"\nPhysics Parameters:")
    print(f"  B_max = {B_max_default} T")
    print(f"  R_M = {R_M_default}")
    print(f"  beta_c = {beta_c_default}")
    print(f"  B_0 = {B_max_default/R_M_default:.2f} T")
    print(f"\nTemperature Scaling:")
    print(f"  T_i = {T_i_coeff} × E_NBI")
    print(f"  T_e = {T_e_coeff} × E_NBI")
    print(f"\nEnergy Range:")
    print(f"  E_b: {E_b_min} - {E_b_max} × 100 keV")
    print(f"\nSize Constraint:")
    print(f"  Type: {size_limit_type}")
    print(f"  Value: {size_limit_value}")
    print(f"  Label: {get_size_limit_label()}")
    print(f"\nGrid Resolution:")
    print(f"  Grid points: {n_grid_points}×{n_grid_points}")
    print(f"\nTest Point:")
    print(f"  E_b = {test_point_E_b100 * 100} keV")
    print(f"  n_20 = {test_point_n_20}")
    print("="*70 + "\n")


if __name__ == "__main__":
    print_configuration()

    print("Example usage:")
    print(f"  Size limit label: {get_size_limit_label()}")
    print(f"  Effective a0 limit: {calculate_effective_a0_limit():.3f} m")
    print(f"  Temperature info: {get_temperature_info_string()}")
