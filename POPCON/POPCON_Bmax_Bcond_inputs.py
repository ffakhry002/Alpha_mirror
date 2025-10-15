"""POPCON_Bmax_Bcond_inputs.py - Configuration for B_max vs B_conductor POPCON"""

import numpy as np
from pathlib import Path

# ============================================================================
# AXIS RANGES - PRIMARY INPUTS
# ============================================================================

r_shield = 0.2      # Shield radius for end cells [m]
r_baker = 0.5       # Baker shield radius for solenoid [m]

# B_max range [T] - maximum mirror field
B_max_min = 20.0
B_max_max = 30.0

# B_conductor range [T] - field at conductor (midplane) = B_max/R_M
B_conductor_min = 2.0
B_conductor_max = 8.0

# Fixed beam energy [keV]
E_b_keV_fixed = 100.0

# ============================================================================
# PHYSICS PARAMETERS
# ============================================================================

# MHD stability limit
beta_c_default = 0.3

# Minimum mirror ratio constraint
R_M_min_limit = 4.0

# Temperature scaling coefficient
T_i_coeff = 2/3  # Ti = (2/3) * E_b [keV]

# Normalized parameters
N_25 = 1.0               # Normalized FLR parameter (fixed at 1)
N_rho = 25 * N_25        # Normalized length parameter

# NBI efficiency
NBI_EFFICIENCY = 0.9 * 0.9  # 81%

# ============================================================================
# GRID RESOLUTION
# ============================================================================

n_grid_points = 300  # Number of grid points per axis (higher = smoother but slower)

# ============================================================================
# CONTOUR LEVELS
# ============================================================================

# Neutron wall loading (NWL) [MW/m²]
# This is the heatmap background
NWL_min = 0.5
NWL_max = 2.0
NWL_background = np.linspace(NWL_min, NWL_max, 30)  # Fine resolution for smooth background
NWL_levels = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0,])  # Contour lines

# Minor radius contour levels [m]
a0_levels = np.array([])

# On-axis field contour levels [T]
B_0_levels = np.array([])

# Mirror ratio contour levels (will be diagonal lines)
R_M_levels = np.array([])

# Q factor contour levels
Q_levels = np.array([])

# NBI power contour levels [MW]
P_NBI_levels = np.array([22,24,26,28,30,32,34,36,38,40])

# HTS power contour levels [kA-m]
HTS_levels = np.array([50000, 100000, 125000, 150000, 175000, 200000, 225000, 250000, 300000, 350000, 400000, 500000])


# Fusion power contour levels [MW]
P_fus_levels = np.array([])

# Density contour levels [10^20 m^-3]
n_20_levels = np.array([])

# Beta contour levels (optional - set to empty to disable)
beta_levels = np.array([])  # Empty = no contours shown
beta_levels = np.array([])  # Uncomment to show

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

# Create figures directory
figures_dir = Path(__file__).parent / 'figures'
figures_dir.mkdir(exist_ok=True)

# Figure settings
figure_dpi = 300
figure_size = (14, 10)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_title_string():
    """Format title for the plot"""
    return (f'Mirror Performance: Optimal Constraint Point ($a_{{0,FLR}} = a_{{0,abs}}$)\n'
            f'$E_b$ = {E_b_keV_fixed:.0f} keV, '
            f'$\\beta_c$ = {beta_c_default}, '
            f'$R_M \\geq {R_M_min_limit:.0f}$, '
            f'$T_i = {T_i_coeff:.2f}E_b$\n'
            f'$r_{{baker}}$ = {r_baker:.1f} m, '
            f'$r_{{shield}}$ = {r_shield:.1f} m')


def print_configuration():
    """Print current configuration"""
    print("="*80)
    print("B_MAX VS B_CONDUCTOR POPCON CONFIGURATION")
    print("="*80)
    print(f"\nAxis Ranges:")
    print(f"  B_max: {B_max_min} - {B_max_max} T")
    print(f"  B_conductor: {B_conductor_min} - {B_conductor_max} T")
    print(f"  Implicit R_M range: {B_max_min/B_conductor_max:.1f} - {B_max_max/B_conductor_min:.1f}")
    print(f"\nFixed Parameters:")
    print(f"  E_b = {E_b_keV_fixed} keV")
    print(f"  beta_c = {beta_c_default}")
    print(f"  R_M_min = {R_M_min_limit}")
    print(f"  T_i = {T_i_coeff} × E_b")
    print(f"\nOperating Point:")
    print(f"  Optimal constraint: a₀,FLR = a₀,abs")
    print(f"  Self-consistent iteration for n₂₀")
    print(f"\nGrid Resolution:")
    print(f"  Grid points: {n_grid_points}×{n_grid_points}")
    print(f"\nContour Levels:")
    print(f"  NWL: {NWL_min} - {NWL_max} MW/m² ({len(NWL_levels)} contours)")
    print(f"  a₀: {len(a0_levels)} levels")
    print(f"  B₀: {len(B_0_levels)} levels")
    print(f"  R_M: {len(R_M_levels)} levels")
    print(f"  Q: {len(Q_levels)} levels")
    print(f"  P_NBI: {len(P_NBI_levels)} levels")
    print(f"  P_fus: {len(P_fus_levels)} levels")
    print(f"  n₂₀: {len(n_20_levels)} levels")
    print(f"  β: {len(beta_levels)} levels")
    print("="*80 + "\n")


def get_example_point():
    """Return an example (B_max, B_conductor) point for analysis"""
    return 26.0, 4.5  # (B_max, B_conductor) in Tesla


if __name__ == "__main__":
    print_configuration()

    print("Example Analysis Point:")
    B_max_ex, B_cond_ex = get_example_point()
    R_M_ex = B_max_ex / B_cond_ex
    print(f"  B_max = {B_max_ex} T")
    print(f"  B_conductor = {B_cond_ex} T")
    print(f"  R_M = {R_M_ex:.2f}")
    print(f"  E_b = {E_b_keV_fixed} keV")
