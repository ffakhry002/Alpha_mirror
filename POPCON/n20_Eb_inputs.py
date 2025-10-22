"""n20_Eb_inputs.py - All configurable parameters for n20 vs Eb POPCON"""

import numpy as np
from pathlib import Path

# Magnetic field parameters
B_max_default = 28.0         # Maximum mirror field [T]
B_central_default = 7.0     # Central (conductor) field [T]
beta_c_default = 0.3         # MHD stability limit

# Temperature scaling coefficients (from Egedal et al.)
T_i_coeff = 2/3          # Ti = (2/3)E_b [keV]
T_e_coeff = 0.1          # Te = 0.1E_b [keV]

# Normalized parameters
N_25 = 1.0               # Normalized FLR parameter (fixed at 1)
N_rho = 25 * N_25        # Normalized length parameter (N_ρ = 25 since N₂₅ = 1)

# NBI efficiency
ETA_ABS = 0.9            # Absorption efficiency
NBI_EFFICIENCY = ETA_ABS

# Beam Axis
E_b_min = 0.4            # Minimum beam energy [100 keV units]
E_b_max = 1.2            # Maximum beam energy [100 keV units]

# Practical engineering limit - minor radius constraint
a0_limit = 1.5              # Maximum minor radius [m]

# Grid resolution
n_grid_points = 500      # Number of grid points (higher = smoother but slower)

# Q factor contour levels
Q_levels = np.array([
    0.23, 0.25
])

# Neutron wall loading (NWL) levels
max_NWL = 3.5
NWL_background = np.linspace(0, max_NWL, 25)  # Fine resolution for smooth background
NWL_levels = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])  # Clean contour lines

# Minor radius contour levels [m]
a0_levels = np.array([
    0.1, 0.15, 0.2, 0.3, 0.5, 1.0
])

# Fusion power contour levels [MW]
P_fus_levels = np.array([])

# NBI power contour levels [MW]
P_NBI_levels = np.array([10, 15, 20, 25, 30])

# On-axis field contour levels [T]
B_0_levels = np.array([0])

# Beta contour levels
beta_levels = np.array([])

# Loss coefficient contour levels [s]
C_levels = np.array([])

# Mirror ratio contour levels
R_M_levels = np.array([4,5,6])

# Collisionality contour levels
nu_levels = np.array([])

test_points_list = [
    (1.0, 1.45),
    (0.8, 0.5),

]

figures_dir = Path(__file__).parent / 'figures'
figures_dir.mkdir(exist_ok=True)

# Figure settings
figure_dpi = 300
figure_size = (11, 8)
