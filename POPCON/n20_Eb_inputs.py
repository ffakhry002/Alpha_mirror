"""n20_Eb_inputs.py - All configurable parameters for n20 vs Eb POPCON"""

import numpy as np
from pathlib import Path

# Magnetic field parameters
B_max_default = 28.0         # Maximum mirror field [T]
B_central_default = 2.8    # Central (conductor) field [T]
beta_c_default = 0.5         # MHD stability limit

# Temperature scaling coefficients (from Egedal et al.)
T_i_coeff = 2/3          # Ti = (2/3)E_b [keV]
T_e_coeff = 0.1          # Te = 0.1E_b [keV]

# NBI efficiency
ETA_ABS = 0.9            # Absorption efficiency
NBI_EFFICIENCY = ETA_ABS

# Grid lifetime and capacity factor parameters
d_grid = 3.0             # Grid erosion thickness limit [mm]
t_replace = 2.5          # Replacement downtime [months]
eta_duty = 7.0/7.0       # Weekly duty cycle (6 days/week)
sigma_x_beam = 4.3       # Horizontal Gaussian beam width [cm]
sigma_y_beam = 10.77     # Vertical Gaussian beam width [cm]
num_grids = 16            # Number of grids to spread power over

# Beam Axis
E_b_min = 0.4            # Minimum beam energy [100 keV units]
E_b_max = 1.2            # Maximum beam energy [100 keV units]

# Practical engineering limit
min_a0 = 0.1                # Minimum minor radius [m]
qw_limit = 5                # Maximum heat flux on end-wall [MW/m^2]

# Grid resolution
n_grid_points = 500      # Number of grid points (higher = smoother but slower)

# Q factor contour levels
Q_levels = np.array([
])

# Neutron wall loading (NWL) levels
max_NWL = 3.5
NWL_background = np.linspace(0, max_NWL, 25)  # Fine resolution for smooth background
NWL_levels = np.array([])  # Clean contour lines

# Fusion power background for POPCON
max_P_fus = 25
P_fus_background = np.linspace(0, max_P_fus, 1000)  # Fine resolution for smooth background

# Revenue per volume background for POPCON [$/yr/m³]
max_rev_per_vol = 500e6  # $20M/yr/m³
Rev_per_Vol_background = np.linspace(0, max_rev_per_vol, 1000)  # Fine resolution

# Minor radius contour levels [m]
a0_levels = np.array([

])

# ⟨P_fus⟩ contour levels (capacity factor adjusted) [MW]
P_fus_levels = np.array([2,4,6,8,10])

# NBI power contour levels [MW]
P_NBI_levels = np.array([10, 20, 30, 40])

# On-axis field contour levels [T]
B_0_levels = np.array([0])

# Beta contour levels
beta_levels = np.array([])

# Loss coefficient contour levels [s]
C_levels = np.array([])

# Mirror ratio contour levels
R_M_levels = np.array([])

# Vortex stabilization applied voltage contours
voltage_levels = np.array([])

# Max Mirror ratio for vortex stabilization contour levels
max_R_M_vortex_levels = np.array([])

# Collisionality contour levels
nu_levels = np.array([])

# Capacity factor contour levels
CF_levels = np.array([])

# End-plug magnetic field levels
Bw_levels = np.array([])

# End-plug radius levels
a_w_levels = np.array([])

# Heat flux contour levels
q_w_levels = np.array([])

# Volume contour levels [m³]
V_levels = np.array([0.5, 1, 2, 3, 4, 5])

test_points_list = [


]

figures_dir = Path(__file__).parent / 'figures'
figures_dir.mkdir(exist_ok=True)

# Figure settings
figure_dpi = 300
figure_size = (11, 8)
