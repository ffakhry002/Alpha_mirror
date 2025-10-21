"""
Simple kA-m vs B_max calculation for R_M = 4, FLR-limited
Just geometry - no need for density solving or fusion physics
"""

import numpy as np
import matplotlib.pyplot as plt

# Physical constants
mu_0 = 4 * np.pi * 1e-7  # H/m

# Fixed parameters
R_M = 4.0  # Mirror ratio
E_b_keV = 60.0  # Beam energy
E_b_100keV = E_b_keV / 100.0

# FLR parameters
N_25 = 1.0  # Normalized FLR parameter
N_rho = 25.0  # Length parameter

# Engineering margins
r_baker = 1.0  # Breeder/TBM/shield radius [m]
r_shield = 0.75  # End shield radius [m]

# Scan B_max
B_max_range = np.linspace(15, 35, 100)

# Storage
kAm_solenoid_array = []
kAm_ring_array = []
kAm_total_array = []
a_0_center_array = []
a_0_mirror_array = []
B_central_array = []

print(f"\n{'='*70}")
print(f"kA-m vs B_max for Fixed Mirror Ratio")
print(f"R_M = {R_M}, E_b = {E_b_keV} keV")
print(f"FLR-limited: a_0 = N_25 * sqrt(E_b) / B_0")
print(f"{'='*70}\n")

print(f"{'B_max':>8} {'B_central':>10} {'a_0_center':>12} {'a_0_mirror':>12} "
      f"{'kAm_sol':>10} {'kAm_rings':>10} {'kAm_total':>10}")
print(f"{'[T]':>8} {'[T]':>10} {'[m]':>12} {'[m]':>12} "
      f"{'[kA-m]':>10} {'[kA-m]':>10} {'[kA-m]':>10}")
print(f"{'-'*80}")

for B_max in B_max_range:
    # Calculate B_central from mirror ratio
    B_central = B_max / R_M

    # For simplicity, assume no diamagnetic effect (or use B_central as B_0)
    # This gives us the FLR radius at center
    B_0 = B_central  # Simplified - could add diamagnetic correction if needed

    # FLR radius at center
    a_0_FLR_center = N_25 * np.sqrt(E_b_100keV) / B_0

    # FLR radius at mirror
    a_0_FLR_mirror = N_25 * np.sqrt(E_b_100keV) / B_max

    # =================================================================
    # SOLENOID kA-m
    # =================================================================
    # From your document:
    # A-m_solenoid = (2π R B_central L) / μ₀
    # where R = 1.1 * (a_0_min + a_0_FLR) / 2 + r_baker
    #       L = N_ρ * a_0_min

    R_solenoid = 1.1 * (a_0_FLR_center + a_0_FLR_mirror) / 2 + r_baker
    L_solenoid = N_rho * a_0_FLR_center

    Am_solenoid = (2 * np.pi * R_solenoid * B_central * L_solenoid) / mu_0
    kAm_solenoid = Am_solenoid / 1000  # Convert to kA-m

    # =================================================================
    # END RING kA-m (for both rings)
    # =================================================================
    # From your document:
    # A-m_ring = (4π R² B_max) / μ₀
    # where R = 1.1 * a_0_FLR + r_shield

    R_ring = 1.1 * a_0_FLR_mirror + r_shield

    Am_ring_single = (4 * np.pi * R_ring**2 * B_max) / mu_0
    kAm_ring_single = Am_ring_single / 1000
    kAm_ring_total = 2 * kAm_ring_single  # Two end rings

    # Total
    kAm_total = kAm_solenoid + kAm_ring_total

    # Store
    B_central_array.append(B_central)
    a_0_center_array.append(a_0_FLR_center)
    a_0_mirror_array.append(a_0_FLR_mirror)
    kAm_solenoid_array.append(kAm_solenoid)
    kAm_ring_array.append(kAm_ring_total)
    kAm_total_array.append(kAm_total)

    print(f"{B_max:>8.1f} {B_central:>10.2f} {a_0_FLR_center:>12.4f} {a_0_FLR_mirror:>12.4f} "
          f"{kAm_solenoid:>10.1f} {kAm_ring_total:>10.1f} {kAm_total:>10.1f}")

# Convert to arrays
B_max_range = np.array(B_max_range)
kAm_total_array = np.array(kAm_total_array)
kAm_solenoid_array = np.array(kAm_solenoid_array)
kAm_ring_array = np.array(kAm_ring_array)
a_0_center_array = np.array(a_0_center_array)
a_0_mirror_array = np.array(a_0_mirror_array)

print(f"\n{'='*70}")
print(f"Summary:")
print(f"{'='*70}")
print(f"B_max range: {B_max_range[0]:.1f} - {B_max_range[-1]:.1f} T")
print(f"kA-m range: {kAm_total_array[0]:.1f} - {kAm_total_array[-1]:.1f} kA-m")
print(f"Solenoid fraction at B_max={B_max_range[0]:.0f}T: {100*kAm_solenoid_array[0]/kAm_total_array[0]:.1f}%")
print(f"Solenoid fraction at B_max={B_max_range[-1]:.0f}T: {100*kAm_solenoid_array[-1]/kAm_total_array[-1]:.1f}%")

# =================================================================
# CREATE PLOT - TRIANGLE LAYOUT
# =================================================================

fig = plt.figure(figsize=(16, 12))

# Create custom grid: 2 plots on top, 1 in middle
ax1 = plt.subplot(2, 2, 1)  # Top left
ax2 = plt.subplot(2, 2, 2)  # Top right
ax3 = plt.subplot(2, 1, 2)  # Bottom center (spans full width)

# Plot 1: Total kA-m vs B_max (TOP LEFT)
ax1.plot(B_max_range, kAm_total_array, 'b-', linewidth=3, label='Total')
ax1.plot(B_max_range, kAm_solenoid_array, 'r--', linewidth=2, label='Solenoid')
ax1.plot(B_max_range, kAm_ring_array, 'g--', linewidth=2, label='End Rings (×2)')
ax1.set_xlabel('$B_{max}$ [T]', fontsize=13, fontweight='bold')
ax1.set_ylabel('Conductor [kA-m]', fontsize=13, fontweight='bold')
ax1.set_title('Total HTS Conductor vs Maximum Field', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: Breakdown by component (TOP RIGHT)
ax2.fill_between(B_max_range, 0, kAm_solenoid_array,
                  alpha=0.5, color='red', label='Solenoid')
ax2.fill_between(B_max_range, kAm_solenoid_array, kAm_total_array,
                  alpha=0.5, color='green', label='End Rings')
ax2.plot(B_max_range, kAm_total_array, 'b-', linewidth=2, label='Total')
ax2.set_xlabel('$B_{max}$ [T]', fontsize=13, fontweight='bold')
ax2.set_ylabel('Conductor [kA-m]', fontsize=13, fontweight='bold')
ax2.set_title('Conductor Breakdown by Component', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# Plot 3: FLR-Limited Plasma Radii (BOTTOM CENTER - SQUARE)
ax3.plot(B_max_range, a_0_center_array, 'b-', linewidth=3, label='$a_{0,FLR}$ (center)')
ax3.plot(B_max_range, a_0_mirror_array, 'r-', linewidth=3, label='$a_{0,FLR}$ (mirror)')
ax3.set_xlabel('$B_{max}$ [T]', fontsize=13, fontweight='bold')
ax3.set_ylabel('Minor Radius [m]', fontsize=13, fontweight='bold')
ax3.set_title('FLR-Limited Plasma Radii', fontsize=13, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

# Make bottom plot more square-like
ax3.set_aspect('auto')
ax3_pos = ax3.get_position()
# Center it and make it more square
new_width = 0.5
new_left = 0.25  # Center horizontally
ax3.set_position([new_left, ax3_pos.y0, new_width, ax3_pos.height])

fig.suptitle(f'HTS Conductor Requirements vs Maximum Field\n'
             f'($R_M$ = {R_M}, $E_b$ = {E_b_keV} keV, FLR-limited, '
             f'$r_{{baker}}$ = {r_baker} m, $r_{{shield}}$ = {r_shield} m)',
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig('figures/kAm_vs_Bmax_simple.png', dpi=300, bbox_inches='tight')

print(f"\n{'='*70}\n")

plt.show()
