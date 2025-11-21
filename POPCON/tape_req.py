"""
Corrected HTS tape requirements vs B_max
Uses actual plasma physics with diamagnetic corrections at a specific operating point
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from equations import (
    calculate_beta_local,
    calculate_B0_with_diamagnetic,
    calculate_a0_FLR_at_mirror,
    calculate_plasma_geometry_frustum,
    calculate_a0_absorption,
    calculate_a0_FLR,
    calculate_beta_limit,
    calculate_fusion_power,
    calculate_NBI_power,
    calculate_NWL,
    calculate_loss_coefficient
)

from n20_Eb_inputs import (
    beta_c_default,
    N_25,
    N_rho,
    T_i_coeff,
    figures_dir,
    figure_dpi
)


def find_n20_for_target_Pfus(E_b_keV, R_M, target_Pfus, B_max, beta_c, n_20_min=0.01):
    """Solve for n_20 that gives target fusion power using bisection method"""
    E_b_100keV = E_b_keV / 100.0
    B_central = B_max / R_M

    n_20_beta_max = calculate_beta_limit(E_b_100keV, B_central, beta_c)
    beta_local_max = calculate_beta_local(n_20_beta_max, E_b_100keV, B_central)

    if beta_local_max is None or np.isnan(beta_local_max):
        return None, None, None, None, None, None, None, None

    B_0_max = calculate_B0_with_diamagnetic(B_central, beta_local_max)
    a_0_abs_max = calculate_a0_absorption(E_b_100keV, n_20_beta_max)
    a_0_FLR_max = calculate_a0_FLR(E_b_100keV, B_0_max, N_25)
    a_0_min_max = np.maximum(a_0_abs_max, a_0_FLR_max)

    a_0_FLR_mirror_max = calculate_a0_FLR_at_mirror(E_b_100keV, B_max, N_25)
    L_plasma_max, V_plasma_max, vessel_surface_area_max = calculate_plasma_geometry_frustum(
        a_0_min_max, a_0_FLR_mirror_max, N_rho
    )

    T_i = T_i_coeff * E_b_keV
    P_fusion_max = calculate_fusion_power(E_b_100keV, n_20_beta_max, V_plasma_max, T_i)

    if target_Pfus > P_fusion_max * 1.01:
        return None, None, None, None, None, None, None, None

    # Bisection method
    tolerance, max_iterations = 1e-6, 100
    n_20_search_max, n_20_search_min = n_20_beta_max, n_20_min

    for iteration in range(max_iterations):
        n_20_mid = (n_20_search_min + n_20_search_max) / 2
        beta_local = calculate_beta_local(n_20_mid, E_b_100keV, B_central)

        if beta_local is None or np.isnan(beta_local):
            n_20_search_max = n_20_mid
            continue

        B_0 = calculate_B0_with_diamagnetic(B_central, beta_local)
        B_0_conductor = B_central

        a_0_abs = calculate_a0_absorption(E_b_100keV, n_20_mid)
        a_0_FLR = calculate_a0_FLR(E_b_100keV, B_0, N_25)
        a_0_min = np.maximum(a_0_abs, a_0_FLR)

        a_0_FLR_mirror = calculate_a0_FLR_at_mirror(E_b_100keV, B_max, N_25)
        L_plasma, V_plasma, vessel_surface_area = calculate_plasma_geometry_frustum(
            a_0_min, a_0_FLR_mirror, N_rho
        )

        C_loss = calculate_loss_coefficient(E_b_100keV, R_M)
        P_fusion = calculate_fusion_power(E_b_100keV, n_20_mid, V_plasma, T_i)
        NWL_current = calculate_NWL(P_fusion, vessel_surface_area)

        if abs(P_fusion - target_Pfus) / target_Pfus < tolerance:
            P_NBI = calculate_NBI_power(n_20_mid, V_plasma, E_b_100keV, R_M, C_loss)
            return n_20_mid, P_NBI, P_fusion, NWL_current, a_0_abs, a_0_FLR, B_0, B_0_conductor

        if P_fusion < target_Pfus:
            n_20_search_min = n_20_mid
        else:
            n_20_search_max = n_20_mid

        if abs(n_20_search_max - n_20_search_min) < 1e-10:
            break

    return None, None, None, None, None, None, None, None

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================
mu_0 = 4 * np.pi * 1e-7  # H/m

# ============================================================================
# OPERATING POINT SPECIFICATION
# ============================================================================
TARGET_PFUS = 2.0      # Target fusion power [MW]
E_B_KEV = 100             # Beam energy [keV]
R_M_VAC = 8.0             # Vacuum mirror ratio
BETA_C = beta_c_default  # Beta limit

# Engineering margins
R_BAKER = 0.6+0.28          # Breeder/TBM/shield radius [m]
R_SHIELD = 0.6        # End shield radius [m]

# B_max scan range
B_MAX_RANGE = np.array([16, 18, 20, 22, 24, 26, 28, 30])

print(f"\n{'='*80}")
print(f"HTS TAPE REQUIREMENTS vs B_max")
print(f"Operating Point: P_fus={TARGET_PFUS} MW, E_b={E_B_KEV} keV, R_M_vac={R_M_VAC}")
print(f"Beta limit: β_c={BETA_C}")
print(f"{'='*80}\n")

# Storage arrays
results = {
    'B_max': [],
    'B_central': [],
    'B_0': [],
    'n_20': [],
    'beta': [],
    'a_0_min': [],
    'a_0_abs': [],
    'a_0_FLR': [],
    'a_0_FLR_mirror': [],
    'L_plasma': [],
    'V_plasma': [],
    'regime': [],
    'kAm_central_magnets': [],
    'kAm_end_magnets': [],
    'kAm_total': [],
    'P_NBI': [],
    'P_fusion': [],
    'NWL_achieved': []
}

print(f"{'B_max':>7} {'B_cent':>7} {'B_0':>7} {'n_20':>7} {'β':>7} {'a_abs':>7} "
      f"{'a_FLR':>7} {'a_min':>7} {'a_mir':>7} {'L_pl':>7} {'V_pl':>7} "
      f"{'kAm_ctr':>9} {'kAm_end':>9} {'kAm_tot':>9} {'Reg':>5}")
print(f"{'[T]':>7} {'[T]':>7} {'[T]':>7} {'[e20]':>7} {'':>7} {'[m]':>7} "
      f"{'[m]':>7} {'[m]':>7} {'[m]':>7} {'[m]':>7} {'[m³]':>7} "
      f"{'[kA-m]':>9} {'[kA-m]':>9} {'[kA-m]':>9} {'':>5}")
print("-" * 105)

for B_max in B_MAX_RANGE:
    try:
        # Calculate conductor field
        B_central = B_max / R_M_VAC

        # Solve for operating point at this B_max
        result = find_n20_for_target_Pfus(
            E_B_KEV, R_M_VAC, TARGET_PFUS, B_max, BETA_C
        )

        n_20, P_NBI, P_fusion, NWL_achieved, a_0_abs, a_0_FLR, B_0, B_0_conductor = result

        if n_20 is None or np.isnan(P_NBI):
            print(f"{B_max:>7.1f} - FAILED: Cannot achieve target P_fus at this B_max")
            continue

        # Calculate beta
        E_b_100keV = E_B_KEV / 100.0
        beta = calculate_beta_local(n_20, E_b_100keV, B_central)

        # Determine limiting constraint
        a_0_min = max(a_0_abs, a_0_FLR)
        regime = 'Abs' if a_0_abs > a_0_FLR else 'FLR'

        # Calculate mirror radius
        a_0_FLR_mirror = calculate_a0_FLR_at_mirror(E_b_100keV, B_max, N_25)

        # Get plasma geometry
        L_plasma, V_plasma, vessel_surface = calculate_plasma_geometry_frustum(
            a_0_min, a_0_FLR_mirror, N_rho
        )

        # ====================================================================
        # CALCULATE TAPE REQUIREMENTS
        # ====================================================================

        # Central magnets: uses B_central (conductor field) and a_0_min
        # Formula: (4 * π * R^2 * B) / μ_0, with 2 rings
        R_central = 1.1 * a_0_min + R_BAKER
        Am_central_single = (4 * np.pi * R_central**2 * B_central) / mu_0
        kAm_central_single = Am_central_single / 1000
        kAm_central_total = 2 * kAm_central_single

        # End magnets: uses B_max at mirror
        # Formula: (4 * π * R^2 * B) / μ_0, with 2 rings
        R_end = 1.1 * a_0_FLR_mirror + R_SHIELD
        Am_end_single = (4 * np.pi * R_end**2 * B_max) / mu_0
        kAm_end_single = Am_end_single / 1000
        kAm_end_total = 2 * kAm_end_single

        # Total
        kAm_total = kAm_central_total + kAm_end_total

        # Store results
        results['B_max'].append(B_max)
        results['B_central'].append(B_central)
        results['B_0'].append(B_0)
        results['n_20'].append(n_20)
        results['beta'].append(beta)
        results['a_0_min'].append(a_0_min)
        results['a_0_abs'].append(a_0_abs)
        results['a_0_FLR'].append(a_0_FLR)
        results['a_0_FLR_mirror'].append(a_0_FLR_mirror)
        results['L_plasma'].append(L_plasma)
        results['V_plasma'].append(V_plasma)
        results['regime'].append(regime)
        results['kAm_central_magnets'].append(kAm_central_total)
        results['kAm_end_magnets'].append(kAm_end_total)
        results['kAm_total'].append(kAm_total)
        results['P_NBI'].append(P_NBI)
        results['P_fusion'].append(P_fusion)
        results['NWL_achieved'].append(NWL_achieved)

        print(f"{B_max:>7.1f} {B_central:>7.2f} {B_0:>7.3f} {n_20:>7.4f} "
              f"{beta:>7.5f} {a_0_abs:>7.4f} {a_0_FLR:>7.4f} {a_0_min:>7.4f} "
              f"{a_0_FLR_mirror:>7.4f} {L_plasma:>7.2f} {V_plasma:>7.3f} "
              f"{kAm_central_total:>9.1f} {kAm_end_total:>9.1f} {kAm_total:>9.1f} "
              f"{regime:>5}")

    except Exception as e:
        print(f"{B_max:>7.1f} - ERROR: {str(e)}")
        continue

# Convert to numpy arrays
for key in results:
    results[key] = np.array(results[key])

if len(results['B_max']) == 0:
    print("\nERROR: No valid operating points found!")
    sys.exit(1)

print(f"\n{'='*80}")
print(f"SUMMARY:")
print(f"{'='*80}")
print(f"Successfully calculated {len(results['B_max'])} operating points")
print(f"B_max range: {results['B_max'][0]:.1f} - {results['B_max'][-1]:.1f} T")
print(f"kA-m range: {results['kAm_total'][0]:.1f} - {results['kAm_total'][-1]:.1f} kA-m")
print(f"Central magnets fraction at B_max={results['B_max'][0]:.0f}T: "
      f"{100*results['kAm_central_magnets'][0]/results['kAm_total'][0]:.1f}%")
print(f"Central magnets fraction at B_max={results['B_max'][-1]:.0f}T: "
      f"{100*results['kAm_central_magnets'][-1]/results['kAm_total'][-1]:.1f}%")
print(f"\nDiamagnetic effect at B_max={results['B_max'][0]:.0f}T: "
      f"B_0/B_central = {results['B_0'][0]/results['B_central'][0]:.3f} "
      f"({100*(1-results['B_0'][0]/results['B_central'][0]):.1f}% reduction)")
print(f"Diamagnetic effect at B_max={results['B_max'][-1]:.0f}T: "
      f"B_0/B_central = {results['B_0'][-1]/results['B_central'][-1]:.3f} "
      f"({100*(1-results['B_0'][-1]/results['B_central'][-1]):.1f}% reduction)")

# ============================================================================
# CREATE PLOTS
# ============================================================================

fig = plt.figure(figsize=(16, 12))

# Create custom grid: 2 plots on top, 1 in middle
ax1 = plt.subplot(2, 2, 1)  # Top left
ax2 = plt.subplot(2, 2, 2)  # Top right
ax3 = plt.subplot(2, 1, 2)  # Bottom center (spans full width)

# ============================================================================
# PLOT 1: Total kA-m vs B_max (TOP LEFT)
# ============================================================================
ax1.plot(results['B_max'], results['kAm_total'], 'b-', linewidth=3,
         label='Total', marker='o', markersize=6)
ax1.plot(results['B_max'], results['kAm_central_magnets'], 'r--', linewidth=2,
         label='Central Magnets (×2)', marker='s', markersize=5)
ax1.plot(results['B_max'], results['kAm_end_magnets'], 'g--', linewidth=2,
         label='End Magnets (×2)', marker='^', markersize=5)
ax1.set_xlabel('$B_{max}$ [T]', fontsize=13, fontweight='bold')
ax1.set_ylabel('Conductor [kA-m]', fontsize=13, fontweight='bold')
ax1.set_title('Total HTS Conductor vs Maximum Field', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# ============================================================================
# PLOT 2: Breakdown by component (TOP RIGHT)
# ============================================================================
ax2.fill_between(results['B_max'], 0, results['kAm_central_magnets'],
                  alpha=0.5, color='red', label='Central Magnets (×2)')
ax2.fill_between(results['B_max'], results['kAm_central_magnets'], results['kAm_total'],
                  alpha=0.5, color='green', label='End Magnets (×2)')
ax2.plot(results['B_max'], results['kAm_total'], 'b-', linewidth=2.5,
         label='Total', marker='o', markersize=5)
ax2.set_xlabel('$B_{max}$ [T]', fontsize=13, fontweight='bold')
ax2.set_ylabel('Conductor [kA-m]', fontsize=13, fontweight='bold')
ax2.set_title('Conductor Breakdown by Component', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# ============================================================================
# PLOT 3: Plasma Radii (BOTTOM CENTER)
# ============================================================================
ax3.plot(results['B_max'], results['a_0_min'], 'b-', linewidth=3,
         label='$a_{0,min}$ (center)', marker='o', markersize=6)
ax3.plot(results['B_max'], results['a_0_FLR_mirror'], 'r-', linewidth=3,
         label='$a_{0,FLR}$ (mirror)', marker='s', markersize=6)
ax3.plot(results['B_max'], results['a_0_abs'], 'g--', linewidth=2,
         label='$a_{0,abs}$ (absorption)', marker='^', markersize=5, alpha=0.7)
ax3.plot(results['B_max'], results['a_0_FLR'], 'm--', linewidth=2,
         label='$a_{0,FLR}$ (center)', marker='d', markersize=5, alpha=0.7)
ax3.set_xlabel('$B_{max}$ [T]', fontsize=13, fontweight='bold')
ax3.set_ylabel('Minor Radius [m]', fontsize=13, fontweight='bold')
ax3.set_title('Plasma Radii at Operating Point', fontsize=13, fontweight='bold')
ax3.legend(fontsize=11, ncol=2)
ax3.grid(True, alpha=0.3)

# Make bottom plot more square-like
ax3.set_aspect('auto')
ax3_pos = ax3.get_position()
new_width = 0.5
new_left = 0.25
ax3.set_position([new_left, ax3_pos.y0, new_width, ax3_pos.height])

fig.suptitle(f'HTS Conductor Requirements vs Maximum Field\n'
             f'Operating Point: P_fus={TARGET_PFUS} MW, $E_b$={E_B_KEV} keV, '
             f'$R_M$={R_M_VAC}, $\\beta_c$={BETA_C}\n'
             f'$r_{{baker}}$={R_BAKER}m, '
             f'$r_{{shield}}$={R_SHIELD}m',
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout()

output_path = figures_dir / 'tape_requirements_corrected.png'
plt.savefig(output_path, dpi=figure_dpi, bbox_inches='tight')

print(f"\n{'='*80}")
print(f"Plot saved to: {output_path}")
print(f"{'='*80}\n")

plt.show()
