import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import os
import pandas as pd
from scipy.interpolate import interp1d
import json

os.makedirs("OCR", exist_ok=True)
os.makedirs("OCR/outputs", exist_ok=True)

# Load and process image
img = cv2.imread('OCR/inputs/C_toplinefilled.png')
if img is None:
    print("ERROR: Could not load image")
    exit(1)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mask_blue = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
kernel = np.ones((2, 2), np.uint8)
mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)

# Interactive calibration
class AxisCalibration:
    def __init__(self, img_rgb):
        self.img = img_rgb
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.points = []
        self.point_values = []
        self.point_index = 0
        self.ax.imshow(self.img)
        self.ax.set_title('Click on x-min (leftmost calibration point)')
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.button_ax = plt.axes([0.8, 0.01, 0.1, 0.04])
        self.done_button = Button(self.button_ax, 'Done')
        self.done_button.on_clicked(self.on_done)

    def onclick(self, event):
        if self.point_index >= 4 or event.xdata is None:
            return
        x, y = int(event.xdata), int(event.ydata)
        self.ax.plot(x, y, 'ro', markersize=8)
        self.points.append((x, y))
        self.point_index += 1
        titles = ['Click on x-max', 'Click on y-min', 'Click on y-max']
        if self.point_index < 4:
            self.ax.set_title(titles[self.point_index-1])
        else:
            self.ax.set_title('All points selected! Click Done.')
        self.fig.canvas.draw()

    def on_done(self, event):
        if self.point_index < 4:
            return
        plt.close(self.fig)
        self.get_point_values()

    def get_point_values(self):
        defaults = ['0', '200', '2.0', '2.8']
        names = ['X-min', 'X-max', 'Y-min', 'Y-max']
        for i, name in enumerate(names):
            fig = plt.figure(figsize=(6, 3))
            fig.suptitle(f'Enter value for {name}')
            axbox = plt.axes([0.2, 0.4, 0.6, 0.1])
            text_box = TextBox(axbox, '', initial=defaults[i])
            value = [None]
            def submit(text):
                try:
                    value[0] = float(text)
                    plt.close(fig)
                except ValueError:
                    pass
            text_box.on_submit(submit)
            submit_ax = plt.axes([0.4, 0.2, 0.2, 0.1])
            Button(submit_ax, 'Submit').on_clicked(lambda e: submit(text_box.text))
            plt.show(block=True)
            self.point_values.append(value[0] if value[0] else float(defaults[i]))

    def get_axis_points(self):
        plt.show(block=True)
        if len(self.points) == 4 and len(self.point_values) == 4:
            return {
                'x_min': {'pixel': self.points[0], 'value': self.point_values[0]},
                'x_max': {'pixel': self.points[1], 'value': self.point_values[1]},
                'y_min': {'pixel': self.points[2], 'value': self.point_values[2]},
                'y_max': {'pixel': self.points[3], 'value': self.point_values[3]},
            }
        return None

def extract_curves_simple(mask, num_curves=3):
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) == 0:
        return []
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    start_y_positions = np.unique(y_coords[x_coords < x_min + 10])
    if len(start_y_positions) == 0:
        return []

    curves = []
    if len(start_y_positions) >= num_curves:
        indices = np.linspace(0, len(start_y_positions)-1, num_curves, dtype=int)
        for idx in indices:
            curves.append([[x_min, start_y_positions[idx]]])
    else:
        for y in start_y_positions:
            curves.append([[x_min, y]])

    for x in range(x_min + 1, x_max + 1):
        y_at_x = np.sort(y_coords[x_coords == x])
        if len(y_at_x) > 0:
            assigned = [False] * len(y_at_x)
            for curve_idx in range(len(curves)):
                if len(curves[curve_idx]) > 0:
                    last_y = curves[curve_idx][-1][1]
                    dists = np.abs(y_at_x - last_y)
                    valid = ~np.array(assigned) & (dists < 30)
                    if np.any(valid):
                        best_idx = np.where(valid)[0][np.argmin(dists[valid])]
                        curves[curve_idx].append([x, y_at_x[best_idx]])
                        assigned[best_idx] = True

    return [np.array(c) for c in curves if len(c) > 50]

# Extract curves
calibrator = AxisCalibration(img_rgb)
axis_pts = calibrator.get_axis_points()
if axis_pts is None:
    exit(1)

a_x = (axis_pts['x_max']['value'] - axis_pts['x_min']['value']) / (axis_pts['x_max']['pixel'][0] - axis_pts['x_min']['pixel'][0])
b_x = axis_pts['x_min']['value'] - a_x * axis_pts['x_min']['pixel'][0]
a_y = (axis_pts['y_max']['value'] - axis_pts['y_min']['value']) / (axis_pts['y_max']['pixel'][1] - axis_pts['y_min']['pixel'][1])
b_y = axis_pts['y_min']['value'] - a_y * axis_pts['y_min']['pixel'][1]

curves_pixel = extract_curves_simple(mask_blue, num_curves=3)
curve_names = ['Top', 'Middle', 'Bottom']
num_points = 1000
x_common = np.linspace(0, 400, num_points)

combined_data = {'E_beam_keV': x_common}
existing_Rm = [4, 8, 16]

for i, curve in enumerate(curves_pixel):
    if i < len(curve_names):
        x_values = a_x * curve[:, 0] + b_x
        y_values = a_y * curve[:, 1] + b_y

        sort_idx = np.argsort(x_values)
        x_values = x_values[sort_idx]
        y_values = y_values[sort_idx]

        valid_range = (x_common >= x_values.min()) & (x_common <= x_values.max())

        if np.sum(valid_range) > 0 and len(x_values) > 1:
            interpolator = interp1d(x_values, y_values, kind='linear',
                                   bounds_error=False, fill_value=np.nan)
            y_interp = interpolator(x_common)
            combined_data[curve_names[i]] = y_interp
            combined_data[f'd_{curve_names[i]}'] = y_interp

# Log interpolation for intermediate Rm values
target_Rm = [5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
all_Rm_values = sorted(existing_Rm + target_Rm)

def log_interpolate(x_target, x1, x2, y1, y2):
    if x1 <= 0 or x2 <= 0 or x_target <= 0:
        return (y1 + y2) / 2
    log_x1, log_x2, log_x_target = np.log10(x1), np.log10(x2), np.log10(x_target)
    if abs(log_x2 - log_x1) < 1e-10:
        return (y1 + y2) / 2
    interpolated = y1 + (y2 - y1) * (log_x_target - log_x1) / (log_x2 - log_x1)
    return interpolated

for i, Rm in enumerate(existing_Rm):
    old_name = f'd_{curve_names[i]}'
    new_name = f'd_Rm{Rm}'
    if old_name in combined_data:
        combined_data[new_name] = combined_data[old_name]

for Rm_target in target_Rm:
    new_curve = np.full(num_points, np.nan)
    for energy_idx in range(num_points):
        values = {}
        for Rm in existing_Rm:
            curve_name = f'd_Rm{Rm}'
            if curve_name in combined_data and not np.isnan(combined_data[curve_name][energy_idx]):
                values[Rm] = combined_data[curve_name][energy_idx]

        if len(values) >= 2:
            interpolated_values = []
            available_Rm = sorted(values.keys())

            for i in range(len(available_Rm)):
                for j in range(i+1, len(available_Rm)):
                    Rm1, Rm2 = available_Rm[i], available_Rm[j]
                    if (Rm1 < Rm_target < Rm2) or (Rm1 > Rm_target > Rm2):
                        interp_value = log_interpolate(Rm_target, Rm1, Rm2, values[Rm1], values[Rm2])
                        if not np.isnan(interp_value):
                            interpolated_values.append(interp_value)

            if len(interpolated_values) > 0:
                new_curve[energy_idx] = np.mean(interpolated_values)

    combined_data[f'd_Rm{Rm_target}'] = new_curve

print(f"\nExtracted {len(curves_pixel)} curves, interpolated to {len(all_Rm_values)} total Rm values")
# ========== CRITICAL DEBUG: Check curve extraction ==========
print(f"\n{'='*60}")
print(f"CURVE EXTRACTION VALIDATION")
print(f"{'='*60}")
print(f"Number of curves extracted: {len(curves_pixel)}")
print(f"Expected: 3 curves for Rm = {existing_Rm}")

# Check what data exists
for i, (curve_name, Rm) in enumerate(zip(curve_names, existing_Rm)):
    old_key = f'd_{curve_name}'
    new_key = f'd_Rm{Rm}'

    print(f"\nCurve #{i+1}: '{curve_name}' → Rm={Rm}")

    if old_key in combined_data:
        data_old = combined_data[old_key]
        n_valid = np.sum(~np.isnan(data_old))
        print(f"  '{old_key}': {n_valid}/{len(data_old)} valid points")
        print(f"  Range: [{np.nanmin(data_old):.4f}, {np.nanmax(data_old):.4f}]")
    else:
        print(f"  '{old_key}': *** NOT FOUND ***")

    if new_key in combined_data:
        data_new = combined_data[new_key]
        n_valid = np.sum(~np.isnan(data_new))
        print(f"  '{new_key}': {n_valid}/{len(data_new)} valid points")
        print(f"  Range: [{np.nanmin(data_new):.4f}, {np.nanmax(data_new):.4f}]")
    else:
        print(f"  '{new_key}': *** NOT FOUND ***")

# Calculate angular correction at 120 keV (HARDCODED)
reference_C_45deg = {4: 230/177, 8: 285/177, 16: 302/177}
target_energy_120 = 120.0
E_idx_120 = np.argmin(np.abs(x_common - target_energy_120))
actual_energy_120 = x_common[E_idx_120]

C_90_at_120 = {}
for Rm in existing_Rm:
    C_val = combined_data[f'd_Rm{Rm}'][E_idx_120]
    if not np.isnan(C_val):
        C_90_at_120[Rm] = C_val

ratios_45_90 = {}
for Rm in existing_Rm:
    if Rm in C_90_at_120 and Rm in reference_C_45deg:
        ratio = reference_C_45deg[Rm] / C_90_at_120[Rm]
        ratios_45_90[Rm] = ratio

fit_Rm = np.array(list(ratios_45_90.keys()))
fit_ratios = np.array(list(ratios_45_90.values()))
log_Rm = np.log10(fit_Rm)
coeffs = np.polyfit(log_Rm, fit_ratios, 1)
b_angular, a_angular = coeffs[0], coeffs[1]

angular_correction_all = {}
for Rm in all_Rm_values:
    angular_correction_all[Rm] = a_angular + b_angular * np.log10(Rm)

print(f"Angular correction: r(Rm) = {a_angular:.4f} + {b_angular:.4f}×log₁₀(Rm)")

# Fit C@90° at all energies
# Fit C@90° at all energies - FIXED VERSION
fit_results_90deg = []
for target_E in np.arange(10, 201, 1.0):
    E_idx = np.argmin(np.abs(x_common - target_E))
    actual_E = x_common[E_idx]
    C_90_at_E = {}

    # FIX: Only use the ORIGINAL extracted curves for fitting, NOT interpolated ones
    for Rm in existing_Rm:  # Changed from all_Rm_values
        C_val = combined_data[f'd_Rm{Rm}'][E_idx]
        if not np.isnan(C_val):
            C_90_at_E[Rm] = C_val

    # Need at least 2 points to fit a line
    if len(C_90_at_E) >= 2:
        Rm_arr = np.array(list(C_90_at_E.keys()))
        C_90_arr = np.array(list(C_90_at_E.values()))
        log_Rm_arr = np.log10(Rm_arr)

        # Linear fit: C@90° = a + b × log₁₀(Rm)
        coeffs = np.polyfit(log_Rm_arr, C_90_arr, 1)
        b_90, a_90 = coeffs[0], coeffs[1]

        # Calculate R² for this fit
        C_90_pred = a_90 + b_90 * log_Rm_arr
        ss_res = np.sum((C_90_arr - C_90_pred)**2)
        ss_tot = np.sum((C_90_arr - np.mean(C_90_arr))**2)
        r_sq = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

        fit_results_90deg.append({
            'E_b_keV': actual_E,
            'a_90deg': a_90,
            'b_90deg': b_90,
            'R_squared_90deg': r_sq
        })

print(f"\n✓ Generated {len(fit_results_90deg)} C@90° fits using ONLY original extracted curves (Rm={existing_Rm})")

# 2D Regression for C@45°
E_b_all, Rm_all, C_45_all = [], [], []
for fit in fit_results_90deg:
    E_b = fit['E_b_keV']
    a_90, b_90 = fit['a_90deg'], fit['b_90deg']
    for Rm in all_Rm_values:
        C_90_fitted = a_90 + b_90 * np.log10(Rm)
        r_ang = a_angular + b_angular * np.log10(Rm)
        C_45 = C_90_fitted * r_ang * 0.1
        E_b_all.append(E_b)
        Rm_all.append(Rm)
        C_45_all.append(C_45)

E_b_all = np.array(E_b_all)
Rm_all = np.array(Rm_all)
C_45_all = np.array(C_45_all)
log_Rm_all = np.log10(Rm_all)

# Design matrix for 2D polynomial
X = np.column_stack([
    np.ones(len(E_b_all)),
    E_b_all,
    log_Rm_all,
    E_b_all**2,
    log_Rm_all**2,
    E_b_all * log_Rm_all
])

coeffs_2d, _, _, _ = np.linalg.lstsq(X, C_45_all, rcond=None)
c0, c1, c2, c3, c4, c5 = coeffs_2d
C_45_pred = X @ coeffs_2d
r_squared_2d = 1 - np.sum((C_45_all - C_45_pred)**2) / np.sum((C_45_all - np.mean(C_45_all))**2)

print(f"\n2D Regression: C@45° = c0 + c1×E_b + c2×log₁₀(Rm) + c3×E_b² + c4×log₁₀(Rm)² + c5×E_b×log₁₀(Rm)")
print(f"c0={c0:.6e}, c1={c1:.6e}, c2={c2:.6e}, c3={c3:.6e}, c4={c4:.6e}, c5={c5:.6e}")
print(f"R² = {r_squared_2d:.6f}")

# Save outputs
regression_2d_data = {
    'coefficients': {'c0': float(c0), 'c1': float(c1), 'c2': float(c2), 'c3': float(c3), 'c4': float(c4), 'c5': float(c5)},
    'R_squared': float(r_squared_2d)
}
with open('OCR/outputs/C45_2D_regression.json', 'w') as f:
    json.dump(regression_2d_data, f, indent=2)

# Generate table of C@45° at 1 keV increments for all Rm from 4-16
# Use GROUND TRUTH pipeline: C@90° data → angular correction
table_energies = np.arange(10, 201, 1.0)
table_Rm = np.arange(4, 17, 1)
table_data = {'E_b_keV': table_energies}

for Rm_val in table_Rm:
    C_45_values_ground_truth = []
    for E_b_val in table_energies:
        E_idx = np.argmin(np.abs(x_common - E_b_val))

        # Get C@90° from extracted/interpolated curves
        col_name = f'd_Rm{Rm_val}'
        if col_name in combined_data:
            C_90 = combined_data[col_name][E_idx]
            if not np.isnan(C_90):
                # Apply angular correction
                r_ang = a_angular + b_angular * np.log10(Rm_val)
                C_45_ground_truth = C_90 * r_ang * 0.1
                C_45_values_ground_truth.append(C_45_ground_truth)
            else:
                C_45_values_ground_truth.append(np.nan)
        else:
            C_45_values_ground_truth.append(np.nan)

    table_data[f'C_45_Rm{Rm_val}_ground_truth'] = C_45_values_ground_truth

# Also add 2D fit predictions for comparison
for Rm_val in table_Rm:
    C_45_values_fit = []
    for E_b_val in table_energies:
        log_Rm_val = np.log10(Rm_val)
        C_45_fit = c0 + c1*E_b_val + c2*log_Rm_val + c3*E_b_val**2 + c4*log_Rm_val**2 + c5*E_b_val*log_Rm_val
        C_45_values_fit.append(C_45_fit)
    table_data[f'C_45_Rm{Rm_val}_2Dfit'] = C_45_values_fit

    # Calculate residuals
    residuals = np.array(table_data[f'C_45_Rm{Rm_val}_ground_truth']) - np.array(C_45_values_fit)
    table_data[f'Residual_Rm{Rm_val}'] = residuals

df_table = pd.DataFrame(table_data)
df_table.to_csv('OCR/outputs/C45_table_1keV_Rm4to16.csv', index=False)
print(f"✓ Saved C@45° table: ground truth + 2D fit + residuals for Rm=4-16 at 1 keV increments")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

# 1. Extracted curves with EXTRAPOLATED Rm shown as dashed lines
fig, ax = plt.subplots(figsize=(12, 7))
colors_main = ['red', 'green', 'blue']
colors_extrap = ['orange', 'cyan', 'magenta', 'brown', 'pink', 'gray', 'olive', 'purple', 'yellow', 'lime']

# Plot extracted (solid lines)
for i, Rm in enumerate(existing_Rm):
    y_data = combined_data[f'd_Rm{Rm}']
    valid = ~np.isnan(y_data)
    ax.plot(x_common[valid], y_data[valid], color=colors_main[i], linewidth=2.5,
            label=f'Rm = {Rm} (extracted)', linestyle='-')

# Plot interpolated (dashed lines)
for i, Rm in enumerate(target_Rm):
    y_data = combined_data[f'd_Rm{Rm}']
    valid = ~np.isnan(y_data)
    if np.sum(valid) > 0:
        ax.plot(x_common[valid], y_data[valid], color=colors_extrap[i % len(colors_extrap)],
                linewidth=1.5, label=f'Rm = {Rm} (interpolated)', linestyle='--', alpha=0.7)

ax.set_xlabel('Beam Energy E_b [keV]', fontsize=12, fontweight='bold')
ax.set_ylabel('C@90° [s]', fontsize=12, fontweight='bold')
ax.set_title('Extracted & Interpolated Curves: C@90° vs Beam Energy', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, ncol=2, loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('OCR/outputs/1_extracted_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved extracted curves with interpolated Rm")

# 2. Angular correction fit - TWO PANELS (vs log(Rm) and vs Rm)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: vs log(Rm)
ax1.plot(log_Rm, fit_ratios, 'ro', markersize=10, label='Measured', zorder=3)
log_Rm_fit = np.linspace(log_Rm.min(), log_Rm.max(), 100)
r_fit = a_angular + b_angular * log_Rm_fit
ax1.plot(log_Rm_fit, r_fit, 'b--', linewidth=2, label=f'r = {a_angular:.4f} + {b_angular:.4f}×log₁₀(Rm)', zorder=2)
ax1.set_xlabel('log₁₀(Mirror Ratio)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Angular Correction Factor r(Rm)', fontsize=12, fontweight='bold')
ax1.set_title(f'Angular Correction vs log₁₀(Rm) at E_b = {actual_energy_120:.0f} keV', fontsize=11, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Right panel: vs Rm
Rm_fit_linear = 10**log_Rm_fit
ax2.plot(fit_Rm, fit_ratios, 'ro', markersize=10, label='Measured', zorder=3)
ax2.plot(Rm_fit_linear, r_fit, 'b--', linewidth=2, label=f'r = {a_angular:.4f} + {b_angular:.4f}×log₁₀(Rm)', zorder=2)
ax2.set_xlabel('Mirror Ratio Rm', fontsize=12, fontweight='bold')
ax2.set_ylabel('Angular Correction Factor r(Rm)', fontsize=12, fontweight='bold')
ax2.set_title(f'Angular Correction vs Rm at E_b = {actual_energy_120:.0f} keV', fontsize=11, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('OCR/outputs/2_angular_correction.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved angular correction (2 panels)")


# 3. 2D Heatmap of C@45° (Rm vs E_b) - HEATMAP ONLY, AXES FLIPPED

# Swap axis: now x = Rm, y = E_b
Rm_plot = np.linspace(4, 16, 100)
E_b_plot = np.linspace(E_b_all.min(), E_b_all.max(), 100)
Rm_mesh, E_b_mesh = np.meshgrid(Rm_plot, E_b_plot)
log_Rm_mesh = np.log10(Rm_mesh)
C_45_mesh = c0 + c1*E_b_mesh + c2*log_Rm_mesh + c3*E_b_mesh**2 + c4*log_Rm_mesh**2 + c5*E_b_mesh*log_Rm_mesh

fig, ax = plt.subplots(figsize=(12, 8))
# Note: Passing mesh grids as (x, y, Z) so x = Rm, y = E_b (axis flipped)
contour = ax.contourf(Rm_mesh, E_b_mesh, C_45_mesh, levels=30, cmap='viridis')
cbar = fig.colorbar(contour, ax=ax, label='C@45° [s]')
ax.set_xlabel('Mirror Ratio Rm', fontsize=13, fontweight='bold')
ax.set_ylabel('Beam Energy E_b [keV]', fontsize=13, fontweight='bold')
ax.set_title(f'2D Regression: C@45° = f(E_b, Rm) | R² = {r_squared_2d:.6f}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('OCR/outputs/3_heatmap_C45.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved C@45° 2D heatmap (axes flipped)")

# 4. Residuals for C@90° fits (Rm=4,8,16 vs energy) - FIXED BUG
residuals_90_by_Rm = {Rm: [] for Rm in [4, 8, 16]}
energies_90 = []
for fit in fit_results_90deg:
    E_b_iter = fit['E_b_keV']
    a_90, b_90 = fit['a_90deg'], fit['b_90deg']
    energies_90.append(E_b_iter)
    E_idx = np.argmin(np.abs(x_common - E_b_iter))
    for Rm_iter in [4, 8, 16]:  # Only the extracted ones
        col_name = f'd_Rm{Rm_iter}'
        if col_name in combined_data:
            C_actual = combined_data[col_name][E_idx]
            C_fitted = a_90 + b_90 * np.log10(Rm_iter)
            residual = C_actual - C_fitted if not np.isnan(C_actual) else np.nan
            residuals_90_by_Rm[Rm_iter].append(residual)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(energies_90, residuals_90_by_Rm[4], color='red', linewidth=2, label=f'Rm = 4', alpha=0.8)
ax.plot(energies_90, residuals_90_by_Rm[8], color='green', linewidth=2, label=f'Rm = 8', alpha=0.8)
ax.plot(energies_90, residuals_90_by_Rm[16], color='blue', linewidth=2, label=f'Rm = 16', alpha=0.8)
ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
ax.set_xlabel('Beam Energy E_b [keV]', fontsize=12, fontweight='bold')
ax.set_ylabel('Residual: C@90°(actual) - C@90°(fit) [s]', fontsize=12, fontweight='bold')
ax.set_title('Residuals for C@90° Fits vs Energy', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('OCR/outputs/4_residuals_C90.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved C@90° residuals (bug fixed)")

# 5. Residuals for C@45° vs energy with RELATIVE ERROR on twin axis
residuals_45_by_Rm = {Rm: [] for Rm in [4, 8, 16]}
rel_error_45_by_Rm = {Rm: [] for Rm in [4, 8, 16]}
energies_45 = []
for fit in fit_results_90deg:
    E_b_iter = fit['E_b_keV']
    a_90, b_90 = fit['a_90deg'], fit['b_90deg']
    energies_45.append(E_b_iter)
    for Rm_iter in [4, 8, 16]:
        C_90_fitted = a_90 + b_90 * np.log10(Rm_iter)
        r_ang = a_angular + b_angular * np.log10(Rm_iter)
        C_45_actual = C_90_fitted * r_ang * 0.1

        # Predicted from 2D model
        log_Rm_iter = np.log10(Rm_iter)
        C_45_pred = c0 + c1*E_b_iter + c2*log_Rm_iter + c3*E_b_iter**2 + c4*log_Rm_iter**2 + c5*E_b_iter*log_Rm_iter
        residual = C_45_actual - C_45_pred
        rel_error = (residual / C_45_actual) * 100 if C_45_actual != 0 else 0
        residuals_45_by_Rm[Rm_iter].append(residual)
        rel_error_45_by_Rm[Rm_iter].append(rel_error)

fig, ax1 = plt.subplots(figsize=(14, 7))

# Left y-axis: Absolute residuals
colors_res = ['red', 'green', 'blue']
for i, Rm_iter in enumerate([4, 8, 16]):
    ax1.plot(energies_45, residuals_45_by_Rm[Rm_iter], color=colors_res[i],
             linewidth=2.5, label=f'Rm = {Rm_iter}', alpha=0.8)
ax1.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
ax1.axvline(x=120, color='orange', linestyle=':', linewidth=2.5, label='120 keV (reference)', alpha=0.9)
ax1.set_xlabel('Beam Energy E_b [keV]', fontsize=13, fontweight='bold')
ax1.set_ylabel('Residual [s]', fontsize=13, fontweight='bold', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left', fontsize=11)

# Right y-axis: Relative error
ax2 = ax1.twinx()
for i, Rm_iter in enumerate([4, 8, 16]):
    ax2.plot(energies_45, rel_error_45_by_Rm[Rm_iter], color=colors_res[i],
             linewidth=1.5, linestyle='--', alpha=0.5,
             label=f'Rm = {Rm_iter} (%)' if i == 0 else '')  # Only label once for legend
ax2.set_ylabel('Relative Error [%]', fontsize=13, fontweight='bold', color='gray')
ax2.tick_params(axis='y', labelcolor='gray')
ax2.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax2.legend(['Rel. Error (dashed)'], loc='upper right', fontsize=10, framealpha=0.9)

ax1.set_title('Residuals for C@45° (2D Regression) with Relative Error', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('OCR/outputs/5_residuals_C45_with_rel_error.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved C@45° residuals with relative error on twin axis")

# 6. C@90° vs log(Rm) for every 10 keV
energies_to_plot = [f for f in fit_results_90deg if abs(f['E_b_keV'] % 10) < 0.5 or abs(f['E_b_keV'] % 10 - 10) < 0.5]
n_plots = len(energies_to_plot)
n_cols = 4
n_rows = int(np.ceil(n_plots / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
axes = axes.flatten() if n_plots > 1 else [axes]

for idx, fit in enumerate(energies_to_plot):
    ax = axes[idx]
    E_b_grid = fit['E_b_keV']
    a_90, b_90 = fit['a_90deg'], fit['b_90deg']

    E_idx = np.argmin(np.abs(x_common - E_b_grid))
    Rm_data, C_90_data = [], []
    for Rm_grid in all_Rm_values:
        C_val = combined_data[f'd_Rm{Rm_grid}'][E_idx]
        if not np.isnan(C_val):
            Rm_data.append(Rm_grid)
            C_90_data.append(C_val)

    if len(Rm_data) > 0:
        log_Rm_data = np.log10(Rm_data)
        ax.plot(log_Rm_data, C_90_data, 'bo', markersize=6, label='Data')
        log_Rm_fit_grid = np.linspace(log_Rm_data.min(), log_Rm_data.max(), 50)
        C_fit = a_90 + b_90 * log_Rm_fit_grid
        ax.plot(log_Rm_fit_grid, C_fit, 'r--', linewidth=1.5, label='Fit')
        ax.set_title(f'E_b = {E_b_grid:.0f} keV', fontsize=9)
        ax.set_xlabel('log₁₀(Rm)', fontsize=8)
        ax.set_ylabel('C@90° [s]', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6)

for idx in range(n_plots, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('OCR/outputs/6_C90_vs_Rm_grid.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved C@90° vs Rm grid")

# 7. C@45° vs log(Rm) for every 10 keV
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
axes = axes.flatten() if n_plots > 1 else [axes]

for idx, fit in enumerate(energies_to_plot):
    ax = axes[idx]
    E_b_grid = fit['E_b_keV']
    a_90, b_90 = fit['a_90deg'], fit['b_90deg']

    Rm_array = np.array(all_Rm_values)
    C_45_array = []
    for Rm_grid in Rm_array:
        C_90_fitted = a_90 + b_90 * np.log10(Rm_grid)
        r_ang = a_angular + b_angular * np.log10(Rm_grid)
        C_45 = C_90_fitted * r_ang * 0.1
        C_45_array.append(C_45)

    C_45_array = np.array(C_45_array)
    log_Rm_array = np.log10(Rm_array)

    ax.plot(log_Rm_array, C_45_array, 'ro', markersize=6, label='Data')
    log_Rm_fit_grid = np.linspace(log_Rm_array.min(), log_Rm_array.max(), 50)
    C_45_fit = []
    for log_rm_val in log_Rm_fit_grid:
        C_45_fit.append(c0 + c1*E_b_grid + c2*log_rm_val + c3*E_b_grid**2 + c4*log_rm_val**2 + c5*E_b_grid*log_rm_val)
    ax.plot(log_Rm_fit_grid, C_45_fit, 'g--', linewidth=1.5, label='2D Fit')
    ax.set_title(f'E_b = {E_b_grid:.0f} keV', fontsize=9)
    ax.set_xlabel('log₁₀(Rm)', fontsize=8)
    ax.set_ylabel('C@45° [s]', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6)

    # Highlight 120 keV
    if abs(E_b_grid - 120) < 5:
        for spine in ax.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(3)

for idx in range(n_plots, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('OCR/outputs/7_C45_vs_Rm_grid.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved C@45° vs Rm grid")
