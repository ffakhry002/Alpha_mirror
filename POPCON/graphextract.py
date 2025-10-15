import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import os
import pandas as pd
from scipy.interpolate import interp1d
import gc  # For garbage collection

os.makedirs("OCR", exist_ok=True)
img_path = 'OCR/inputs/C_toplinefilled.png'  # Change this to your image path
img = cv2.imread(img_path)
if img is None:
    print(f"ERROR: Could not load image from {img_path}")
    exit(1)

# Convert to HSV for better color detection
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])

# Create mask for blue colors
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
kernel = np.ones((2, 2), np.uint8)
mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
cv2.imwrite('OCR/mask.png', mask_blue)

# --- Interactive Axis Calibration ---
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
        self.ax.text(x+10, y, f'Point {self.point_index+1}',
                    color='white', backgroundcolor='black', fontsize=8)

        self.points.append((x, y))
        self.point_index += 1

        titles = ['Click on x-max', 'Click on y-min', 'Click on y-max']
        if self.point_index < 4:
            self.ax.set_title(titles[self.point_index-1] if self.point_index > 0 else titles[0])
        else:
            self.ax.set_title('All points selected! Click Done.')

        self.fig.canvas.draw()

    def on_done(self, event):
        if self.point_index < 4:
            self.ax.set_title('Please click all 4 points first!')
            self.fig.canvas.draw()
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
            submit_btn = Button(submit_ax, 'Submit')
            submit_btn.on_clicked(lambda e: submit(text_box.text))

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

# Better curve extraction with continuity tracking
def extract_curves_simple(mask, num_curves=3):
    """
    Extract curves with continuity tracking to prevent jumping
    """
    y_coords, x_coords = np.where(mask > 0)

    if len(x_coords) == 0:
        return []

    # Get range
    x_min = np.min(x_coords)
    x_max = np.max(x_coords)

    start_x = x_min
    start_y_positions = []

    for x in range(start_x, min(start_x + 10, x_max)):
        y_at_x = y_coords[x_coords == x]
        if len(y_at_x) > 0:
            start_y_positions.extend(y_at_x)

    if len(start_y_positions) == 0:
        return []

    start_y_positions = np.unique(start_y_positions)
    curves = []
    if len(start_y_positions) >= num_curves:
        indices = np.linspace(0, len(start_y_positions)-1, num_curves, dtype=int)
        for idx in indices:
            curves.append([[start_x, start_y_positions[idx]]])
    else:
        for y in start_y_positions:
            curves.append([[start_x, y]])
        while len(curves) < num_curves:
            curves.append([])


    # Track each curve with strict continuity
    for x in range(start_x + 1, x_max + 1):
        y_at_x = y_coords[x_coords == x]

        if len(y_at_x) > 0:
            y_at_x = np.sort(y_at_x)
            assigned = [False] * len(y_at_x)

            for curve_idx in range(num_curves):
                if len(curves[curve_idx]) > 0:
                    last_point = curves[curve_idx][-1]
                    last_y = last_point[1]

                    min_dist = float('inf')
                    best_idx = -1

                    for j, y in enumerate(y_at_x):
                        if not assigned[j]:
                            dist = abs(y - last_y)
                            if dist < min_dist and dist < 30:
                                min_dist = dist
                                best_idx = j

                    if best_idx >= 0:
                        curves[curve_idx].append([x, y_at_x[best_idx]])
                        assigned[best_idx] = True

    result = []
    for i, curve in enumerate(curves):
        if len(curve) > 50:
            curve_array = np.array(curve)
            result.append(curve_array)

    return result

# Main execution
calibrator = AxisCalibration(img_rgb)
axis_pts = calibrator.get_axis_points()

if axis_pts is None:
    exit(1)

# Calculate transformation parameters
a_x = (axis_pts['x_max']['value'] - axis_pts['x_min']['value']) / \
      (axis_pts['x_max']['pixel'][0] - axis_pts['x_min']['pixel'][0])
b_x = axis_pts['x_min']['value'] - a_x * axis_pts['x_min']['pixel'][0]

a_y = (axis_pts['y_max']['value'] - axis_pts['y_min']['value']) / \
      (axis_pts['y_max']['pixel'][1] - axis_pts['y_min']['pixel'][1])
b_y = axis_pts['y_min']['value'] - a_y * axis_pts['y_min']['pixel'][1]
curves_pixel = extract_curves_simple(mask_blue, num_curves=3)
curve_names = ['Top', 'Middle', 'Bottom']

x_min = 0
x_max = 400

num_points = 1000
x_common = np.linspace(x_min, x_max, num_points)

combined_data = {'E_beam_keV': x_common}

for i, curve in enumerate(curves_pixel):
    if i < len(curve_names):
        x_values = a_x * curve[:, 0] + b_x
        y_values = a_y * curve[:, 1] + b_y

        sort_idx = np.argsort(x_values)
        x_values = x_values[sort_idx]
        y_values = y_values[sort_idx]

        # Interpolate to common x-axis
        # Only interpolate within the data range
        valid_range = (x_common >= x_values.min()) & (x_common <= x_values.max())
        y_interp = np.full(num_points, np.nan)

        if len(x_values) > 1:
            f = interp1d(x_values, y_values, kind='linear', bounds_error=False, fill_value=np.nan)
            y_interp[valid_range] = f(x_common[valid_range])

        combined_data[f'd_{curve_names[i]}'] = y_interp
# Add logarithmic interpolation for Rm=5, 6, 7, 9, 10, 11, 12, 13, 14, and 15
# Mirror ratios for existing curves and target curves
existing_Rm = [4, 8, 16]  # Top, Middle, Bottom
target_Rm = [5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
target_names = ['Rm5', 'Rm6', 'Rm7', 'Rm9', 'Rm10', 'Rm11', 'Rm12', 'Rm13', 'Rm14', 'Rm15']

def log_interpolate(x_target, x1, x2, y1, y2):
    """
    Interpolate in logarithmic space for x values
    """
    if x1 <= 0 or x2 <= 0 or x_target <= 0:
        return np.nan

    log_x1 = np.log10(x1)
    log_x2 = np.log10(x2)
    log_x_target = np.log10(x_target)

    # Linear interpolation in log space
    if log_x2 == log_x1:
        return (y1 + y2) / 2

    interpolated = y1 + (y2 - y1) * (log_x_target - log_x1) / (log_x2 - log_x1)
    return interpolated

existing_curve_data = {}
for i, Rm in enumerate(existing_Rm):
    old_name = f'd_{curve_names[i]}'
    new_name = f'd_Rm{Rm}'
    if old_name in combined_data:
        combined_data[new_name] = combined_data[old_name]
        existing_curve_data[Rm] = combined_data[old_name]

for target_idx, Rm_target in enumerate(target_Rm):
    new_curve = np.full(num_points, np.nan)

    for energy_idx, energy in enumerate(x_common):
        if not np.isnan(energy):
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
                            interp_value = log_interpolate(Rm_target, Rm1, Rm2,
                                                         values[Rm1], values[Rm2])
                            if not np.isnan(interp_value):
                                interpolated_values.append(interp_value)

                if len(interpolated_values) > 0:
                    new_curve[energy_idx] = np.mean(interpolated_values)

    combined_data[f'd_{target_names[target_idx]}'] = new_curve

    valid_points = np.sum(~np.isnan(new_curve))

df_combined = pd.DataFrame(combined_data)
df_combined = df_combined.dropna()
df_combined.to_csv('OCR/extracted_data.csv', index=False)
# Create side-by-side visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Left plot: Original image with overlaid extracted curves
ax1.imshow(img_rgb)
ax1.set_title('Original Image with Extracted Curves Overlay')
ax1.set_xlabel('Pixel X')
ax1.set_ylabel('Pixel Y')

colors = ['red', 'green', 'blue']
for i, curve in enumerate(curves_pixel):
    if i < len(curve_names) and len(curve) > 0:
        ax1.plot(curve[:, 0], curve[:, 1], colors[i], linewidth=3,
                alpha=0.8, label=f'{curve_names[i]} curve (Rm={existing_Rm[i]})')

if axis_pts:
    calib_points = [axis_pts['x_min']['pixel'], axis_pts['x_max']['pixel'],
                   axis_pts['y_min']['pixel'], axis_pts['y_max']['pixel']]
    calib_labels = ['X-min', 'X-max', 'Y-min', 'Y-max']
    for point, label in zip(calib_points, calib_labels):
        ax1.plot(point[0], point[1], 'yo', markersize=8, markeredgecolor='black')
        ax1.text(point[0]+10, point[1], label, color='yellow',
                backgroundcolor='black', fontsize=8)

ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

ax2.set_title('Extracted & Interpolated Curves (Rm=4 to 16)')
ax2.set_xlabel('E_beam (keV)')
ax2.set_ylabel('d')
ax2.grid(True, alpha=0.3)

# Plot all curves with consistent Rm labeling
all_Rm_values = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
all_colors = ['red', 'orange', 'lime', 'purple', 'green', 'gold', 'brown', 'magenta', 'pink', 'gray', 'cyan', 'olive', 'blue']
all_linestyles = ['-', '--', '--', '--', '-', '--', '--', '--', '--', '--', '--', '--', '-']
plot_labels = [f'Rm={Rm}' for Rm in all_Rm_values]

# Map Rm values to column names
Rm_to_column = {
    4: 'd_Rm4',
    5: 'd_Rm5',
    6: 'd_Rm6',
    7: 'd_Rm7',
    8: 'd_Rm8',
    9: 'd_Rm9',
    10: 'd_Rm10',
    11: 'd_Rm11',
    12: 'd_Rm12',
    13: 'd_Rm13',
    14: 'd_Rm14',
    15: 'd_Rm15',
    16: 'd_Rm16'
}

for i, Rm in enumerate(all_Rm_values):
    col = Rm_to_column[Rm]
    if col in combined_data and i < len(all_colors):
        # Filter out NaN values for plotting
        valid_mask = ~np.isnan(combined_data[col])
        x_valid = np.array(combined_data['E_beam_keV'])[valid_mask]
        y_valid = np.array(combined_data[col])[valid_mask]

        if len(x_valid) > 0:
            ax2.plot(x_valid, y_valid, color=all_colors[i], linewidth=2,
                    linestyle=all_linestyles[i], label=plot_labels[i])

ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.set_xlim(0, max(combined_data['E_beam_keV']) * 1.05)

try:
    plt.tight_layout()
    plt.savefig('OCR/extracted_curves.png', dpi=150, bbox_inches='tight')
    # plt.show()  # Commented out to prevent segfault
    plt.close()  # Close figure to free memory
    gc.collect()  # Force garbage collection
except Exception as e:
    print(f"Warning: Error in main plotting: {e}")
    plt.close('all')

# Target energy for analysis
target_energy = 120.0
reference_C_45deg = {
    4: 230/177,
    8: 285/177,
    16: 302/177
}

# Find the closest energy index to 120 keV
energy_diff = np.abs(x_common - target_energy)
target_energy_idx = np.argmin(energy_diff)
actual_energy = x_common[target_energy_idx]

# Extract d values at 120 keV for all Rm values
Rm_values_all = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
C_90deg_120kev = {}  # These are C@90° values (extracted from graph)

for Rm in Rm_values_all:
    col_name = f'd_Rm{Rm}'
    if col_name in combined_data:
        d_val = combined_data[col_name][target_energy_idx]
        if not np.isnan(d_val):
            C_90deg_120kev[Rm] = d_val  # d represents C@90°

ratios_45_90 = {}
for Rm in [4, 8, 16]:
    if Rm in C_90deg_120kev and Rm in reference_C_45deg:
        ratio = reference_C_45deg[Rm] / C_90deg_120kev[Rm]  # Fixed: C@45°/C@90°
        ratios_45_90[Rm] = ratio

# Fit the ratios to a + b*log10(Rm) relationship
if len(ratios_45_90) >= 2:
    fit_Rm = np.array(list(ratios_45_90.keys()))
    fit_ratios = np.array(list(ratios_45_90.values()))

    log_Rm = np.log10(fit_Rm)
    coeffs = np.polyfit(log_Rm, fit_ratios, 1)
    b_ratio, a_ratio = coeffs[0], coeffs[1]

    # Calculate R-squared
    predicted_ratios = a_ratio + b_ratio * log_Rm
    ss_res = np.sum((fit_ratios - predicted_ratios) ** 2)
    ss_tot = np.sum((fit_ratios - np.mean(fit_ratios)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Generate fitted ratios for all Rm values
    fitted_ratios_all = {}
    for Rm in Rm_values_all:
        fitted_ratio = a_ratio + b_ratio * np.log10(Rm)
        fitted_ratios_all[Rm] = fitted_ratio
else:
    fitted_ratios_all = {}
    a_ratio, b_ratio, r_squared = np.nan, np.nan, np.nan

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.set_title(f'C@45° vs Rm (Reference Values)')
ax1.set_xlabel('Mirror Ratio (Rm)')
ax1.set_ylabel('C@45°')
ax1.grid(True, alpha=0.3)

ref_Rm = list(reference_C_45deg.keys())
ref_C = list(reference_C_45deg.values())
ax1.plot(ref_Rm, ref_C, 'ro-', linewidth=2, markersize=8, label='Reference C@45°')

for Rm, C_val in reference_C_45deg.items():
    ax1.annotate(f'{C_val:.3f}', (Rm, C_val), textcoords="offset points", xytext=(0,10), ha='center')

ax1.legend()
ax1.set_xlim(2, 18)

ax2.set_title(f'C@45°/C@90° Ratio vs Rm at Eb={actual_energy:.1f} keV')
ax2.set_xlabel('Mirror Ratio (Rm)')
ax2.set_ylabel('C@45°/C@90° Ratio')
ax2.grid(True, alpha=0.3)

if ratios_45_90:
    ratio_Rm = list(ratios_45_90.keys())
    ratio_values = list(ratios_45_90.values())
    ax2.plot(ratio_Rm, ratio_values, 'bo', markersize=8, label='Calculated Points')

    for Rm, ratio in ratios_45_90.items():
        ax2.annotate(f'{ratio:.3f}', (Rm, ratio), textcoords="offset points", xytext=(0,10), ha='center')

if fitted_ratios_all:
    fit_Rm_plot = np.array(sorted(fitted_ratios_all.keys()))
    fit_ratios_plot = np.array([fitted_ratios_all[Rm] for Rm in fit_Rm_plot])
    ax2.plot(fit_Rm_plot, fit_ratios_plot, 'r-', linewidth=2,
            label=f'Fit: {a_ratio:.3f} + {b_ratio:.3f}×log₁₀(Rm)\nR² = {r_squared:.4f}')

    interp_Rm = [Rm for Rm in fit_Rm_plot if Rm not in ratios_45_90]
    interp_ratios = [fitted_ratios_all[Rm] for Rm in interp_Rm]
    if interp_ratios:
        ax2.plot(interp_Rm, interp_ratios, 'rs', markersize=6, label='Fitted Points')

ax2.legend()
ax2.set_xlim(2, 18)

try:
    plt.tight_layout()
    plt.savefig('OCR/analysis_C_vs_Rm_120keV.png', dpi=150, bbox_inches='tight')
    # plt.show()  # Commented out to prevent segfault
    plt.close()  # Close figure to free memory
    gc.collect()  # Force garbage collection
except Exception as e:
    print(f"Warning: Error in analysis plotting: {e}")
    plt.close('all')

# Add fitted ratios to the main extraction data
if fitted_ratios_all:
    ratio_column = np.full(num_points, np.nan)

    for Rm in Rm_values_all:
        if Rm in fitted_ratios_all:
            col_name = f'd_Rm{Rm}'
            if col_name in combined_data:
                ratio_col_name = f'ratio_45_90_Rm{Rm}'
                ratio_values = np.full(num_points, fitted_ratios_all[Rm])
                combined_data[ratio_col_name] = ratio_values

    # Re-save the main CSV with ratio data
    df_combined_with_ratios = pd.DataFrame(combined_data)
    df_combined_with_ratios = df_combined_with_ratios.dropna()
    df_combined_with_ratios.to_csv('OCR/extracted_data.csv', index=False)

# Save analysis data to CSV
analysis_data = {
    'Rm': [],
    'C_90deg_120keV': [],
    'C_45deg_reference': [],
    'C_ratio_45_90_calculated': [],
    'C_ratio_45_90_fitted': []
}

for Rm in sorted(Rm_values_all):
    analysis_data['Rm'].append(Rm)
    analysis_data['C_90deg_120keV'].append(C_90deg_120kev.get(Rm, np.nan))
    analysis_data['C_45deg_reference'].append(reference_C_45deg.get(Rm, np.nan))
    analysis_data['C_ratio_45_90_calculated'].append(ratios_45_90.get(Rm, np.nan))
    analysis_data['C_ratio_45_90_fitted'].append(fitted_ratios_all.get(Rm, np.nan))

df_analysis = pd.DataFrame(analysis_data)
df_analysis.to_csv('OCR/analysis_C_vs_Rm_120keV.csv', index=False)

# ============================================================================
# Additional Analysis: C@90° vs log(Rm) at Eb=100keV
# ============================================================================

# Find the closest energy index to 100 keV
target_energy_100 = 100.0
energy_diff_100 = np.abs(x_common - target_energy_100)
target_energy_idx_100 = np.argmin(energy_diff_100)
actual_energy_100 = x_common[target_energy_idx_100]

# Extract C@90° values at 100 keV for all Rm values
C_90deg_100kev = {}
for Rm in Rm_values_all:
    col_name = f'd_Rm{Rm}'
    if col_name in combined_data:
        d_val = combined_data[col_name][target_energy_idx_100]
        if not np.isnan(d_val):
            C_90deg_100kev[Rm] = d_val

# Create plot of C@90° vs log(Rm) at 100 keV
fig_log, ax_log = plt.subplots(figsize=(10, 7))

if C_90deg_100kev:
    Rm_array = np.array(list(C_90deg_100kev.keys()))
    C_array = np.array(list(C_90deg_100kev.values()))
    log_Rm_array = np.log10(Rm_array)

    # Plot the data
    ax_log.plot(log_Rm_array, C_array, 'bo-', linewidth=2, markersize=8,
                label=f'C@90° at Eb={actual_energy_100:.1f} keV')

    # Add annotations
    for Rm, C_val in C_90deg_100kev.items():
        log_Rm = np.log10(Rm)
        ax_log.annotate(f'Rm={Rm}\nC={C_val:.3f}', (log_Rm, C_val),
                       textcoords="offset points", xytext=(0, 10), ha='center',
                       fontsize=8)

    # Fit a linear relationship: C = a + b*log10(Rm)
    coeffs_log = np.polyfit(log_Rm_array, C_array, 1)
    b_log, a_log = coeffs_log[0], coeffs_log[1]

    # Calculate R-squared
    C_predicted = a_log + b_log * log_Rm_array
    ss_res = np.sum((C_array - C_predicted) ** 2)
    ss_tot = np.sum((C_array - np.mean(C_array)) ** 2)
    r_squared_log = 1 - (ss_res / ss_tot)

    # Plot the fit
    log_Rm_fit = np.linspace(log_Rm_array.min(), log_Rm_array.max(), 100)
    C_fit = a_log + b_log * log_Rm_fit
    ax_log.plot(log_Rm_fit, C_fit, 'r--', linewidth=2,
                label=f'Fit: C = {a_log:.3f} + {b_log:.3f}×log₁₀(Rm)\nR² = {r_squared_log:.4f}')

    ax_log.set_xlabel('log₁₀(Rm)', fontsize=12, fontweight='bold')
    ax_log.set_ylabel('C@90° (confinement parameter)', fontsize=12, fontweight='bold')
    ax_log.set_title(f'C@90° vs log₁₀(Rm) at Eb={actual_energy_100:.1f} keV',
                     fontsize=14, fontweight='bold')
    ax_log.grid(True, alpha=0.3)
    ax_log.legend(fontsize=11)

    try:
        plt.tight_layout()
        plt.savefig('OCR/C_vs_logRm_100keV.png', dpi=150, bbox_inches='tight')
        plt.close()
        gc.collect()
        print(f"\n✓ Saved C@90° vs log(Rm) plot at Eb={actual_energy_100:.1f} keV")
        print(f"  Fit: C = {a_log:.3f} + {b_log:.3f}×log₁₀(Rm)")
        print(f"  R² = {r_squared_log:.4f}")
    except Exception as e:
        print(f"Warning: Error in C vs log(Rm) plotting: {e}")
        plt.close('all')
else:
    print("Warning: No C@90° data available at 100 keV")

# Save C@90° vs Rm data at 100 keV to CSV
analysis_100kev = {
    'Rm': [],
    'log10_Rm': [],
    'C_90deg_100keV': []
}

for Rm in sorted(Rm_values_all):
    if Rm in C_90deg_100kev:
        analysis_100kev['Rm'].append(Rm)
        analysis_100kev['log10_Rm'].append(np.log10(Rm))
        analysis_100kev['C_90deg_100keV'].append(C_90deg_100kev[Rm])

df_100kev = pd.DataFrame(analysis_100kev)
df_100kev.to_csv('OCR/C_vs_logRm_100keV.csv', index=False)
print(f"✓ Saved analysis data to OCR/C_vs_logRm_100keV.csv")

# ============================================================================
# C@45° vs log(Rm) at Eb=100keV
# ============================================================================

# Calculate C@45° = C@90° × ratio_45_90 × 0.1
C_45deg_100kev = {}
if C_90deg_100kev and fitted_ratios_all:
    for Rm in C_90deg_100kev.keys():
        if Rm in fitted_ratios_all:
            C_45deg_100kev[Rm] = C_90deg_100kev[Rm] * fitted_ratios_all[Rm] * 0.1

# Create two-panel plot of C@45° vs log(Rm) and vs Rm at 100 keV
fig_45, (ax_log, ax_lin) = plt.subplots(1, 2, figsize=(16, 7))

if C_45deg_100kev:
    Rm_array_45 = np.array(list(C_45deg_100kev.keys()))
    C_array_45 = np.array(list(C_45deg_100kev.values()))
    log_Rm_array_45 = np.log10(Rm_array_45)

    # ========================================================================
    # LEFT PANEL: C@45° vs log₁₀(Rm)
    # ========================================================================

    # Plot the data
    ax_log.plot(log_Rm_array_45, C_array_45, 'ro-', linewidth=2, markersize=8,
                label=f'C@45° at Eb={actual_energy_100:.1f} keV')

    # Add annotations
    for Rm, C_val in C_45deg_100kev.items():
        log_Rm = np.log10(Rm)
        ax_log.annotate(f'Rm={Rm}\nC={C_val:.4f}', (log_Rm, C_val),
                       textcoords="offset points", xytext=(0, 10), ha='center',
                       fontsize=8)

    # Fit a linear relationship: C = a + b*log10(Rm)
    coeffs_45 = np.polyfit(log_Rm_array_45, C_array_45, 1)
    b_45, a_45 = coeffs_45[0], coeffs_45[1]

    # Calculate R-squared
    C_predicted_45 = a_45 + b_45 * log_Rm_array_45
    ss_res_45 = np.sum((C_array_45 - C_predicted_45) ** 2)
    ss_tot_45 = np.sum((C_array_45 - np.mean(C_array_45)) ** 2)
    r_squared_45 = 1 - (ss_res_45 / ss_tot_45)

    # Plot the fit
    log_Rm_fit_45 = np.linspace(log_Rm_array_45.min(), log_Rm_array_45.max(), 100)
    C_fit_45 = a_45 + b_45 * log_Rm_fit_45
    ax_log.plot(log_Rm_fit_45, C_fit_45, 'g--', linewidth=2,
                label=f'Fit: C = {a_45:.4f} + {b_45:.4f}×log₁₀(Rm)\nR² = {r_squared_45:.4f}')

    ax_log.set_xlabel('log₁₀(Rm)', fontsize=12, fontweight='bold')
    ax_log.set_ylabel('C@45° (loss coefficient) [s]', fontsize=12, fontweight='bold')
    ax_log.set_title(f'C@45° vs log₁₀(Rm) at Eb={actual_energy_100:.1f} keV',
                     fontsize=14, fontweight='bold')
    ax_log.grid(True, alpha=0.3)
    ax_log.legend(fontsize=10)

    # ========================================================================
    # RIGHT PANEL: C@45° vs Rm (linear scale)
    # ========================================================================

    # Plot the data
    ax_lin.plot(Rm_array_45, C_array_45, 'bo-', linewidth=2, markersize=8,
                label=f'C@45° at Eb={actual_energy_100:.1f} keV')

    # Add annotations
    for Rm, C_val in C_45deg_100kev.items():
        ax_lin.annotate(f'Rm={Rm}\nC={C_val:.4f}', (Rm, C_val),
                       textcoords="offset points", xytext=(0, 10), ha='center',
                       fontsize=8)

    # Plot the fit (converted back to linear Rm scale)
    Rm_fit_45 = np.logspace(log_Rm_array_45.min(), log_Rm_array_45.max(), 100)
    C_fit_linear_45 = a_45 + b_45 * np.log10(Rm_fit_45)
    ax_lin.plot(Rm_fit_45, C_fit_linear_45, 'g--', linewidth=2,
                label=f'Fit: C = {a_45:.4f} + {b_45:.4f}×log₁₀(Rm)\nR² = {r_squared_45:.4f}')

    ax_lin.set_xlabel('Mirror Ratio (Rm)', fontsize=12, fontweight='bold')
    ax_lin.set_ylabel('C@45° (loss coefficient) [s]', fontsize=12, fontweight='bold')
    ax_lin.set_title(f'C@45° vs Rm at Eb={actual_energy_100:.1f} keV',
                     fontsize=14, fontweight='bold')
    ax_lin.grid(True, alpha=0.3)
    ax_lin.legend(fontsize=10)

    try:
        plt.tight_layout()
        plt.savefig('OCR/C45_vs_logRm_100keV.png', dpi=150, bbox_inches='tight')
        plt.close()
        gc.collect()
        print(f"\n✓ Saved C@45° vs log(Rm) plot at Eb={actual_energy_100:.1f} keV")
        print(f"  Fit: C@45° = {a_45:.4f} + {b_45:.4f}×log₁₀(Rm)")
        print(f"  R² = {r_squared_45:.4f}")
    except Exception as e:
        print(f"Warning: Error in C@45° vs log(Rm) plotting: {e}")
        plt.close('all')
else:
    print("Warning: No C@45° data available at 100 keV")

# Save C@45° vs Rm data at 100 keV to CSV
analysis_45deg_100kev = {
    'Rm': [],
    'log10_Rm': [],
    'C_90deg_100keV': [],
    'ratio_45_90': [],
    'C_45deg_100keV': []
}

for Rm in sorted(Rm_values_all):
    if Rm in C_45deg_100kev:
        analysis_45deg_100kev['Rm'].append(Rm)
        analysis_45deg_100kev['log10_Rm'].append(np.log10(Rm))
        analysis_45deg_100kev['C_90deg_100keV'].append(C_90deg_100kev.get(Rm, np.nan))
        analysis_45deg_100kev['ratio_45_90'].append(fitted_ratios_all.get(Rm, np.nan))
        analysis_45deg_100kev['C_45deg_100keV'].append(C_45deg_100kev[Rm])

df_45deg = pd.DataFrame(analysis_45deg_100kev)
df_45deg.to_csv('OCR/C45_vs_logRm_100keV.csv', index=False)
print(f"✓ Saved C@45° analysis data to OCR/C45_vs_logRm_100keV.csv")
