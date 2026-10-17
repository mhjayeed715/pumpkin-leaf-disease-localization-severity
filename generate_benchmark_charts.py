import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('results/figures', exist_ok=True)

# -------------------------------------------------------------
# FIGURE E: 200-IMAGE FIELD CLUTTER AUDIT DISTRIBUTION
# -------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

categories = [
    'Natural Daylight\n(lighting_quality)',
    'Focal Leaf\nObvious',
    'Multi-Leaf Clutter\n(>1 leaf present)',
    'Diffuse\nChlorosis',
    'Lamina\nNecrosis',
    'Weeds &\nTearing'
]
percentages = [99.5, 99.0, 84.0, 56.5, 46.5, 27.0]
counts = [199, 198, 168, 113, 93, 54]
colors = ['#0284C7', '#0284C7', '#DC2626', '#EA580C', '#D97706', '#64748B']

bars = ax.bar(categories, percentages, color=colors, width=0.55, edgecolor='#1E293B', linewidth=1.2, zorder=3)

# Add decision threshold line at 25%
ax.axhline(25.0, color='#DC2626', linestyle='--', linewidth=1.8, label='Pre-registered Mandate Threshold (25.0%)', zorder=4)

# Data labels
for bar, pct, count in zip(bars, percentages, counts):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.8, f'{pct:.1f}%\n({count}/200)',
            ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#0F172A')

ax.set_ylim(0, 115)
ax.set_ylabel('Prevalence Rate in Field Audit (%)', fontsize=11, fontweight='bold', labelpad=8)
ax.set_title('Empirical Clutter & Foliar Defect Distribution (N = 200 In-Field Images)\nStandardized Field Clutter Protocol — 84.0% Canopy Clutter Mandates Target-Leaf Isolation',
             fontsize=12, fontweight='bold', pad=14, color='#0F172A')
ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, fontsize=10)
ax.grid(axis='y', linestyle=':', alpha=0.6, zorder=0)

plt.tight_layout()
fig_e_path = 'results/figures/figure_e_audit_distribution.png'
plt.savefig(fig_e_path, dpi=300)
plt.close()
print(f'Saved Figure E to {fig_e_path}')

# -------------------------------------------------------------
# FIGURE F: 5-FOLD CV SEGMENTATION & SEVERITY BENCHMARK
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300, gridspec_kw={'width_ratios': [1.3, 1.0]})

models = ['DualTaskUNet\n(Ours)', 'Two-Stage U-Net\n(Separate)', 'LinkNet\n(ResNet-18)', 'DeepLabV3+\n(MobileNetV2)']
x = np.arange(len(models))
width = 0.28

leaf_dice = [0.9584, 0.9540, 0.9434, 0.9342]
leaf_err = [0.0100, 0.0028, 0.0121, 0.0103]

lesion_dice = [0.8456, 0.8019, 0.7621, 0.7246]
lesion_err = [0.0153, 0.0079, 0.0216, 0.0353]

# Left plot: Segmentation Dice
rects1 = ax1.bar(x - width/2, leaf_dice, width, yerr=leaf_err, capsize=4, label='Leaf Lamina Dice',
                 color='#2563EB', edgecolor='#1E293B', linewidth=1.1, zorder=3)
rects2 = ax1.bar(x + width/2, lesion_dice, width, yerr=lesion_err, capsize=4, label='Lesion Dice (Diseased)',
                 color='#EA580C', edgecolor='#1E293B', linewidth=1.1, zorder=3)

# Reference line for initialization vs final reference masks
ax1.axhline(0.542, color='#64748B', linestyle=':', linewidth=1.6,
            label='Initialization vs Final Reference Masks (0.542)', zorder=4)

# Overlay parameter count as discrete markers with secondary axis
ax1_twin = ax1.twinx()
params = [31.04, 49.54, 11.54, 3.21]
ax1_twin.scatter(x, params, color='#0F172A', s=80, marker='D', zorder=5, label='Parameters (M)')
for i, txt in enumerate(params):
    ax1_twin.annotate(f'{txt:.1f}M', (x[i], params[i] + 2.0), ha='center', fontsize=8.5, fontweight='bold', color='#0F172A')

ax1_twin.set_ylabel('Model Parameters (Millions)', fontsize=10, fontweight='bold', labelpad=8)
ax1_twin.set_ylim(0, 65)
ax1_twin.grid(False)

ax1.set_ylabel('5-Fold Cross-Validation Dice Score', fontsize=10, fontweight='bold', labelpad=8)
ax1.set_ylim(0.45, 1.05)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=9.5, fontweight='bold')
ax1.set_title('Segmentation Accuracy & Model Complexity\n(Markers = Parameter Count; No Continuous Trend Implied)',
              fontsize=11, fontweight='bold', pad=12)
ax1.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9, fontsize=8.5)
ax1_twin.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=8.5)
ax1.grid(axis='y', linestyle=':', alpha=0.6, zorder=0)

# Right plot: Severity Estimation Error & Healthy Trade-Off
dis_mae = [4.06, 3.21, 4.17, 4.15]
healthy_mae = [16.59, 5.25, 7.39, 4.70]
w2 = 0.32
x2 = np.arange(len(models))

rects3 = ax2.bar(x2 - w2/2, dis_mae, w2, label='Diseased-Leaf Severity MAE (%)',
                 color='#059669', edgecolor='#1E293B', linewidth=1.1, zorder=3)
rects4 = ax2.bar(x2 + w2/2, healthy_mae, w2, label='Healthy-Leaf False Alarm MAE (%)',
                 color='#DC2626', edgecolor='#1E293B', linewidth=1.1, zorder=3)

for bar in rects3:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f'{yval:.2f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

for bar in rects4:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f'{yval:.2f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax2.set_ylabel('Severity Mean Absolute Error (MAE %)', fontsize=10, fontweight='bold', labelpad=8)
ax2.set_ylim(0, 22)
ax2.set_xticks(x2)
ax2.set_xticklabels(models, fontsize=9.5, fontweight='bold')
ax2.set_title('Severity Estimation & Healthy Leaf Trade-Off\n(DualTaskUNet Suffers Healthy False Alarms -> Mandates Gating)',
              fontsize=11, fontweight='bold', pad=12)
ax2.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=8.5)
ax2.grid(axis='y', linestyle=':', alpha=0.6, zorder=0)

plt.tight_layout()
fig_f_path = 'results/figures/figure_f_benchmark_tradeoff.png'
plt.savefig(fig_f_path, dpi=300)
plt.close()
print(f'Saved Figure F to {fig_f_path}')
