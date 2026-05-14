"""
plot_dtbin_heatmap.py  --  publication-quality heatmap for IJPRAI

Run:
    python plot_dtbin_heatmap.py

Saves:
    images/fig_dtbin_heatmap.pdf  (vector, for LaTeX)
    images/fig_dtbin_heatmap.png  (300 dpi preview)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Data
bins = ["0–1d", "2d", "3–7d", "8–14d", "15–30d", "31–90d", "91–365d", ">365d"]
classes = ["NEW", "RESOLVED", "STABLE_PRESENT"]

f1_new = [0.094, 0.072, 0.130, 0.143, 0.148, 0.062, 0.050, 0.048]
f1_res = [0.125, 0.139, 0.085, 0.089, 0.122, 0.063, 0.059, 0.072]
f1_stp = [0.115, 0.222, 0.204, 0.085, 0.036, 0.057, 0.033, 0.000]

data = np.array([f1_new, f1_res, f1_stp])

# Style (paper-friendly)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.linewidth": 0.6,
})

fig, ax = plt.subplots(figsize=(5.2, 2.6))

# Heatmap
im = ax.imshow(data, aspect="auto", cmap="viridis")

# Axes labels
ax.set_xticks(np.arange(len(bins)))
ax.set_xticklabels(bins)
ax.set_yticks(np.arange(len(classes)))
ax.set_yticklabels(classes)

ax.set_xlabel(r"Follow-up interval ($\Delta t$ bin)")
ax.set_ylabel("Class")

# Annotate values inside cells
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        val = data[i, j]
        ax.text(j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=6,
                color="white" if val > 0.12 else "black")

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.ax.tick_params(labelsize=7)
cbar.set_label("Micro-F1", fontsize=8)

# Clean look
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.xticks(rotation=30, ha='right')
fig.tight_layout()
fig.savefig("images/fig_dtbin_heatmap.pdf", dpi=300)
fig.savefig("images/fig_dtbin_heatmap.png", dpi=300)

print("Saved heatmap figure.")