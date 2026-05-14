import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

CSV = "artifacts/test_pairs_with_delta.csv"
PAIR_INDEX = 579

df = pd.read_csv(CSV, low_memory=False)
dt = float(df.loc[PAIR_INDEX, "dt_days"])

# fixed positions for schematic (not to scale)
x = np.array([0.25, 0.75])   # prior, current

fig, ax = plt.subplots(figsize=(3.2, 0.9), dpi=300)

# time axis line
ax.plot([0.08, 0.92], [0, 0], lw=2.2, color="#333")

# prior/current circles
for i, xi in enumerate(x):
    ax.add_patch(Circle((xi, 0), 0.06, facecolor="white", edgecolor="#333", lw=3.0))

# Δt badge above
mid = (x[0] + x[1]) / 2
badge = FancyBboxPatch((mid - 0.16, 0.12), 0.32, 0.13,
                       boxstyle="round,pad=0.02,rounding_size=0.06",
                       linewidth=1.0, edgecolor="#999", facecolor="#e8f2ff")
ax.add_patch(badge)
ax.text(mid, 0.185, f"Δt = {int(dt)} d", ha="center", va="center", fontsize=10, color="#222")

# labels
ax.text(x[0], -0.16, "prior", ha="center", va="center", fontsize=9, color="#333")
ax.text(x[1], -0.16, "current", ha="center", va="center", fontsize=9, color="#333")

ax.set_xlim(0, 1)
ax.set_ylim(-0.25, 0.33)
ax.axis("off")

plt.savefig("timeline_pair_only.svg", transparent=True, bbox_inches="tight")
plt.savefig("timeline_pair_only.png", transparent=True, bbox_inches="tight")
print("Saved: timeline_pair_only.svg and timeline_pair_only.png")
