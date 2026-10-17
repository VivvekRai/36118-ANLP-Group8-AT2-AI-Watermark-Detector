r"""
Draws Figure 4 from the saved hard_test runs.
Run from the project root:  python tests\make_fig4.py
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

STYLE = {
    "deletion": ("Deletion", "o", "#8aa5a8"),
    "paraphrase": ("Paraphrase", "^", "#17414b"),
    "compound": ("Compound", "s", "#cc6a3a"),
}

fig, ax = plt.subplots(figsize=(7.1, 4.0), dpi=200)
n = 0
for path in sorted(glob.glob("tests/results_*_seed*.json")):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for r in data["attacks"]:
        if r["attack"] not in STYLE or r["similarity"] is None:
            continue
        label, marker, colour = STYLE[r["attack"]]
        ax.scatter(
            r["similarity"], r["attacked"]["z"], marker=marker, s=38,
            facecolors=colour if r["eligible"] else "none",
            edgecolors=colour, linewidths=1.2,
        )
        n += 1

ax.axhline(4, color="#333333", linestyle="--", linewidth=1)
ax.text(0.01, 4.12, "detection threshold, z = 4", fontsize=8, color="#333333",
        transform=ax.get_yaxis_transform())
ax.text(0.01, 0.97, "still detected", transform=ax.transAxes, va="top", fontsize=8, color="#5a7f84")
ax.text(0.01, 0.03, "no longer detected", transform=ax.transAxes, va="bottom", fontsize=8, color="#5a7f84")
ax.set_xlim(0.5, 1.02)
ax.set_ylim(-1.7, 6.8)
ax.set_xlabel("MiniLM cosine similarity to the original")
ax.set_ylabel("z-score after attack")
handles = [Line2D([0], [0], marker=m, color="w", markerfacecolor=c, markeredgecolor=c,
                  markersize=7, label=lab) for lab, m, c in STYLE.values()]
ax.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
          bbox_to_anchor=(0.5, 1.12), fontsize=9)
fig.tight_layout()

os.makedirs("docs", exist_ok=True)
out = "docs/fig4_attack_scatter.png"
fig.savefig(out)
print(f"Saved {out} with {n} points (expected 54)")