import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

OUT_DIR = Path("kg_output")
OUT_DIR.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 7))
ax.set_xlim(0, 16)
ax.set_ylim(0, 7)
ax.axis("off")

# ── 颜色 ──────────────────────────────────────────────────────
C_INPUT   = "#5B8DB8"
C_IMG     = "#9B59B6"
C_KG      = "#20B2AA"
C_RULE    = "#3CB371"
C_FUSION  = "#E67E22"
C_OUTPUT  = "#C0392B"
C_ARROW   = "#444444"
C_BOX_BG  = "#F8F9FA"

BOX_W = 1.6
BOX_H = 0.75

def draw_box(ax, x, y, title, subtitle, color, text_color="white"):
    box = FancyBboxPatch((x - BOX_W/2, y - BOX_H/2), BOX_W, BOX_H,
                          boxstyle="round,pad=0.08",
                          facecolor=color, edgecolor="white",
                          linewidth=1.5, zorder=3)
    ax.add_patch(box)
    ax.text(x, y + 0.12, title, ha="center", va="center",
            fontsize=9, fontweight="bold", color=text_color, zorder=4)
    if subtitle:
        ax.text(x, y - 0.18, subtitle, ha="center", va="center",
                fontsize=7.5, color=text_color, alpha=0.92, zorder=4)

def arrow(ax, x1, y1, x2, y2, color=C_ARROW, label="", label_side="top"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.5, mutation_scale=14),
                zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        dy = 0.18 if label_side == "top" else -0.18
        ax.text(mx, my+dy, label, ha="center", fontsize=7.5,
                color=color, style="italic")

# ══════════════════════════════════════════════════════════════
# 上分支：图像流
# ══════════════════════════════════════════════════════════════
# Fundus Image (input)
draw_box(ax, 1.3, 5.5, "Fundus Image", "178–408 px RGB", C_INPUT)

# ResNet50
draw_box(ax, 3.3, 5.5, "ResNet50", "2048-dim", C_IMG)
arrow(ax, 1.3+BOX_W/2, 5.5, 3.3-BOX_W/2, 5.5, label="encode")

# PCA
draw_box(ax, 5.3, 5.5, "PCA", "64-dim", C_IMG)
arrow(ax, 3.3+BOX_W/2, 5.5, 5.3-BOX_W/2, 5.5, label="reduce")

# Classifier
draw_box(ax, 7.5, 5.5, "GB Classifier", "g: R⁶⁴→[0,1]", C_IMG)
arrow(ax, 5.3+BOX_W/2, 5.5, 7.5-BOX_W/2, 5.5, label="train")

# p_img label
ax.text(7.5, 4.95, r"$p_{\mathrm{img}}^{(i)}$",
        ha="center", fontsize=10, color=C_IMG, fontweight="bold")

# ══════════════════════════════════════════════════════════════
# 下分支：KG流
# ══════════════════════════════════════════════════════════════
# JSON Annotation (input)
draw_box(ax, 1.3, 2.0, "JSON Annotation", "11 biomarker fields", C_INPUT)

# KG Construction
draw_box(ax, 3.3, 2.0, "GlaKG", "3,460 nodes\n15,472 edges", C_KG)
arrow(ax, 1.3+BOX_W/2, 2.0, 3.3-BOX_W/2, 2.0, label="parse")

# Clinical Rules
draw_box(ax, 3.3, 0.8, "Clinical Rules", "K=11 rules\nstrong/moderate/weak", C_RULE)
arrow(ax, 3.3, 2.0-BOX_H/2, 3.3, 0.8+BOX_H/2, label="encode", label_side="top")

# Chain Score
draw_box(ax, 5.3, 2.0, "Chain Score", r"CS = 3n_s+2n_m+n_w", C_KG)
arrow(ax, 3.3+BOX_W/2, 2.0, 5.3-BOX_W/2, 2.0, label="activate")
arrow(ax, 3.3, 0.8+BOX_H/2, 5.3-BOX_W/2, 2.0-0.1,
      color=C_RULE, label="")

# Normalization
draw_box(ax, 7.5, 2.0, "Normalize", r"$s_{\mathrm{KG}}^{(i)}\in[0,1]$", C_KG)
arrow(ax, 5.3+BOX_W/2, 2.0, 7.5-BOX_W/2, 2.0, label="min-max")

# s_KG label
ax.text(7.5, 1.45, r"$s_{\mathrm{KG}}^{(i)}$",
        ha="center", fontsize=10, color=C_KG, fontweight="bold")

# ══════════════════════════════════════════════════════════════
# 融合层
# ══════════════════════════════════════════════════════════════
draw_box(ax, 10.2, 3.75, "KG-Fusion",
         r"$p_{\mathrm{final}}=(1-\alpha)p_{\mathrm{img}}+\alpha s_{\mathrm{KG}}$",
         C_FUSION)

# 上分支 → Fusion
arrow(ax, 7.5+BOX_W/2, 5.5, 10.2-BOX_W/2, 3.75+0.25,
      color=C_IMG, label=r"$p_{\mathrm{img}}$", label_side="top")
# 下分支 → Fusion
arrow(ax, 7.5+BOX_W/2, 2.0, 10.2-BOX_W/2, 3.75-0.25,
      color=C_KG, label=r"$s_{\mathrm{KG}}$", label_side="top")

# α 标注
ax.text(9.3, 3.75, r"$\alpha^*=0.5$", ha="center", fontsize=9,
        color=C_FUSION, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                  edgecolor=C_FUSION, alpha=0.9))

# ══════════════════════════════════════════════════════════════
# 输出
# ══════════════════════════════════════════════════════════════
# Binary Classification
draw_box(ax, 13.0, 5.2, "Binary Diagnosis", "glaucoma / normal\nF1=0.9953", C_OUTPUT)
arrow(ax, 10.2+BOX_W/2, 3.75+0.2, 13.0-BOX_W/2, 5.2,
      color=C_OUTPUT, label="Task 1", label_side="top")

# Risk Stratification
draw_box(ax, 13.0, 2.3, "Risk Stratification",
         "high/moderate/healthy\nAcc=0.930", C_OUTPUT)
arrow(ax, 10.2+BOX_W/2, 3.75-0.2, 13.0-BOX_W/2, 2.3,
      color=C_OUTPUT, label="Task 2", label_side="top")

# Reasoning Chain
draw_box(ax, 13.0, 3.75, "Reasoning Chain",
         "activated rules\n+ evidence strength", "#6A5ACD")
arrow(ax, 10.2+BOX_W/2, 3.75, 13.0-BOX_W/2, 3.75,
      color="#6A5ACD", label="XAI")

# ══════════════════════════════════════════════════════════════
# 分支标签
# ══════════════════════════════════════════════════════════════
ax.text(0.3, 5.5, "Image\nBranch", ha="center", fontsize=9,
        color=C_IMG, fontweight="bold", style="italic")
ax.text(0.3, 2.0, "KG\nBranch", ha="center", fontsize=9,
        color=C_KG, fontweight="bold", style="italic")

# ══════════════════════════════════════════════════════════════
# 图例
# ══════════════════════════════════════════════════════════════
legend_items = [
    mpatches.Patch(color=C_INPUT,  label="Input data"),
    mpatches.Patch(color=C_IMG,    label="Image processing"),
    mpatches.Patch(color=C_KG,     label="KG construction"),
    mpatches.Patch(color=C_RULE,   label="Clinical rules"),
    mpatches.Patch(color=C_FUSION, label="KG-fusion"),
    mpatches.Patch(color=C_OUTPUT, label="Output"),
]
ax.legend(handles=legend_items, loc="lower center",
          ncol=6, fontsize=8, framealpha=0.9,
          bbox_to_anchor=(0.5, -0.02))

ax.set_title("GlaKG End-to-End Framework", fontsize=13,
             fontweight="bold", pad=12)

plt.tight_layout()
plt.savefig(OUT_DIR / "fig_framework.png", dpi=200, bbox_inches="tight")
plt.savefig(OUT_DIR / "fig_framework.pdf", bbox_inches="tight")
plt.show()
print("Saved: kg_output/fig_framework.png / .pdf")