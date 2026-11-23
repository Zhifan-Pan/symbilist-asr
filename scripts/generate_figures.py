#!/usr/bin/env python3
"""Generate publication-grade IEEE two-column vector PDF figures for the upgraded manuscript.

Outputs:
  - manuscript/figures/mechanism_and_risk.pdf (+ .png for visual inspection)
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

UPG = Path(__file__).resolve().parents[1]
FIG_DIR = UPG / "manuscript" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.0,
    "axes.titlesize": 8.2,
    "axes.labelsize": 7.8,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
    "legend.fontsize": 6.8,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

C_BLUE = "#0072B2"
C_ORANGE = "#D55E00"
C_TEAL = "#009E73"
C_SKY = "#56B4E9"
C_PURPLE = "#CC79A7"
C_YELLOW = "#E69F00"
C_DARK = "#222222"
C_GRAY = "#555555"
C_RED_BG = "#FDECEA"
C_GREEN_BG = "#E8F6F0"
C_BLUE_BG = "#EAF3FB"
C_AMBER_BG = "#FEF5E7"


def build_fig2_mechanism_and_risk():
    """Fig. 2: Three-panel scientific mechanism figure (zero text overlap)."""
    per_lang = json.loads((UPG / "results" / "fleurs_per_lang_summary.json").read_text())
    oracles = json.loads((UPG / "results" / "current_support_oracles.json").read_text())["FLEURS"]
    db = json.loads((UPG / "results" / "stage1_direction_b.json").read_text())["B2_conformal_deferral"]
    sbl = json.loads((UPG / "results" / "stage1_symbilist.json").read_text())["SymBiList_phrase"]["summary"]["FLEURS"]

    fig = plt.figure(figsize=(7.16, 2.55), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.32, 1.08, 0.98], wspace=0.39)

    # Panel (a)
    ax = fig.add_subplot(gs[0, 0])
    langs_sorted = sorted(per_lang.keys(), key=lambda lg: per_lang[lg]["anc_w"] - per_lang[lg]["don_w"], reverse=True)
    y_pos = np.arange(len(langs_sorted))

    a_minus_f_w = [per_lang[lg]["a_w"] - per_lang[lg]["f_w"] for lg in langs_sorted]
    sbl_minus_f_w = [per_lang[lg]["sbl_w"] - per_lang[lg]["f_w"] for lg in langs_sorted]
    sbl_minus_f_c = [per_lang[lg]["sbl_c"] - per_lang[lg]["f_c"] for lg in langs_sorted]

    n_anc_worse = sum(1 for lg in langs_sorted if per_lang[lg]["anc_w"] > per_lang[lg]["don_w"])
    ax.axhspan(-0.5, n_anc_worse - 0.5, color=C_RED_BG, alpha=0.85, zorder=0)
    ax.text(-5.95, 1.45, "Anchor\nweaker\n(Anc>Don)", fontsize=5.9, color=C_ORANGE,
            fontweight="bold", ha="left", va="center")

    ax.axvline(0.0, color=C_DARK, lw=0.8, ls="--", zorder=1)
    for i, (aw, sw) in enumerate(zip(a_minus_f_w, sbl_minus_f_w)):
        ax.plot([aw, sw], [i, i], color="#999999", lw=0.85, zorder=2)

    ax.scatter(a_minus_f_w, y_pos, color=C_YELLOW, edgecolor=C_DARK, linewidth=0.45,
               s=20, marker="o", label="Agree A ($\\Delta$WER)", zorder=3)
    ax.scatter(sbl_minus_f_w, y_pos, color=C_BLUE, edgecolor="white", linewidth=0.4,
               s=28, marker="D", label="SymBiList ($\\Delta$WER)", zorder=4)
    ax.scatter(sbl_minus_f_c, y_pos, color=C_TEAL, edgecolor="white", linewidth=0.35,
               s=18, marker="s", label="SymBiList ($\\Delta$CER)", zorder=3)

    def _fmt_diff(val):
        r = int(round(val))
        return "0" if r == 0 else f"{r:+d}"

    short_labels = [f"{lg.split('_')[0]} ({_fmt_diff(per_lang[lg]['anc_w']-per_lang[lg]['don_w'])})"
                    for lg in langs_sorted]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_labels, fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlim(-6.2, 1.15)
    ax.set_xlabel("Method $-$ Fixed $L$ (pp; $<0$ better)", fontsize=7.0)
    ax.set_ylabel("Lang (Anc$-$Don WER)", fontsize=7.0, labelpad=2)
    ax.set_title("(a) Per-Language vs. Fixed $L$", fontsize=7.7, fontweight="bold", pad=5)
    ax.grid(axis="x", linestyle=":", alpha=0.5, lw=0.6)
    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.94,
              edgecolor="#CCCCCC", borderpad=0.25, handletextpad=0.22, labelspacing=0.18, fontsize=5.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel (b)
    ax2 = fig.add_subplot(gs[0, 1])
    pts = [
        ("Whole (B)", oracles["B"]["macro"][0], 22.722, C_GRAY, "o", (-42, -13)),
        ("Slots", oracles["slots"]["macro"][0], 22.525, C_PURPLE, "^", (5, 2)),
        ("Random (Q)", oracles["Q"]["macro"][0], 22.369, "#888888", "v", (5, 3)),
        ("Fixed (F)", oracles["F"]["macro"][0], 22.264, C_ORANGE, "s", (5, -9)),
        ("Agree A", oracles["A"]["macro"][0], 21.948, C_YELLOW, "o", (6, -2)),
        ("SymBiPhrase", oracles["A"]["macro"][0], 21.738, C_SKY, "P", (6, 2)),
        ("SymBiList", oracles["A"]["macro"][0], sbl["macro_WER_mean"], C_BLUE, "*", (7, -6)),
    ]

    xs_line = np.linspace(16.0, 20.0, 50)
    for gap in [3.5, 4.5, 5.5, 6.5]:
        ax2.plot(xs_line, xs_line + gap, ls=":", lw=0.6, color="#BBBBBB", zorder=1)
        lx = 19.65 if (19.65 + gap) < 22.92 else (22.90 - gap)
        ly = lx + gap
        if 16.15 < lx < 19.8 and 21.50 < ly < 22.94:
            ax2.text(lx, ly - 0.03, f"+{gap:.1f}", fontsize=5.7, color="#888888",
                     rotation=28, ha="right", va="top")

    traj_x = [oracles["B"]["macro"][0], oracles["A"]["macro"][0],
              oracles["Q"]["macro"][0], oracles["F"]["macro"][0], oracles["slots"]["macro"][0]]
    traj_y = [22.722, 21.948, 22.369, 22.264, 22.525]
    ax2.plot(traj_x, traj_y, color="#AAAAAA", lw=0.9, ls="--", zorder=2)

    ax2.annotate("", xy=(oracles["A"]["macro"][0], sbl["macro_WER_mean"] + 0.045),
                 xytext=(oracles["A"]["macro"][0], 21.948 - 0.04),
                 arrowprops=dict(arrowstyle="-|>", color=C_BLUE, lw=1.15), zorder=4)

    for name, ox, ay, col, mk, (dx, dy) in pts:
        sz = 82 if mk == "*" else (36 if name == "SymBiPhrase" else 30)
        ax2.scatter([ox], [ay], color=col, edgecolor=C_DARK if mk != "*" else "white",
                    linewidth=0.55, s=sz, marker=mk, zorder=5)
        ax2.annotate(name, (ox, ay), xytext=(dx, dy), textcoords="offset points",
                     fontsize=6.2, fontweight="bold" if name == "SymBiList" else "normal",
                     color=C_BLUE if name == "SymBiList" else C_DARK)

    ax2.set_xlim(16.05, 19.95)
    ax2.set_ylim(21.48, 22.95)
    ax2.set_xlabel("Space Oracle WER (%)", fontsize=7.0)
    ax2.set_ylabel("Actual Macro WER (%)", fontsize=7.0, labelpad=2)
    ax2.set_title("(b) Support vs. Selection Gap", fontsize=7.7, fontweight="bold", pad=5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Panel (c)
    ax3 = fig.add_subplot(gs[0, 2])
    alphas = [0.02, 0.04, 0.06, 0.08, 0.12]
    fl_harm = [db[f"CRC_alpha_{a:.2f}"]["FLEURS"]["empirical_harm_rate"] * 100 for a in alphas]
    kb_harm = [db[f"CRC_alpha_{a:.2f}"]["Kathbath"]["empirical_harm_rate"] * 100 for a in alphas]
    fl_rep = [db[f"CRC_alpha_{a:.2f}"]["FLEURS"]["empirical_repair_rate"] * 100 for a in alphas]
    a_pct = [a * 100 for a in alphas]

    ax3.fill_between([0, 12.8], [0, 12.8], [0, 0], color=C_GREEN_BG, alpha=0.65, zorder=0)
    ax3.plot([0, 12.8], [0, 12.8], color=C_DARK, ls="--", lw=0.85, label="Ceiling $R \\leq \\alpha$", zorder=1)
    ax3.plot(a_pct, fl_harm, color=C_BLUE, marker="o", ms=4.0, lw=1.25, label="FLEURS harm", zorder=3)
    ax3.plot(a_pct, kb_harm, color=C_TEAL, marker="s", ms=3.8, lw=1.25, label="Kathbath harm", zorder=3)
    ax3.plot(a_pct, fl_rep, color=C_ORANGE, marker="^", ms=3.8, lw=1.0, ls="-.", label="FLEURS repair", zorder=2)

    ax3.annotate("5.2$\\times$ R:H\n(8.1% vs 1.6%)", xy=(2.2, 8.12), xytext=(5.2, 11.4),
                 fontsize=5.75, color=C_ORANGE, fontweight="bold",
                 arrowprops=dict(arrowstyle="-|>", color=C_ORANGE, lw=0.75, connectionstyle="arc3,rad=-0.12"))
    ax3.text(12.2, 0.8, "Safe Zone\n($\\hat{R} \\leq \\alpha$)", ha="right", va="bottom",
             fontsize=6.0, color=C_TEAL, fontweight="bold")

    ax3.set_xlim(1.0, 12.8)
    ax3.set_ylim(0.0, 22.0)
    ax3.set_xlabel("Target Harm Ceiling $\\alpha$ (%)", fontsize=7.0)
    ax3.set_ylabel("Empirical Rate vs. Whole (%)", fontsize=7.0, labelpad=2)
    ax3.set_title("(c) Conformal Risk Control", fontsize=7.7, fontweight="bold", pad=5)
    ax3.grid(linestyle=":", alpha=0.5, lw=0.6)
    ax3.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.94,
               edgecolor="#CCCCCC", borderpad=0.22, handletextpad=0.22, labelspacing=0.16, fontsize=5.7)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    out_pdf = FIG_DIR / "mechanism_and_risk.pdf"
    out_png = FIG_DIR / "mechanism_and_risk.png"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.03, dpi=250)
    plt.close(fig)
    print(f"Saved {out_pdf} and {out_png}")


if __name__ == "__main__":
    build_fig2_mechanism_and_risk()
