#!/usr/bin/env python3
"""Generate crisp full-width 1x3 IEEE Figure 2 (mechanism_and_risk.pdf).

Zero-overlap publication layout (7.25 in x 2.28 in):
  (a) Per-Language Gains vs. Fixed-L (FLEURS-12)
  (b) Official FLEURS-12 Test (9,671) Accuracy-Latency Pareto & CPU->GPU Cascade
  (c) Conformal Risk Control (Official FLEURS-12 Test 3.00x R:H, Kathbath Test, CV26)
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UPG = Path(__file__).resolve().parents[1]
FIG_DIR = UPG / "manuscript" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.5,
    "axes.titlesize": 7.8,
    "axes.labelsize": 7.3,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.0,
    "axes.linewidth": 0.72,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

C_NAVY = "#1B365D"
C_BLUE = "#1D4ED8"
C_TEAL = "#0F766E"
C_AMBER = "#D97706"
C_RED = "#C53030"
C_PURPLE = "#6D28D9"
C_GRAY = "#64748B"
C_DARK = "#1E293B"
C_RED_BG = "#FEF2F2"
C_GREEN_BG = "#ECFDF5"


def build_fig2_full_width():
    rigor = json.loads((UPG / "results" / "asr_rigor_upgrade.json").read_text())
    fp_fl = json.loads((UPG / "results" / "fp_fleurs12_official_test.json").read_text())["systems"]
    crc_sbl = json.loads((UPG / "results" / "p1_symbilist_crc.json").read_text())
    casc = json.loads((UPG / "results" / "cscwd_cascade_and_q17_ablation.json").read_text())
    audit = json.loads((UPG / "results" / "chatgpt_audit_checks.json").read_text())

    fig = plt.figure(figsize=(7.25, 2.04), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.06, 1.22, 1.00], wspace=0.40)

    # =========================================================================
    # Panel (a): Per-Language Gains vs. Bidir. Whole (12/12) & Fixed-L on FLEURS-12 Test
    # =========================================================================
    ax = fig.add_subplot(gs[0, 0])
    pl_om = fp_fl["OmniASR_1B_v2"]["per_lang"]
    pl_mm = fp_fl["MMS_1B_all"]["per_lang"]
    pl_wh = fp_fl["Whole_sym_Listwise"]["per_lang"]
    pl_fx = fp_fl["Fixed_L3_HGB24"]["per_lang"]
    pl_sy = fp_fl["SymBiList_frozen"]["per_lang"]

    langs_sorted = sorted(pl_sy.keys(), key=lambda lg: pl_om[lg]["WER"] - pl_mm[lg]["WER"], reverse=True)
    y_pos = np.arange(len(langs_sorted))

    sbl_vs_wh_w = [pl_sy[lg]["WER"] - pl_wh[lg]["WER"] for lg in langs_sorted]
    sbl_vs_fx_w = [pl_sy[lg]["WER"] - pl_fx[lg]["WER"] for lg in langs_sorted]
    sbl_vs_fx_c = [pl_sy[lg]["CER"] - pl_fx[lg]["CER"] for lg in langs_sorted]

    n_anc_worse = sum(1 for lg in langs_sorted if pl_om[lg]["WER"] > pl_mm[lg]["WER"])
    ax.axhspan(-0.5, n_anc_worse - 0.5, color=C_RED_BG, alpha=0.92, zorder=0)
    ax.text(-2.38, 2.15, "Omni weaker", fontsize=5.4, color=C_RED,
            fontweight="bold", ha="left", va="center")

    ax.axvline(0.0, color=C_DARK, lw=0.75, ls="--", zorder=1)
    for i, dw in enumerate(sbl_vs_wh_w):
        ax.plot([0.0, dw], [i, i], color="#BFDBFE", lw=1.15, zorder=2)

    # Plot vs. Bidir. Whole (12/12 wins!), vs. Fixed-L WER (clipping am=-6.34 with arrow), vs. Fixed-L CER (12/12)
    fx_in_x, fx_in_y = [], []
    for i, val in enumerate(sbl_vs_fx_w):
        if val < -2.35:
            ax.annotate(f"{val:.2f}", xy=(-2.42, i), xytext=(-1.72, i),
                        fontsize=5.3, color="#B45309", fontweight="bold", va="center", ha="left",
                        arrowprops=dict(arrowstyle="-|>", color=C_AMBER, lw=0.95), zorder=5)
        else:
            fx_in_x.append(val)
            fx_in_y.append(i)

    # Same for CER if any outlier < -2.35 (am CER is -3.57!)
    fxc_in_x, fxc_in_y = [], []
    for i, val in enumerate(sbl_vs_fx_c):
        if val < -2.35:
            ax.annotate("", xy=(-2.42, i + 0.25), xytext=(-2.05, i + 0.25),
                        arrowprops=dict(arrowstyle="-|>", color=C_TEAL, lw=0.85), zorder=5)
        else:
            fxc_in_x.append(val)
            fxc_in_y.append(i)

    ax.scatter(sbl_vs_wh_w, y_pos, color=C_BLUE, edgecolor="white", linewidth=0.35,
               s=25, marker="D", label="vs. Whole (WER)", zorder=4)
    ax.scatter(fx_in_x, fx_in_y, color=C_AMBER, edgecolor=C_DARK, linewidth=0.35,
               s=17, marker="o", label="vs. Fix-$L$ (WER)", zorder=3)
    ax.scatter(fxc_in_x, fxc_in_y, color=C_TEAL, edgecolor="white", linewidth=0.35,
               s=16, marker="s", label="vs. Fix-$L$ (CER)", zorder=3)

    def _fmt_diff(val):
        r = int(round(val))
        return "0" if r == 0 else f"{r:+d}"

    short_labels = [f"{lg.split('_')[0]} ({_fmt_diff(pl_om[lg]['WER']-pl_mm[lg]['WER'])})"
                    for lg in langs_sorted]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_labels, fontsize=6.2)
    ax.invert_yaxis()
    ax.set_xlim(-2.68, 0.72)
    ax.set_xticks([-2.0, -1.0, 0.0, 0.5])
    ax.set_xticklabels(["-2.0", "-1.0", "0.0", "0.5"], fontsize=6.4)
    ax.set_xlabel("SymBiList $-$ Baseline (pp; $<0$ better)", fontsize=6.8, labelpad=2)
    ax.set_ylabel("Lang (Omni$-$MMS)", fontsize=6.9, labelpad=2)
    ax.set_title("(a) Per-Language Test Gains", fontsize=7.7, fontweight="bold", pad=5)
    ax.grid(axis="x", linestyle=":", alpha=0.5, lw=0.55)
    ax.legend(loc="lower left", bbox_to_anchor=(0.005, 0.005), frameon=True, facecolor="white", framealpha=0.95,
              edgecolor="#CBD5E1", borderpad=0.14, handletextpad=0.14, labelspacing=0.12, fontsize=4.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # =========================================================================
    # Panel (b): Official FLEURS-12 Test (9,671) Accuracy-Latency Pareto Frontier
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    lr = casc["cascade_pareto"]["learned_router"]
    rb = casc["cascade_pareto"]["random_baseline"]
    keys = ["0pct", "10pct", "20pct", "25pct", "30pct", "40pct", "50pct", "75pct", "100pct"]
    lr_x = [lr[k]["avg_latency_ms"] for k in keys]
    lr_y = [lr[k]["macro_WER"] for k in keys]
    rb_x = [rb[k]["avg_latency_ms"] for k in keys]
    rb_y = [rb[k]["macro_WER"] for k in keys]

    # Reference baselines on Official FLEURS-12 Test (9,671)
    ax2.scatter([2.0], [26.094], color=C_GRAY, marker="^", s=28, edgecolor=C_DARK, lw=0.4, zorder=4)
    ax2.annotate("Exact MBR", (2.0, 26.094), xytext=(5, -2), textcoords="offset points", fontsize=6.0, color=C_GRAY)

    ax2.scatter([2.5], [22.522], color=C_GRAY, marker="o", s=26, edgecolor=C_DARK, lw=0.4, zorder=4)
    ax2.annotate("Whole (sym.)", (2.5, 22.522), xytext=(5, 5), textcoords="offset points", fontsize=5.9, color=C_DARK)

    ax2.scatter([65.0], [22.211], color=C_AMBER, marker="s", s=30, edgecolor=C_DARK, lw=0.45, zorder=4)
    ax2.annotate("Whole+1.7B+ng", (65.0, 22.211), xytext=(6, -2), textcoords="offset points",
                 fontsize=5.6, color="#92400E", ha="left")

    # Qwen3-8B GER & Anchor-Guarded Qwen3-8B
    ax2.scatter([5300.0], [25.907], color=C_RED, marker="X", s=44, zorder=5)
    ax2.annotate("Qwen3-8B GER\n(8.6% OOL)", (5300.0, 25.907), xytext=(-6, 0), textcoords="offset points",
                 fontsize=5.3, color=C_RED, fontweight="bold", ha="right", va="center")
    ax2.scatter([5300.0], [23.683], color=C_TEAL, marker="o", s=32, facecolors="white", edgecolors=C_TEAL, lw=1.25, zorder=5)
    ax2.annotate("", xy=(5300.0, 23.90), xytext=(5300.0, 25.64),
                 arrowprops=dict(arrowstyle="-|>", color=C_TEAL, lw=1.0), zorder=4)
    ax2.annotate("+Anchor Guard\n(0% OOL, -2.2pp)", (5300.0, 23.683), xytext=(-6, -3), textcoords="offset points",
                 fontsize=5.2, color=C_TEAL, ha="right", va="center")

    # Cascade Pareto Curves: Learned Uncertainty Router vs. Random Routing
    ax2.plot(rb_x, rb_y, color="#94A3B8", ls="--", lw=1.05, label="Random", zorder=3)
    ax2.plot(lr_x, lr_y, color=C_PURPLE, lw=1.65, marker="o", ms=2.8, label="CPU$\\to$GPU", zorder=5)
    ax2.scatter([3.0], [21.419], color=C_NAVY, marker="*", s=74, edgecolor="white", lw=0.45, zorder=7, label="SymBiList")
    ax2.scatter([129.0], [21.016], color=C_BLUE, marker="D", s=34, edgecolor="white", lw=0.45, zorder=7, label="+Qwen3-1.7B")

    ax2.annotate("3.0 ms CPU\n(21.42%)", (3.0, 21.419), xytext=(2, 5.5), textcoords="offset points",
                 fontsize=5.3, fontweight="bold", color=C_NAVY, ha="left")
    ax2.annotate("53% GPU (~69ms est.)\n88% gain", (lr["50pct"]["avg_latency_ms"], lr["50pct"]["macro_WER"]),
                 xytext=(-2, -21), textcoords="offset points", fontsize=5.0, color=C_PURPLE, ha="center",
                 arrowprops=dict(arrowstyle="-|>", color=C_PURPLE, lw=0.7))
    ax2.annotate("21.02%", (129.0, 21.016), xytext=(6, -2), textcoords="offset points",
                 fontsize=5.8, fontweight="bold", color=C_BLUE)

    ax2.set_xscale("log")
    ax2.set_xlim(1.3, 9500)
    ax2.set_ylim(19.30, 26.85)
    ax2.set_xticks([3, 20, 129, 1000, 5000])
    ax2.set_xticklabels(["3ms", "20ms", "129ms", "1s", "5s"], fontsize=6.5)
    ax2.set_xlabel("Post-Decoding Latency (log)", fontsize=7.0, labelpad=2)
    ax2.set_ylabel("FLEURS-12 Test WER (%)", fontsize=7.0, labelpad=2)
    ax2.set_title("(b) Test WER vs. Latency", fontsize=7.7, fontweight="bold", pad=5)
    ax2.grid(linestyle=":", alpha=0.5, lw=0.55)
    # Compact legend cleanly in open middle-left area -> HUGE white gap from Qwen3-8B labels in PDF!
    ax2.legend(loc="upper left", bbox_to_anchor=(0.01, 0.86), frameon=True, facecolor="white",
               framealpha=0.95, edgecolor="#CBD5E1", borderpad=0.18, handlelength=1.35,
               handletextpad=0.25, labelspacing=0.14, fontsize=5.1)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # =========================================================================
    # Panel (c): Conformal Risk Control (Official FLEURS-12 Test + KB Test + CV26)
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    alphas_kb = [0.02, 0.04, 0.06, 0.08, 0.12]
    fk = crc_sbl["FLEURS_Kathbath"]
    cv_crc = crc_sbl["CV26_Z3"]["all8"]
    kt_harm = [fk["Kathbath_testunk"][f"CRC_a{a:.2f}"]["empirical_harm_rate"] * 100 for a in alphas_kb]
    cv_harm = [cv_crc[f"CRC_a{a:.2f}"]["empirical_harm_rate"] * 100 for a in alphas_kb]
    a_kb_pct = [a * 100 for a in alphas_kb]

    # Canonical frozen 3-seed Official FLEURS-12 Test CRC sweep (100% consistent with Table III!)
    crc_te = rigor["crc_canonical_3seed_fleurs12_test"]
    alphas_te = [0.02, 0.04, 0.06, 0.08, 0.12]
    fl_te_harm = [crc_te[f"a{a:.2f}"]["empirical_harm_rate"] * 100 for a in alphas_te]
    fl_te_rep = [crc_te[f"a{a:.2f}"]["empirical_repair_rate"] * 100 for a in alphas_te]
    a_te_pct = [a * 100 for a in alphas_te]
    rh_lbl = f"{crc_te['a0.06']['repair_to_harm']:.2f}$\\times$ R:H"
    rep_at_6 = crc_te["a0.06"]["empirical_repair_rate"] * 100

    # Strict independent split-conformal exchangeable Dev check (zero refit, 5-fold mean)
    sp_dev = audit["check2_independent_split_crc"]["protocolB_3of5_fit_1of5_cal_exchangeable_dev_mean5"]
    sp_a = [4.0, 6.0, 8.0]
    sp_h = [sp_dev[f"{a/100:.2f}"]["harm_mean"] * 100 for a in sp_a]

    ax3.fill_between([0, 12.8], [0, 12.8], [0, 0], color=C_GREEN_BG, alpha=0.72, zorder=0)
    ax3.plot([0, 12.8], [0, 12.8], color=C_DARK, ls="--", lw=0.8, label="Target $\\alpha$", zorder=1)
    ax3.plot(a_te_pct, fl_te_harm, color=C_BLUE, marker="o", ms=3.1, lw=1.25, label="FL-Test", zorder=4)
    ax3.plot(sp_a, sp_h, color=C_NAVY, marker="v", ms=3.2, lw=1.1, ls="-", label="Split-Dev", zorder=5)
    ax3.plot(a_kb_pct, kt_harm, color=C_TEAL, marker="s", ms=2.9, lw=1.15, label="KB-Test", zorder=3)
    ax3.plot(a_kb_pct, cv_harm, color=C_PURPLE, marker="d", ms=2.8, lw=1.05, ls=":", label="CV26", zorder=3)
    ax3.plot(a_te_pct, fl_te_rep, color=C_AMBER, marker="^", ms=3.1, lw=1.15, ls="-.", label="FL-Repair", zorder=2)

    ax3.annotate(rh_lbl, xy=(6.0, rep_at_6), xytext=(8.2, 13.8),
                 fontsize=6.0, color="#B45309", fontweight="bold",
                 arrowprops=dict(arrowstyle="-|>", color=C_AMBER, lw=0.85, connectionstyle="arc3,rad=-0.12"))
    ax3.text(12.4, 0.55, "Below target $\\alpha$", ha="right", va="bottom",
             fontsize=5.8, color=C_TEAL, fontweight="bold")

    ax3.set_xlim(1.2, 12.8)
    ax3.set_ylim(0.0, 34.2)
    ax3.set_xticks([2, 4, 6, 8, 10, 12])
    ax3.set_yticks([0, 5, 10, 15, 20, 25])
    ax3.set_xlabel("Target Risk Ceiling $\\alpha$ (%)", fontsize=7.0, labelpad=2)
    ax3.set_ylabel("Rate vs. Whole (%)", fontsize=7.0, labelpad=2)
    ax3.set_title("(c) CRC Risk Control", fontsize=7.7, fontweight="bold", pad=5)
    ax3.grid(linestyle=":", alpha=0.5, lw=0.55)
    # Compact 2-column horizontal legend cleanly seated above the 22.6% curve -> ZERO overlap in PDF!
    ax3.legend(loc="upper center", bbox_to_anchor=(0.50, 1.01), frameon=True, facecolor="white", framealpha=0.95,
               edgecolor="#CBD5E1", borderpad=0.16, handlelength=1.25, handletextpad=0.22, columnspacing=0.85, labelspacing=0.12, fontsize=4.8, ncol=2)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    out_pdf = FIG_DIR / "mechanism_and_risk.pdf"
    out_png = FIG_DIR / "mechanism_and_risk.png"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.025, dpi=260)
    plt.close(fig)
    print(f"Saved zero-overlap 1x3 {out_pdf} and {out_png}")


if __name__ == "__main__":
    build_fig2_full_width()
