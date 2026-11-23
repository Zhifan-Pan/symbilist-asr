"""Stage 1 Step 4: Direction B — Model Family / Context / Pairwise / Listwise Rankers + Conformal Risk-Controlled Deferral."""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agree_phrase_asr.data_bridge import load_frozen_dev_bundle, SEEDS
from agree_phrase_asr.align import build_projection_bundle, make_projector, flatten_cells, splice_cells
from agree_phrase_asr.partition import agreement_partition, soft_agreement_partition, validate_partition
from agree_phrase_asr.features import build_phrase_graph
from agree_phrase_asr.eval import edit_counts, evaluate_predictions, paired_cluster_bootstrap_diff
from agree_phrase_asr.models import (
    DEFAULT_HGB_PARAMS,
    run_grouped_oof,
    prepare_fit_targets,
    predict_block_costs,
    choose_from_block_costs,
)


# ---------------- B1-a: Pairwise Difference GBDT Ranker (PairRanker style) ----------------

def build_pairwise_arrays(train_uids, graphs, meta):
    per_lang = defaultdict(list)
    active = defaultdict(int)
    for u in train_uids:
        r = meta[u]
        rw = r["reference"].split()
        per_lang[r["language"]].append([len(rw), len("".join(rw))])
        if graphs[u][1]:
            active[r["language"]] += 1
    scale = {l: np.mean(vals, axis=0) for l, vals in per_lang.items()}

    X_pairs, Y_pairs, W_pairs = [], [], []
    for u in train_uids:
        cells, blocks = graphs[u]
        if not blocks:
            continue
        ref = meta[u]["reference"]
        lang = meta[u]["language"]
        sc = scale[lang]
        u_pairs = []
        u_y = []
        for bl in blocks:
            X_all = bl["X_all"]
            m = len(X_all)
            errs = np.array([[edit_counts(ref, h)[0]["E"], edit_counts(ref, h)[1]["E"]] for h in bl["full"]], dtype=np.float64) / sc
            for a in range(m):
                for b in range(m):
                    if a == b:
                        continue
                    u_pairs.append(np.concatenate([X_all[a] - X_all[b], X_all[a], X_all[b]]))
                    u_y.append(errs[a] - errs[b])
        if u_pairs:
            X_pairs.append(np.asarray(u_pairs, dtype=np.float64))
            Y_pairs.append(np.asarray(u_y, dtype=np.float64))
            W_pairs.append(np.full(len(u_pairs), 1.0 / (active[lang] * len(u_pairs)), dtype=np.float64))
    X = np.vstack(X_pairs)
    Y = np.vstack(Y_pairs)
    W = np.concatenate(W_pairs)
    W *= len(W) / W.sum()
    return X, Y, W


def predict_pairwise_block_scores(graphs, uids, models):
    pair_rows = []
    meta_slices = []
    for u in uids:
        for bi, bl in enumerate(graphs[u][1]):
            X_all = bl["X_all"]
            m = len(X_all)
            start = len(pair_rows)
            for a in range(m):
                for b in range(m):
                    if a == b:
                        continue
                    pair_rows.append(np.concatenate([X_all[a] - X_all[b], X_all[a], X_all[b]]))
            meta_slices.append((u, bi, m, start, len(pair_rows)))
    if not pair_rows:
        return {u: [] for u in uids}
    stacked = np.asarray(pair_rows, dtype=np.float64)
    preds = np.stack([mod.predict(stacked) for mod in models], axis=1)  # (P, 2)
    out = {u: [] for u in uids}
    for u, bi, m, s, e in meta_slices:
        z = np.zeros((m, 2), dtype=np.float64)
        idx = s
        for a in range(m):
            for b in range(m):
                if a == b:
                    continue
                z[a] += preds[idx]
                idx += 1
        z /= float(m)  # mean expected edit disadvantage vs all options in block
        out[u].append(z)
    return out


def run_pairwise_oof(graphs, bundles, meta, folds_by_seed, fleurs_uids, kathbath_uids, alpha_grid=(1.0, 0.5)):
    all_preds = {}
    for seed in SEEDS:
        fs = folds_by_seed[seed]
        preds = {}
        for f in range(6):
            tr_ids = [u for u in fleurs_uids if f == 5 or fs[u]["outer"] != f]
            ev_ids = kathbath_uids if f == 5 else [u for u in fleurs_uids if fs[u]["outer"] == f]

            def inner_id(u):
                return fs[u]["full"] if f == 5 else fs[u]["inner"][str(f)]

            inner_preds = {a: {} for a in alpha_grid}
            for j in range(3):
                in_tr = [u for u in tr_ids if inner_id(u) != j]
                in_va = [u for u in tr_ids if inner_id(u) == j]
                Xp, Yp, Wp = build_pairwise_arrays(in_tr, graphs, meta)
                mods = [HistGradientBoostingRegressor(**DEFAULT_HGB_PARAMS).fit(Xp, Yp[:, i], sample_weight=Wp) for i in (0, 1)]
                sc = predict_pairwise_block_scores(graphs, in_va, mods)
                for a in alpha_grid:
                    for u in in_va:
                        preds_h, _ = choose_from_block_costs(graphs[u], sc[u], bundles[u], alpha_c=a)
                        inner_preds[a][u] = preds_h
            inner_ev = {a: (evaluate_predictions(inner_preds[a], meta, sorted(inner_preds[a]))["macro_WER"],
                            evaluate_predictions(inner_preds[a], meta, sorted(inner_preds[a]))["macro_CER"]) for a in alpha_grid}
            winner = min(alpha_grid, key=lambda a: (inner_ev[a][0], inner_ev[a][1], -a))

            Xp, Yp, Wp = build_pairwise_arrays(tr_ids, graphs, meta)
            mods = [HistGradientBoostingRegressor(**DEFAULT_HGB_PARAMS).fit(Xp, Yp[:, i], sample_weight=Wp) for i in (0, 1)]
            sc = predict_pairwise_block_scores(graphs, ev_ids, mods)
            for u in ev_ids:
                preds[u], _ = choose_from_block_costs(graphs[u], sc[u], bundles[u], alpha_c=winner)
        all_preds[seed] = preds
    return _summarize(all_preds, meta, fleurs_uids, kathbath_uids), all_preds


# ---------------- B1-b: PyTorch Listwise Expected-Edit Neural Ranker ----------------

class ListwiseBlockNet(nn.Module):
    def __init__(self, d_in: int, hidden: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def fit_listwise_ranker(train_uids, graphs, meta, alpha_c: float = 0.5, seed: int = 1701, epochs: int = 35):
    torch.manual_seed(seed)
    np.random.seed(seed)
    per_lang = defaultdict(list)
    active = defaultdict(int)
    for u in train_uids:
        r = meta[u]
        rw = r["reference"].split()
        per_lang[r["language"]].append([len(rw), len("".join(rw))])
        if graphs[u][1]:
            active[r["language"]] += 1
    scale = {l: np.mean(vals, axis=0) for l, vals in per_lang.items()}

    all_rows = []
    block_records = []
    for u in train_uids:
        cells, blocks = graphs[u]
        if not blocks:
            continue
        ref = meta[u]["reference"]
        lang = meta[u]["language"]
        sc = scale[lang]
        wt = 1.0 / (active[lang] * len(blocks))
        for bl in blocks:
            X_all = bl["X_all"]
            errs = np.array([[edit_counts(ref, h)[0]["E"], edit_counts(ref, h)[1]["E"]] for h in bl["full"]], dtype=np.float64) / sc
            comb_err = errs[:, 0] + alpha_c * errs[:, 1]
            comb_err = comb_err - comb_err.min()  # block regret >= 0
            all_rows.append(X_all)
            block_records.append((X_all, comb_err, wt))

    scaler = StandardScaler().fit(np.vstack(all_rows))
    # Normalize weights
    w_sum = sum(r[2] for r in block_records)
    # Group blocks into padded tensor (B, max_m, D) for fast vectorized training on CPU!
    max_m = max(len(r[0]) for r in block_records)
    B = len(block_records)
    D = block_records[0][0].shape[1]
    X_pad = np.zeros((B, max_m, D), dtype=np.float32)
    Y_pad = np.zeros((B, max_m), dtype=np.float32)
    M_pad = np.zeros((B, max_m), dtype=np.bool_)
    W_arr = np.zeros(B, dtype=np.float32)
    for i, (xa, ye, wt) in enumerate(block_records):
        m = len(xa)
        X_pad[i, :m] = scaler.transform(xa).astype(np.float32)
        Y_pad[i, :m] = ye.astype(np.float32)
        M_pad[i, :m] = True
        W_arr[i] = float(wt * B / w_sum)

    Xt = torch.from_numpy(X_pad)
    Yt = torch.from_numpy(Y_pad)
    Mt = torch.from_numpy(M_pad)
    Wt = torch.from_numpy(W_arr)

    model = ListwiseBlockNet(D, hidden=48)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=2e-2)
    batch_size = 512
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(B)
        for s in range(0, B, batch_size):
            idx = perm[s : s + batch_size]
            xb, yb, mb, wb = Xt[idx], Yt[idx], Mt[idx], Wt[idx]
            logits = model(xb)
            logits = logits.masked_fill(~mb, -1e9)
            probs = torch.softmax(logits, dim=-1)
            # Differentiable expected block regret loss!
            exp_regret = (probs * yb).sum(dim=-1)
            loss = (exp_regret * wb).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model, scaler


def predict_listwise_ranker(graphs, uids, model, scaler):
    out = {}
    with torch.no_grad():
        for u in uids:
            cells, blocks = graphs[u]
            acts = []
            for bl in blocks:
                x = torch.from_numpy(scaler.transform(bl["X_all"]).astype(np.float32))
                scores = model(x).numpy()
                j = int(np.argmax(scores))
                acts.append((bl["lo"], bl["hi"], bl["alts"][j]))
            out[u] = splice_cells(cells, acts)
    return out


def run_listwise_oof(graphs, meta, folds_by_seed, fleurs_uids, kathbath_uids, alpha_grid=(1.0, 0.5)):
    all_preds = {}
    for seed in SEEDS:
        fs = folds_by_seed[seed]
        preds = {}
        for f in range(6):
            tr_ids = [u for u in fleurs_uids if f == 5 or fs[u]["outer"] != f]
            ev_ids = kathbath_uids if f == 5 else [u for u in fleurs_uids if fs[u]["outer"] == f]

            def inner_id(u):
                return fs[u]["full"] if f == 5 else fs[u]["inner"][str(f)]

            inner_preds = {a: {} for a in alpha_grid}
            for j in range(3):
                in_tr = [u for u in tr_ids if inner_id(u) != j]
                in_va = [u for u in tr_ids if inner_id(u) == j]
                for a in alpha_grid:
                    mod, sc = fit_listwise_ranker(in_tr, graphs, meta, alpha_c=a, seed=1701 + j)
                    inner_preds[a].update(predict_listwise_ranker(graphs, in_va, mod, sc))
            inner_ev = {a: (evaluate_predictions(inner_preds[a], meta, sorted(inner_preds[a]))["macro_WER"],
                            evaluate_predictions(inner_preds[a], meta, sorted(inner_preds[a]))["macro_CER"]) for a in alpha_grid}
            winner = min(alpha_grid, key=lambda a: (inner_ev[a][0], inner_ev[a][1], -a))

            mod, sc = fit_listwise_ranker(tr_ids, graphs, meta, alpha_c=winner, seed=1701)
            preds.update(predict_listwise_ranker(graphs, ev_ids, mod, sc))
        all_preds[seed] = preds
    return _summarize(all_preds, meta, fleurs_uids, kathbath_uids), all_preds


# ---------------- B2: Conformal Risk-Controlled Deferral (CRC-Deferral) ----------------

def run_conformal_deferral(res_base, graphs, bundles, meta, folds_by_seed, fleurs_uids, kathbath_uids):
    """Calibrate utterance-level recombination margin threshold lambda on inner folds to control harm rate <= alpha_risk."""
    risk_targets = [0.02, 0.04, 0.06, 0.08, 0.12]
    # Also include WER-optimal calibrated threshold on inner folds
    lambda_grid = np.concatenate([[0.0], np.geomspace(1e-4, 0.5, 40)])

    def utterance_phrase_vs_whole(u, graph, z_list, bundle, alpha_c):
        cells, blocks = graph
        pool = bundle["texts"]
        proj = bundle["proj"]
        whole_reg = np.zeros(len(pool), dtype=np.float64)
        phrase_reg_sum = 0.0
        acts = []
        for bl, z in zip(blocks, z_list):
            reg = z - z.min(axis=0)
            r = np.maximum(reg[:, 0], alpha_c * reg[:, 1])
            j = int(np.argmin(r))
            acts.append((bl["lo"], bl["hi"], bl["alts"][j]))
            phrase_reg_sum += float(r[j])
            for pi, h in enumerate(pool):
                whole_reg[pi] += float(r[bl["alts"].index(flatten_cells(proj[h][bl["lo"]:bl["hi"]]))])
        h_phrase = splice_cells(cells, acts)
        best_w = int(np.argmin(whole_reg))
        h_whole = pool[best_w]
        margin = float(whole_reg[best_w] - phrase_reg_sum)  # >= 0: regret reduction from recombining
        return h_phrase, h_whole, margin

    # Evaluate CRC-Deferral using out-of-fold models from res_base (strictly nested calibration on inner OOF!)
    fit_targets = prepare_fit_targets(graphs, meta, fleurs_uids)
    out_summary = {}
    preds_by_mode = {f"CRC_alpha_{a:.2f}": {s: {} for s in SEEDS} for a in risk_targets}
    preds_by_mode["Margin_inner_WER"] = {s: {} for s in SEEDS}
    harm_stats = {k: {"FLEURS": [], "Kathbath": []} for k in preds_by_mode}

    for seed in SEEDS:
        fs = folds_by_seed[seed]
        for f in range(6):
            tr_ids = [u for u in fleurs_uids if f == 5 or fs[u]["outer"] != f]
            ev_ids = kathbath_uids if f == 5 else [u for u in fleurs_uids if fs[u]["outer"] == f]
            alpha_c = res_base["models"][seed][f]["alpha"]

            def inner_id(u):
                return fs[u]["full"] if f == 5 else fs[u]["inner"][str(f)]

            # Gather inner OOF margins & true harm indicators for calibration
            cal_records = []
            for j in range(3):
                in_tr = [u for u in tr_ids if inner_id(u) != j]
                in_va = [u for u in tr_ids if inner_id(u) == j]
                from agree_phrase_asr.models import fit_dual_regressor
                in_mod = fit_dual_regressor("hgb", in_tr, fit_targets, meta, seed=1701)
                in_sc = predict_block_costs(graphs, in_va, in_mod)
                for u in in_va:
                    hp, hw, margin = utterance_phrase_vs_whole(u, graphs[u], in_sc[u], bundles[u], alpha_c)
                    ref = meta[u]["reference"]
                    ep = edit_counts(ref, hp)[0]["E"]
                    ew = edit_counts(ref, hw)[0]["E"]
                    cal_records.append((u, hp, hw, margin, ep, ew))

            # Calibrate Conformal Risk Control threshold per risk target alpha
            n_cal = len(cal_records)
            chosen_lambdas = {}
            for ar in risk_targets:
                # Find smallest lambda where upper-bounded harm rate (sum(harm_i(lam)) + 1)/(n_cal + 1) <= ar
                best_lam = float(lambda_grid[-1])
                for lam in lambda_grid:
                    harms = sum(1 for _, hp, hw, m, ep, ew in cal_records if (hp != hw and m >= lam and ep > ew))
                    if (harms + 1.0) / (n_cal + 1.0) <= ar:
                        best_lam = float(lam)
                        break
                chosen_lambdas[f"CRC_alpha_{ar:.2f}"] = best_lam

            # Inner-WER optimal lambda
            best_wer_lam, best_wer_val = 0.0, 999.0
            for lam in lambda_grid:
                pr_cal = {u: (hp if m >= lam else hw) for u, hp, hw, m, _, _ in cal_records}
                val = evaluate_predictions(pr_cal, meta, [r[0] for r in cal_records])["macro_WER"]
                if val < best_wer_val - 1e-9:
                    best_wer_val = val
                    best_wer_lam = float(lam)
            chosen_lambdas["Margin_inner_WER"] = best_wer_lam

            # Apply calibrated lambdas to ev_ids using outer model
            out_mod = res_base["models"][seed][f]["models"]
            ev_sc = predict_block_costs(graphs, ev_ids, out_mod)
            for u in ev_ids:
                hp, hw, margin = utterance_phrase_vs_whole(u, graphs[u], ev_sc[u], bundles[u], alpha_c)
                for mode_name, lam in chosen_lambdas.items():
                    preds_by_mode[mode_name][seed][u] = hp if margin >= lam else hw

    for mode_name, pdict in preds_by_mode.items():
        summ = _summarize(pdict, meta, fleurs_uids, kathbath_uids)
        # Also compute empirical harm rate & repair rate vs whole_fixed on seed 17
        for ds_name, ds_uids in [("FLEURS", fleurs_uids), ("Kathbath", kathbath_uids)]:
            repairs = harms = outside = 0
            for u in ds_uids:
                hp = pdict[17][u]
                hw = res_base["whole_preds"][17][u]
                if hp not in bundles[u]["texts"]:
                    outside += 1
                if hp != hw:
                    ref = meta[u]["reference"]
                    ep = edit_counts(ref, hp)[0]["E"]
                    ew = edit_counts(ref, hw)[0]["E"]
                    if ep < ew:
                        repairs += 1
                    elif ep > ew:
                        harms += 1
            summ[ds_name]["empirical_harm_rate"] = harms / len(ds_uids)
            summ[ds_name]["empirical_repair_rate"] = repairs / len(ds_uids)
            summ[ds_name]["outside_count"] = outside
            summ[ds_name]["repairs_vs_B"] = repairs
            summ[ds_name]["harms_vs_B"] = harms
        out_summary[mode_name] = summ
    return out_summary, preds_by_mode


def _summarize(all_preds, meta, fleurs_uids, kathbath_uids):
    summary = {}
    for ds_name, ds_uids in [("FLEURS", fleurs_uids), ("Kathbath", kathbath_uids)]:
        wers, cers = [], []
        for seed in SEEDS:
            ev = evaluate_predictions(all_preds[seed], meta, ds_uids)
            wers.append(ev["macro_WER"])
            cers.append(ev["macro_CER"])
        summary[ds_name] = {
            "macro_WER_mean": float(np.mean(wers)),
            "macro_WER_sd": float(np.std(wers, ddof=1)),
            "macro_CER_mean": float(np.mean(cers)),
            "macro_CER_sd": float(np.std(cers, ddof=1)),
        }
    return summary


def main():
    t0 = time.monotonic()
    bundle = load_frozen_dev_bundle()
    all_data = bundle["all_data"]
    meta = bundle["meta"]
    folds = bundle["folds_by_seed"]
    fleurs_uids = bundle["fleurs_uids"]
    kathbath_uids = bundle["kathbath_uids"]
    all_uids = fleurs_uids + kathbath_uids
    char_lm = bundle["char_lm_fn"]
    word_lm = bundle["word_lm"]
    norm_fn = bundle["normalize_fn"]
    proj_fn = make_projector(bundle["align_fn"])

    pb_omni = {u: build_projection_bundle(all_data[u][0], all_data[u][1], anchor_idx=0, normalize_fn=norm_fn, project_fn=proj_fn) for u in all_uids}
    g_A = {u: build_phrase_graph(pb_omni[u], agreement_partition(pb_omni[u]), char_lm, word_lm) for u in all_uids}
    g_A_ctx = {u: build_phrase_graph(pb_omni[u], agreement_partition(pb_omni[u]), char_lm, word_lm, add_block_context_feats=True) for u in all_uids}

    print(f"[{time.monotonic()-t0:.1f}s] Built graphs for Direction B.", flush=True)

    res_A = run_grouped_oof(g_A, pb_omni, meta, folds, fleurs_uids, kathbath_uids, need_whole=True)
    results = {"baseline_A": res_A["summary"], "B1_rankers": {}, "B2_conformal_deferral": {}, "bootstrap_vs_A": {}}

    # B1: Model family + Intra-block Context GBDT
    print("\n=== B1: Model Family & Block-Context Rankers ===", flush=True)
    for label, g_use, kind in [
        ("Ridge_24d", g_A, "ridge"),
        ("MLP_24d", g_A, "mlp"),
        ("HGB_deep_24d", g_A, "hgb_deep"),
        ("HGB_Context_32d", g_A_ctx, "hgb"),
        ("HGB_deep_Context_32d", g_A_ctx, "hgb_deep"),
        ("MLP_Context_32d", g_A_ctx, "mlp"),
    ]:
        r = run_grouped_oof(g_use, pb_omni, meta, folds, fleurs_uids, kathbath_uids, model_kind=kind)
        results["B1_rankers"][label] = r["summary"]
        sf, sk = r["summary"]["FLEURS"], r["summary"]["Kathbath"]
        print(f"  {label:24s} | FLEURS {sf['macro_WER_mean']:6.3f}±{sf['macro_WER_sd']:.3f} / {sf['macro_CER_mean']:5.3f} | Kathbath {sk['macro_WER_mean']:6.3f} / {sk['macro_CER_mean']:5.3f}", flush=True)
        if label in ("HGB_Context_32d", "HGB_deep_Context_32d"):
            ci_f = paired_cluster_bootstrap_diff(r["preds"][17], res_A["preds"][17], meta, fleurs_uids, n_draws=10000)
            ci_k = paired_cluster_bootstrap_diff(r["preds"][17], res_A["preds"][17], meta, kathbath_uids, n_draws=10000, stratify_by_language=True)
            results["bootstrap_vs_A"][label] = {"FLEURS": ci_f, "Kathbath": ci_k}

    # B1: Pairwise GBDT Ranker (PairRanker style)
    print("\n=== B1: Pairwise Difference GBDT Ranker (PairRanker) ===", flush=True)
    pw_sum, pw_preds = run_pairwise_oof(g_A_ctx, pb_omni, meta, folds, fleurs_uids, kathbath_uids)
    results["B1_rankers"]["Pairwise_HGB_Context"] = pw_sum
    sf, sk = pw_sum["FLEURS"], pw_sum["Kathbath"]
    print(f"  {'Pairwise_HGB_Context':24s} | FLEURS {sf['macro_WER_mean']:6.3f}±{sf['macro_WER_sd']:.3f} / {sf['macro_CER_mean']:5.3f} | Kathbath {sk['macro_WER_mean']:6.3f} / {sk['macro_CER_mean']:5.3f}", flush=True)
    results["bootstrap_vs_A"]["Pairwise_HGB_Context"] = {
        "FLEURS": paired_cluster_bootstrap_diff(pw_preds[17], res_A["preds"][17], meta, fleurs_uids, n_draws=10000),
        "Kathbath": paired_cluster_bootstrap_diff(pw_preds[17], res_A["preds"][17], meta, kathbath_uids, n_draws=10000, stratify_by_language=True),
    }

    # B1: Listwise Expected-Edit Neural Ranker (PyTorch)
    print("\n=== B1: Listwise Expected-Edit Neural Ranker (PyTorch) ===", flush=True)
    lw_sum, lw_preds = run_listwise_oof(g_A_ctx, meta, folds, fleurs_uids, kathbath_uids)
    results["B1_rankers"]["Listwise_Neural_Context"] = lw_sum
    sf, sk = lw_sum["FLEURS"], lw_sum["Kathbath"]
    print(f"  {'Listwise_Neural_Context':24s} | FLEURS {sf['macro_WER_mean']:6.3f}±{sf['macro_WER_sd']:.3f} / {sf['macro_CER_mean']:5.3f} | Kathbath {sk['macro_WER_mean']:6.3f} / {sk['macro_CER_mean']:5.3f}", flush=True)
    results["bootstrap_vs_A"]["Listwise_Neural_Context"] = {
        "FLEURS": paired_cluster_bootstrap_diff(lw_preds[17], res_A["preds"][17], meta, fleurs_uids, n_draws=10000),
        "Kathbath": paired_cluster_bootstrap_diff(lw_preds[17], res_A["preds"][17], meta, kathbath_uids, n_draws=10000, stratify_by_language=True),
    }

    # B2: Conformal Risk-Controlled Deferral
    print("\n=== B2: Conformal Risk-Controlled Deferral ===", flush=True)
    crc_sum, crc_preds = run_conformal_deferral(res_A, g_A, pb_omni, meta, folds, fleurs_uids, kathbath_uids)
    results["B2_conformal_deferral"] = crc_sum
    for k, v in crc_sum.items():
        sf, sk = v["FLEURS"], v["Kathbath"]
        print(f"  {k:20s} | FLEURS {sf['macro_WER_mean']:6.3f}/{sf['macro_CER_mean']:5.3f} (harm={sf['empirical_harm_rate']:.3f}, R/H={sf['repairs_vs_B']}/{sf['harms_vs_B']}) | Kathb {sk['macro_WER_mean']:6.3f}/{sk['macro_CER_mean']:5.3f} (harm={sk['empirical_harm_rate']:.3f}, R/H={sk['repairs_vs_B']}/{sk['harms_vs_B']})", flush=True)

    out_path = ROOT / "results" / "stage1_direction_b.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n[{time.monotonic()-t0:.1f}s] Saved Direction B results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
