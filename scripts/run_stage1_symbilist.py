"""Stage 1 Final Synthesis: SymBiList (Symmetric Bidirectional Agreement Phrase Recombination + Differentiable Listwise Expected-Edit Neural Ranker)."""
from __future__ import annotations

import json
import pickle
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
from agree_phrase_asr.align import build_projection_bundle, make_projector, flatten_cells, splice_cells, lev_dist
from agree_phrase_asr.partition import agreement_partition, validate_partition
from agree_phrase_asr.features import build_phrase_graph
from agree_phrase_asr.eval import edit_counts, evaluate_predictions, paired_cluster_bootstrap_diff


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


def fit_listwise_ensemble(train_uids, graphs, meta, alpha_c: float = 0.5, seed: int = 1701, n_ens: int = 3, epochs: int = 40):
    per_lang = defaultdict(list)
    active = defaultdict(int)
    for u in train_uids:
        r = meta[u]
        rw = r["reference"].split()
        per_lang[r["language"]].append([len(rw), len("".join(rw))])
        if graphs[u][1]:
            active[r["language"]] += 1
    scale = {l: np.mean(vals, axis=0) for l, vals in per_lang.items()}

    all_rows, block_records = [], []
    for u in train_uids:
        _, blocks = graphs[u]
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
            comb_err = comb_err - comb_err.min()
            all_rows.append(X_all)
            block_records.append((X_all, comb_err, wt))

    scaler = StandardScaler().fit(np.vstack(all_rows))
    w_sum = sum(r[2] for r in block_records)
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

    Xt, Yt, Mt, Wt = torch.from_numpy(X_pad), torch.from_numpy(Y_pad), torch.from_numpy(M_pad), torch.from_numpy(W_arr)

    models = []
    for ei in range(n_ens):
        torch.manual_seed(seed + ei * 17)
        np.random.seed(seed + ei * 17)
        model = ListwiseBlockNet(D, hidden=48)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=2e-2)
        model.train()
        for _ in range(epochs):
            perm = torch.randperm(B)
            for s in range(0, B, 512):
                idx = perm[s : s + 512]
                xb, yb, mb, wb = Xt[idx], Yt[idx], Mt[idx], Wt[idx]
                logits = model(xb).masked_fill(~mb, -1e9)
                probs = torch.softmax(logits, dim=-1)
                loss = ((probs * yb).sum(dim=-1) * wb).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        model.eval()
        models.append(model)
    return models, scaler


def predict_listwise_ensemble(graphs, bundles, uids, models, scaler, need_whole: bool = False):
    out = {}
    whole_out = {}
    with torch.no_grad():
        for u in uids:
            cells, blocks = graphs[u]
            acts = []
            pool = bundles[u]["texts"]
            proj = bundles[u]["proj"]
            whole_score = np.zeros(len(pool), dtype=np.float64)
            for bl in blocks:
                x = torch.from_numpy(scaler.transform(bl["X_all"]).astype(np.float32))
                # Average log-softmax probability (or normalized logits) across ensemble members
                probs = np.mean([torch.softmax(m(x), dim=-1).numpy() for m in models], axis=0)
                j = int(np.argmax(probs))
                acts.append((bl["lo"], bl["hi"], bl["alts"][j]))
                if need_whole:
                    logp = np.log(np.clip(probs, 1e-12, 1.0))
                    for pi, h in enumerate(pool):
                        whole_score[pi] += float(logp[bl["alts"].index(flatten_cells(proj[h][bl["lo"]:bl["hi"]]))])
            out[u] = splice_cells(cells, acts)
            if need_whole:
                whole_out[u] = pool[int(np.argmax(whole_score))]
    return out, whole_out


def symmetric_sentence_feats(h: str, b0: dict, char_lm, word_lm) -> np.ndarray:
    hw = h.split()
    hc = "".join(hw)
    c4_raw = float(char_lm(h))
    c4_per = c4_raw / (len(h) + 1)
    w2 = np.asarray(word_lm.scores(h), dtype=np.float64)
    g1w, g2w = b0["g1"].split(), b0["g2"].split()
    g1c, g2c = "".join(g1w), "".join(g2w)
    A_sym = max(1.0, 0.5 * (len(g1w) + len(g2w)))
    C_sym = max(1.0, 0.5 * (len(g1c) + len(g2c)))
    r1w = sum(wt * lev_dist(hw, k.split()) for k, wt in b0["p1"].items()) / A_sym
    r2w = sum(wt * lev_dist(hw, k.split()) for k, wt in b0["p2"].items()) / A_sym
    r1c = sum(wt * lev_dist(hc, "".join(k.split())) for k, wt in b0["p1"].items()) / C_sym
    r2c = sum(wt * lev_dist(hc, "".join(k.split())) for k, wt in b0["p2"].items()) / C_sym
    return np.array([
        c4_per, c4_raw / (C_sym + 1.0),
        w2[0], w2[1], w2[2],
        r1w, r2w, 0.5 * (r1w + r2w),
        r1c, r2c, 0.5 * (r1c + r2c),
        float(len(hw)) / A_sym, float(len(hc)) / C_sym,
        float(h in b0["p1"]), float(h in b0["p2"]),
    ], dtype=np.float64)


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
    pb_mms = {u: build_projection_bundle(all_data[u][0], all_data[u][1], anchor_idx=1, normalize_fn=norm_fn, project_fn=proj_fn) for u in all_uids}
    g1 = {u: build_phrase_graph(pb_omni[u], agreement_partition(pb_omni[u]), char_lm, word_lm, add_block_context_feats=True) for u in all_uids}
    g2 = {u: build_phrase_graph(pb_mms[u], agreement_partition(pb_mms[u]), char_lm, word_lm, add_block_context_feats=True) for u in all_uids}
    print(f"[{time.monotonic()-t0:.1f}s] Built bidirectional 32d context graphs.", flush=True)

    alpha_grid = (1.0, 0.5)
    preds_d1 = {s: {} for s in SEEDS}
    preds_d2 = {s: {} for s in SEEDS}
    preds_whole_d1 = {s: {} for s in SEEDS}
    preds_sym_sent = {s: {} for s in SEEDS}
    preds_sym_phrase = {s: {} for s in SEEDS}
    saved_full_artifact = None

    for seed in SEEDS:
        fs = folds[seed]
        for f in range(6):
            tr_ids = [u for u in fleurs_uids if f == 5 or fs[u]["outer"] != f]
            ev_ids = kathbath_uids if f == 5 else [u for u in fleurs_uids if fs[u]["outer"] == f]

            def inner_id(u):
                return fs[u]["full"] if f == 5 else fs[u]["inner"][str(f)]

            in_p1 = {a: {} for a in alpha_grid}
            in_p2 = {a: {} for a in alpha_grid}
            for j in range(3):
                in_tr = [u for u in tr_ids if inner_id(u) != j]
                in_va = [u for u in tr_ids if inner_id(u) == j]
                for a in alpha_grid:
                    m1, sc1 = fit_listwise_ensemble(in_tr, g1, meta, alpha_c=a, seed=1701 + j, n_ens=1, epochs=35)
                    m2, sc2 = fit_listwise_ensemble(in_tr, g2, meta, alpha_c=a, seed=1701 + j, n_ens=1, epochs=35)
                    p1, _ = predict_listwise_ensemble(g1, pb_omni, in_va, m1, sc1)
                    p2, _ = predict_listwise_ensemble(g2, pb_mms, in_va, m2, sc2)
                    in_p1[a].update(p1)
                    in_p2[a].update(p2)

            ev1 = {a: (evaluate_predictions(in_p1[a], meta, sorted(in_p1[a]))["macro_WER"],
                       evaluate_predictions(in_p1[a], meta, sorted(in_p1[a]))["macro_CER"]) for a in alpha_grid}
            ev2 = {a: (evaluate_predictions(in_p2[a], meta, sorted(in_p2[a]))["macro_WER"],
                       evaluate_predictions(in_p2[a], meta, sorted(in_p2[a]))["macro_CER"]) for a in alpha_grid}
            a1 = min(alpha_grid, key=lambda a: (ev1[a][0], ev1[a][1], -a))
            a2 = min(alpha_grid, key=lambda a: (ev2[a][0], ev2[a][1], -a))

            # Train strictly anti-symmetric residual-span & sentence arbiter on inner OOF (in_p1[a1], in_p2[a2])
            per_lang = defaultdict(list)
            for u in tr_ids:
                rw = meta[u]["reference"].split()
                per_lang[meta[u]["language"]].append([len(rw), len("".join(rw))])
            lscale = {l: np.mean(v, axis=0) for l, v in per_lang.items()}

            Xas, Yas, Xap, Yap = [], [], [], []
            for u in tr_ids:
                y1, y2 = in_p1[a1][u], in_p2[a2][u]
                if y1 == y2:
                    continue
                ref = meta[u]["reference"]
                sc = lscale[meta[u]["language"]]
                f1 = symmetric_sentence_feats(y1, pb_omni[u], char_lm, word_lm)
                f2 = symmetric_sentence_feats(y2, pb_omni[u], char_lm, word_lm)
                e1 = np.array([edit_counts(ref, y1)[0]["E"], edit_counts(ref, y1)[1]["E"]], dtype=np.float64) / sc
                e2 = np.array([edit_counts(ref, y2)[0]["E"], edit_counts(ref, y2)[1]["E"]], dtype=np.float64) / sc
                Xas.append(np.concatenate([f2 - f1, 0.5 * (f1 + f2)])); Yas.append(e2 - e1)
                Xas.append(np.concatenate([f1 - f2, 0.5 * (f1 + f2)])); Yas.append(e1 - e2)

                cells_r = proj_fn(y1, y1)
                proj_y2 = proj_fn(y1, y2)
                rranges, lo = [], 0
                for idx_c, cval in enumerate(cells_r):
                    if idx_c % 2 == 1 and proj_y2[idx_c] == cval:
                        if idx_c > lo:
                            rranges.append((lo, idx_c))
                        lo = idx_c + 1
                if lo < len(cells_r):
                    rranges.append((lo, len(cells_r)))
                for lo, hi in rranges:
                    pa, pb = flatten_cells(cells_r[lo:hi]), flatten_cells(proj_y2[lo:hi])
                    if pa == pb:
                        continue
                    hb = splice_cells(cells_r, [(lo, hi, pb)])
                    fb = symmetric_sentence_feats(hb, pb_omni[u], char_lm, word_lm)
                    eb = np.array([edit_counts(ref, hb)[0]["E"], edit_counts(ref, hb)[1]["E"]], dtype=np.float64) / sc
                    Xap.append(np.concatenate([fb - f1, 0.5 * (f1 + fb)])); Yap.append(eb - e1)
                    Xap.append(np.concatenate([f1 - fb, 0.5 * (f1 + fb)])); Yap.append(e1 - eb)

            arb_p = dict(max_iter=60, learning_rate=0.05, max_leaf_nodes=7, max_depth=3, min_samples_leaf=20, l2_regularization=5.0, early_stopping=False, random_state=1701)
            Xas_a, Yas_a = np.asarray(Xas), np.asarray(Yas)
            Xap_a, Yap_a = np.asarray(Xap), np.asarray(Yap)
            arb_sent = [HistGradientBoostingRegressor(**arb_p).fit(Xas_a, Yas_a[:, i]) for i in (0, 1)]
            arb_span = [HistGradientBoostingRegressor(**arb_p).fit(Xap_a, Yap_a[:, i]) for i in (0, 1)]

            # Outer full-fold models (3-seed Listwise ensemble for low variance!)
            m1_out, sc1_out = fit_listwise_ensemble(tr_ids, g1, meta, alpha_c=a1, seed=1701, n_ens=3, epochs=40)
            m2_out, sc2_out = fit_listwise_ensemble(tr_ids, g2, meta, alpha_c=a2, seed=1701, n_ens=3, epochs=40)
            ev_y1, ev_wh1 = predict_listwise_ensemble(g1, pb_omni, ev_ids, m1_out, sc1_out, need_whole=True)
            ev_y2, _ = predict_listwise_ensemble(g2, pb_mms, ev_ids, m2_out, sc2_out, need_whole=False)

            if seed == 17 and f == 5:
                saved_full_artifact = {
                    "m1_state": [m.state_dict() for m in m1_out],
                    "m2_state": [m.state_dict() for m in m2_out],
                    "sc1": sc1_out, "sc2": sc2_out, "a1": a1, "a2": a2,
                    "arb_sent": arb_sent, "arb_span": arb_span,
                }

            a_sym = 0.5 * (a1 + a2)
            for u in ev_ids:
                y1, y2 = ev_y1[u], ev_y2[u]
                preds_d1[seed][u] = y1
                preds_d2[seed][u] = y2
                preds_whole_d1[seed][u] = ev_wh1[u]
                if y1 == y2:
                    preds_sym_sent[seed][u] = y1
                    preds_sym_phrase[seed][u] = y1
                    continue
                f1 = symmetric_sentence_feats(y1, pb_omni[u], char_lm, word_lm)
                f2 = symmetric_sentence_feats(y2, pb_omni[u], char_lm, word_lm)
                xf = np.concatenate([f2 - f1, 0.5 * (f1 + f2)])[None, :]
                xr = np.concatenate([f1 - f2, 0.5 * (f1 + f2)])[None, :]
                dw = 0.5 * (arb_sent[0].predict(xf)[0] - arb_sent[0].predict(xr)[0])
                dc = 0.5 * (arb_sent[1].predict(xf)[0] - arb_sent[1].predict(xr)[0])
                r0 = max(max(0.0, -dw), a_sym * max(0.0, -dc))
                r1 = max(max(0.0, dw), a_sym * max(0.0, dc))
                preds_sym_sent[seed][u] = y2 if r1 < r0 else y1

                cells_r = proj_fn(y1, y1)
                proj_y2 = proj_fn(y1, y2)
                rranges, lo = [], 0
                for idx_c, cval in enumerate(cells_r):
                    if idx_c % 2 == 1 and proj_y2[idx_c] == cval:
                        if idx_c > lo:
                            rranges.append((lo, idx_c))
                        lo = idx_c + 1
                if lo < len(cells_r):
                    rranges.append((lo, len(cells_r)))
                acts = []
                for lo, hi in rranges:
                    pa, pb = flatten_cells(cells_r[lo:hi]), flatten_cells(proj_y2[lo:hi])
                    if pa == pb:
                        continue
                    hb = splice_cells(cells_r, [(lo, hi, pb)])
                    fb = symmetric_sentence_feats(hb, pb_omni[u], char_lm, word_lm)
                    xf = np.concatenate([fb - f1, 0.5 * (f1 + fb)])[None, :]
                    xr = np.concatenate([f1 - fb, 0.5 * (f1 + fb)])[None, :]
                    sw = 0.5 * (arb_span[0].predict(xf)[0] - arb_span[0].predict(xr)[0])
                    sc_v = 0.5 * (arb_span[1].predict(xf)[0] - arb_span[1].predict(xr)[0])
                    sr0 = max(max(0.0, -sw), a_sym * max(0.0, -sc_v))
                    sr1 = max(max(0.0, sw), a_sym * max(0.0, sc_v))
                    acts.append((lo, hi, pb if sr1 < sr0 else pa))
                preds_sym_phrase[seed][u] = splice_cells(cells_r, acts)
            print(f"  seed={seed} fold={f} done ({time.monotonic()-t0:.1f}s)", flush=True)

    # Save frozen full-FLEURS SymBiList checkpoint for zero-shot / external evaluation
    ckpt_path = ROOT / "results" / "frozen_symbilist_fleurs12.pkl"
    with ckpt_path.open("wb") as fp:
        pickle.dump(saved_full_artifact, fp)

    # Summarize & run paired cluster bootstrap against original Baseline A AND Whole fixed B!
    base_A = {u: bundle["baseline_preds"][17][u]["CTC_phrase"] for u in all_uids}
    base_B = {u: bundle["baseline_preds"][17][u]["CTC_whole_fixed"] for u in all_uids}

    out = {}
    for name, pd in [
        ("Listwise_dir1_omni", preds_d1),
        ("Listwise_dir2_mms", preds_d2),
        ("Listwise_whole_fixed", preds_whole_d1),
        ("SymBiList_sent", preds_sym_sent),
        ("SymBiList_phrase", preds_sym_phrase),
    ]:
        summ = {}
        for ds_name, ds_uids in [("FLEURS", fleurs_uids), ("Kathbath", kathbath_uids)]:
            ws, cs = [], []
            pl = defaultdict(lambda: {"WER": [], "CER": []})
            for s in SEEDS:
                ev = evaluate_predictions(pd[s], meta, ds_uids)
                ws.append(ev["macro_WER"]); cs.append(ev["macro_CER"])
                for l, lv in ev["per_lang"].items():
                    pl[l]["WER"].append(lv["WER"]); pl[l]["CER"].append(lv["CER"])
            summ[ds_name] = {
                "macro_WER_mean": float(np.mean(ws)), "macro_WER_sd": float(np.std(ws, ddof=1)),
                "macro_CER_mean": float(np.mean(cs)), "macro_CER_sd": float(np.std(cs, ddof=1)),
                "per_lang_mean": {l: {"WER": float(np.mean(v["WER"])), "CER": float(np.mean(v["CER"]))} for l, v in sorted(pl.items())},
            }
        ci_A_f = paired_cluster_bootstrap_diff(pd[17], base_A, meta, fleurs_uids, n_draws=20000)
        ci_A_k = paired_cluster_bootstrap_diff(pd[17], base_A, meta, kathbath_uids, n_draws=20000, stratify_by_language=True)
        ci_B_f = paired_cluster_bootstrap_diff(pd[17], base_B, meta, fleurs_uids, n_draws=20000)
        ci_B_k = paired_cluster_bootstrap_diff(pd[17], base_B, meta, kathbath_uids, n_draws=20000, stratify_by_language=True)
        out[name] = {"summary": summ, "ci_vs_A": {"FLEURS": ci_A_f, "Kathbath": ci_A_k}, "ci_vs_B": {"FLEURS": ci_B_f, "Kathbath": ci_B_k}}
        sf, sk = summ["FLEURS"], summ["Kathbath"]
        print(
            f"  {name:22s} | FLEURS {sf['macro_WER_mean']:6.3f}±{sf['macro_WER_sd']:.3f} / {sf['macro_CER_mean']:5.3f} "
            f"[vs A: {ci_A_f['delta_WER_pp']:+.3f} {ci_A_f['delta_WER_CI']}] | "
            f"Kathb {sk['macro_WER_mean']:6.3f} / {sk['macro_CER_mean']:5.3f} "
            f"[vs A: {ci_A_k['delta_WER_pp']:+.3f} {ci_A_k['delta_WER_CI']}]",
            flush=True,
        )

    out_path = ROOT / "results" / "stage1_symbilist.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[{time.monotonic()-t0:.1f}s] Saved SymBiList results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
