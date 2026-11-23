"""Models, OOF training loop, unsupervised MBR baselines, Direction A & Direction B learners."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Callable
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .align import flatten_cells, splice_cells, lev_dist
from .eval import edit_counts, evaluate_predictions


DEFAULT_HGB_PARAMS = dict(
    max_iter=100,
    learning_rate=0.05,
    max_leaf_nodes=7,
    max_depth=3,
    min_samples_leaf=25,
    l2_regularization=2.0,
    early_stopping=False,
    random_state=1701,
)


def prepare_fit_targets(
    graphs: dict[str, tuple],
    meta: dict[str, dict[str, Any]],
    uids: list[str],
) -> dict[str, tuple[np.ndarray, np.ndarray, list[int]]]:
    """Compute per-utterance (X, raw_edit_deltas[w,c], block_sizes) for non-anchor alternatives."""
    by_uid: dict[str, tuple[np.ndarray, np.ndarray, list[int]]] = {}
    for u in uids:
        cells, blocks = graphs[u]
        if not blocks:
            continue
        ref = meta[u]["reference"]
        base = " ".join(flatten_cells(cells))
        w0, c0 = edit_counts(ref, base)
        e0 = np.array([w0["E"], c0["E"]], dtype=np.float64)
        xs = []
        ys = []
        bsizes = []
        for bl in blocks:
            xs.append(bl["X"])
            bsizes.append(len(bl["X"]))
            for h in bl["full"][1:]:
                wh, ch = edit_counts(ref, h)
                ys.append([wh["E"] - e0[0], ch["E"] - e0[1]])
        if xs:
            by_uid[u] = (np.vstack(xs), np.asarray(ys, dtype=np.float64), bsizes)
    return by_uid


def build_training_arrays(
    train_uids: list[str],
    fit_targets: dict[str, tuple[np.ndarray, np.ndarray, list[int]]],
    meta: dict[str, dict[str, Any]],
    col_indices: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, list[int]]]]:
    per_lang = defaultdict(list)
    active = Counter()
    for u in train_uids:
        r = meta[u]
        rw = r["reference"].split()
        lang = r["language"]
        per_lang[lang].append([len(rw), len("".join(rw))])
        if u in fit_targets:
            active[lang] += 1
    scale = {l: np.mean(vals, axis=0) for l, vals in per_lang.items()}

    X_list, Y_list, W_list = [], [], []
    groups = []
    for u in train_uids:
        if u not in fit_targets:
            continue
        x, y, bsizes = fit_targets[u]
        if col_indices is not None:
            x = x[:, col_indices]
        lang = meta[u]["language"]
        X_list.append(x)
        Y_list.append(y / scale[lang])
        W_list.append(np.full(len(x), 1.0 / (active[lang] * len(x)), dtype=np.float64))
        groups.append((u, bsizes))

    X = np.vstack(X_list)
    Y = np.vstack(Y_list)
    W = np.concatenate(W_list)
    W *= len(W) / W.sum()
    return X, Y, W, groups


def fit_dual_hgb(
    train_uids: list[str],
    fit_targets: dict[str, tuple[np.ndarray, np.ndarray, list[int]]],
    meta: dict[str, dict[str, Any]],
    col_indices: list[int] | None = None,
    hgb_params: dict | None = None,
) -> list[HistGradientBoostingRegressor]:
    params = dict(DEFAULT_HGB_PARAMS)
    if hgb_params:
        params.update(hgb_params)
    X, Y, W, _ = build_training_arrays(train_uids, fit_targets, meta, col_indices=col_indices)
    return [
        HistGradientBoostingRegressor(**params).fit(X, Y[:, i], sample_weight=W)
        for i in (0, 1)
    ]


def fit_dual_regressor(
    model_kind: str,
    train_uids: list[str],
    fit_targets: dict[str, tuple[np.ndarray, np.ndarray, list[int]]],
    meta: dict[str, dict[str, Any]],
    col_indices: list[int] | None = None,
    seed: int = 1701,
) -> list[Any]:
    X, Y, W, _ = build_training_arrays(train_uids, fit_targets, meta, col_indices=col_indices)
    if model_kind == "hgb":
        params = dict(DEFAULT_HGB_PARAMS, random_state=seed)
        return [HistGradientBoostingRegressor(**params).fit(X, Y[:, i], sample_weight=W) for i in (0, 1)]
    elif model_kind == "hgb_deep":
        params = dict(DEFAULT_HGB_PARAMS, max_iter=150, max_leaf_nodes=15, max_depth=4, min_samples_leaf=20, random_state=seed)
        return [HistGradientBoostingRegressor(**params).fit(X, Y[:, i], sample_weight=W) for i in (0, 1)]
    elif model_kind == "ridge":
        models = []
        for i in (0, 1):
            pipe = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            pipe.fit(X, Y[:, i], ridge__sample_weight=W)
            models.append(pipe)
        return models
    elif model_kind == "mlp":
        models = []
        for i in (0, 1):
            pipe = make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    alpha=1e-2,
                    learning_rate_init=1e-3,
                    max_iter=80,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=seed + i,
                ),
            )
            pipe.fit(X, Y[:, i])
            models.append(pipe)
        return models
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")


def predict_block_costs(
    graphs: dict[str, tuple],
    uids: list[str],
    models: list[Any],
    col_indices: list[int] | None = None,
) -> dict[str, list[np.ndarray]]:
    all_x = []
    for u in uids:
        for bl in graphs[u][1]:
            x = bl["X"] if col_indices is None else bl["X"][:, col_indices]
            all_x.append(x)
    if not all_x:
        return {u: [] for u in uids}
    stacked = np.vstack(all_x)
    pred = np.stack([m.predict(stacked) for m in models], axis=1)
    pos = 0
    out: dict[str, list[np.ndarray]] = {}
    for u in uids:
        zz = []
        for bl in graphs[u][1]:
            n = len(bl["X"])
            zz.append(np.vstack((np.zeros((1, 2), dtype=np.float64), pred[pos : pos + n])))
            pos += n
        out[u] = zz
    assert pos == len(pred)
    return out


def choose_from_block_costs(
    graph: tuple,
    block_costs: list[np.ndarray],
    bundle: dict,
    alpha_c: float = 0.5,
    objective: str = "minimax",
    need_whole: bool = False,
) -> tuple[str, str | None]:
    cells, blocks = graph
    actions = []
    pool = bundle["texts"]
    whole_cost = np.zeros(len(pool), dtype=np.float64)
    proj = bundle["proj"]

    for bl, z in zip(blocks, block_costs):
        lo, hi, opts = bl["lo"], bl["hi"], bl["alts"]
        reg = z - z.min(axis=0)
        if objective == "minimax":
            score = np.maximum(reg[:, 0], alpha_c * reg[:, 1])
        elif objective == "weighted_sum":
            score = reg[:, 0] + alpha_c * reg[:, 1]
        elif objective == "word_only":
            score = z[:, 0]
        elif objective == "char_only":
            score = z[:, 1]
        else:
            raise ValueError(f"Unknown objective: {objective}")
        best_j = int(np.argmin(score))
        actions.append((lo, hi, opts[best_j]))
        if need_whole:
            for pi, h in enumerate(pool):
                ph = flatten_cells(proj[h][lo:hi])
                whole_cost[pi] += score[opts.index(ph)]

    hyp = splice_cells(cells, actions)
    whole_hyp = pool[int(np.argmin(whole_cost))] if need_whole else None
    return hyp, whole_hyp


def run_grouped_oof(
    graphs: dict[str, tuple],
    bundles: dict[str, dict],
    meta: dict[str, dict[str, Any]],
    folds_by_seed: dict[int, dict[str, dict]],
    fleurs_uids: list[str],
    kathbath_uids: list[str],
    col_indices: list[int] | None = None,
    model_kind: str = "hgb",
    objective: str = "minimax",
    alpha_grid: tuple[float, ...] = (1.0, 0.5),
    need_whole: bool = False,
    train_frac: float = 1.0,
    seeds: tuple[int, ...] = (17, 42, 2026),
) -> dict[str, Any]:
    """Run exact 3-seed x 5-outer x 3-inner grouped OOF on FLEURS + full refit on Kathbath valid."""
    fit_targets = prepare_fit_targets(graphs, meta, fleurs_uids)
    all_preds: dict[int, dict[str, str]] = {}
    all_whole_preds: dict[int, dict[str, str]] = {}
    fold_models_by_seed: dict[int, dict[int, Any]] = {}
    selection_log: dict[int, list[dict]] = {}

    for seed in seeds:
        fs = folds_by_seed[seed]
        preds: dict[str, str] = {}
        whole_preds: dict[str, str] = {}
        fold_models_by_seed[seed] = {}
        selection_log[seed] = []

        for f in range(6):
            tr_ids = [u for u in fleurs_uids if f == 5 or fs[u]["outer"] != f]
            ev_ids = kathbath_uids if f == 5 else [u for u in fleurs_uids if fs[u]["outer"] == f]

            if train_frac < 1.0:
                # Subsample intact prompt groups deterministically per (seed, fold)
                groups_sorted = sorted({meta[u]["cluster"] for u in tr_ids})
                rng = np.random.Generator(np.random.PCG64(seed * 100 + f))
                k_keep = max(3, int(round(len(groups_sorted) * train_frac)))
                keep_groups = set(rng.choice(groups_sorted, size=k_keep, replace=False).tolist())
                tr_ids = [u for u in tr_ids if meta[u]["cluster"] in keep_groups]

            if len(alpha_grid) == 1:
                winner = alpha_grid[0]
                inner_ev = None
            else:
                def inner_id(u: str) -> int:
                    return fs[u]["full"] if f == 5 else fs[u]["inner"][str(f)]

                inner_preds = {a: {} for a in alpha_grid}
                for j in range(3):
                    in_tr = [u for u in tr_ids if inner_id(u) != j]
                    in_va = [u for u in tr_ids if inner_id(u) == j]
                    if not in_tr or not in_va:
                        continue
                    in_mod = fit_dual_regressor(model_kind, in_tr, fit_targets, meta, col_indices=col_indices, seed=1701)
                    in_scores = predict_block_costs(graphs, in_va, in_mod, col_indices=col_indices)
                    for a in alpha_grid:
                        for u in in_va:
                            h, _ = choose_from_block_costs(graphs[u], in_scores[u], bundles[u], alpha_c=a, objective=objective)
                            inner_preds[a][u] = h
                inner_ev = {}
                for a in alpha_grid:
                    ev = evaluate_predictions(inner_preds[a], meta, sorted(inner_preds[a].keys()))
                    inner_ev[a] = (ev["macro_WER"], ev["macro_CER"])
                winner = min(alpha_grid, key=lambda a: (inner_ev[a][0], inner_ev[a][1], -a))

            mod = fit_dual_regressor(model_kind, tr_ids, fit_targets, meta, col_indices=col_indices, seed=1701)
            fold_models_by_seed[seed][f] = {"models": mod, "alpha": winner}
            ev_scores = predict_block_costs(graphs, ev_ids, mod, col_indices=col_indices)
            for u in ev_ids:
                h, wh = choose_from_block_costs(
                    graphs[u], ev_scores[u], bundles[u], alpha_c=winner, objective=objective, need_whole=need_whole
                )
                preds[u] = h
                if need_whole and wh is not None:
                    whole_preds[u] = wh
            selection_log[seed].append({"fold": f, "winner": winner, "n_train": len(tr_ids)})

        all_preds[seed] = preds
        if need_whole:
            all_whole_preds[seed] = whole_preds

    # Aggregate 3-seed mean +/- split SD
    summary: dict[str, Any] = {}
    for ds_name, ds_uids in [("FLEURS", fleurs_uids), ("Kathbath", kathbath_uids)]:
        wers, cers, mwers, mcers = [], [], [], []
        per_lang_agg = defaultdict(lambda: {"WER": [], "CER": []})
        for seed in seeds:
            ev = evaluate_predictions(all_preds[seed], meta, ds_uids)
            wers.append(ev["macro_WER"])
            cers.append(ev["macro_CER"])
            mwers.append(ev["micro_WER"])
            mcers.append(ev["micro_CER"])
            for l, lv in ev["per_lang"].items():
                per_lang_agg[l]["WER"].append(lv["WER"])
                per_lang_agg[l]["CER"].append(lv["CER"])
        summary[ds_name] = {
            "macro_WER_mean": float(np.mean(wers)),
            "macro_WER_sd": float(np.std(wers, ddof=1)) if len(wers) > 1 else 0.0,
            "macro_CER_mean": float(np.mean(cers)),
            "macro_CER_sd": float(np.std(cers, ddof=1)) if len(cers) > 1 else 0.0,
            "micro_WER_mean": float(np.mean(mwers)),
            "micro_CER_mean": float(np.mean(mcers)),
            "per_seed_WER": wers,
            "per_seed_CER": cers,
            "per_lang_mean": {
                l: {"WER": float(np.mean(v["WER"])), "CER": float(np.mean(v["CER"]))}
                for l, v in sorted(per_lang_agg.items())
            },
        }

    return {
        "summary": summary,
        "preds": all_preds,
        "whole_preds": all_whole_preds if need_whole else None,
        "models": fold_models_by_seed,
        "selection_log": selection_log,
    }


# ---------------- Unsupervised Baselines (Exact MBR & Alignment Cell Voting) ----------------

def predict_exact_nbest_mbr(
    bundles: dict[str, dict],
    uids: list[str],
    metric: str = "word",
    source_weight_1: float = 0.5,
    char_lm_fn: Callable[[str], float] | None = None,
    lm_weight: float = 0.0,
) -> dict[str, str]:
    """Exact Multi-Source Model-Based MBR (Jinnai et al., ICML 2024 style) over pooled N-best union."""
    out: dict[str, str] = {}
    for u in uids:
        b = bundles[u]
        pool = b["texts"]
        if len(pool) == 1:
            out[u] = pool[0]
            continue
        p_mix = Counter()
        for h, w in b["p1"].items():
            p_mix[h] += source_weight_1 * w
        for h, w in b["p2"].items():
            p_mix[h] += (1.0 - source_weight_1) * w

        items = list(p_mix.items())
        best_h = pool[0]
        best_score = float("inf")
        A = max(1, len(b["base"].split()))
        C = max(1, len("".join(b["base"].split())))
        for h in pool:
            hw = h.split()
            hc = "".join(hw)
            if metric == "word":
                risk = sum(w * lev_dist(hw, k.split()) for k, w in items) / A
            elif metric == "char":
                risk = sum(w * lev_dist(hc, "".join(k.split())) for k, w in items) / C
            elif metric == "dual":
                rw = sum(w * lev_dist(hw, k.split()) for k, w in items) / A
                rc = sum(w * lev_dist(hc, "".join(k.split())) for k, w in items) / C
                risk = rw + 0.5 * rc
            else:
                raise ValueError(metric)
            if lm_weight > 0.0 and char_lm_fn is not None:
                # Subtract normalized char4 log-prob per char
                risk -= lm_weight * (float(char_lm_fn(h)) / (len(h) + 1))
            if risk < best_score - 1e-12:
                best_score = risk
                best_h = h
        out[u] = best_h
    return out


def predict_unsupervised_block_mbr(
    graphs: dict[str, tuple],
    uids: list[str],
    metric: str = "word",
) -> dict[str, str]:
    """Unsupervised Agreement-Anchored Block MBR (selects argmin expected edit risk per block)."""
    out: dict[str, str] = {}
    for u in uids:
        cells, blocks = graphs[u]
        acts = []
        for bl in blocks:
            scores = bl["word_risk"] if metric == "word" else bl["char_risk"]
            j = int(np.argmin(scores))
            acts.append((bl["lo"], bl["hi"], bl["alts"][j]))
        out[u] = splice_cells(cells, acts)
    return out


def predict_alignment_cell_vote(bundles: dict[str, dict], uids: list[str]) -> dict[str, str]:
    """Unsupervised slot-level weighted majority vote on anchor-aligned cells."""
    out: dict[str, str] = {}
    for u in uids:
        b = bundles[u]
        p_mix = Counter()
        for h, w in b["p1"].items():
            p_mix[h] += 0.5 * w
        for h, w in b["p2"].items():
            p_mix[h] += 0.5 * w
        proj = b["proj"]
        n_cells = len(b["cells"])
        words = []
        for i in range(n_cells):
            max_len = max(len(proj[h][i]) for h in p_mix)
            for j in range(max_len):
                tally = Counter()
                for h, w in p_mix.items():
                    tok = proj[h][i][j] if j < len(proj[h][i]) else None
                    tally[tok] += w
                best_tok = max(tally.keys(), key=lambda k: tally[k])
                if best_tok is not None:
                    words.append(best_tok)
        out[u] = " ".join(words)
    return out
