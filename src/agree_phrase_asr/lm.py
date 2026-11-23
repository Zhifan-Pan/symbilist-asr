"""Clean, fast, standalone Witten-Bell character 4-gram and interpolated word 2-gram LMs."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterable


class CharNgramLM:
    """Character n-gram LM retaining spaces + EOS with Witten-Bell backoff and additive-0.1 unigrams."""

    def __init__(self, order: int = 4, alpha: float = 0.1):
        self.order = order
        self.alpha = alpha
        self.counts: list[Counter] = [Counter() for _ in range(order)]
        self.ctx_totals: list[Counter] = [Counter() for _ in range(order)]
        self.ctx_types: list[Counter] = [Counter() for _ in range(order)]
        self.total_uni = 0
        self.vocab_size = 0
        self.BOS = "\x02"
        self.EOS = "\x03"

    def fit(self, texts: Iterable[str]) -> "CharNgramLM":
        vocab = set()
        for t in texts:
            seq = (self.BOS * (self.order - 1)) + t + self.EOS
            for ch in t:
                vocab.add(ch)
            vocab.add(self.EOS)
            n = len(seq)
            for i in range(self.order - 1, n):
                ch = seq[i]
                self.counts[0][ch] += 1
                self.total_uni += 1
                for k in range(2, self.order + 1):
                    ctx = seq[i - k + 1 : i]
                    ng = seq[i - k + 1 : i + 1]
                    if self.counts[k - 1][ng] == 0:
                        self.ctx_types[k - 1][ctx] += 1
                    self.counts[k - 1][ng] += 1
                    self.ctx_totals[k - 1][ctx] += 1
        # +1 for unknown character symbol
        self.vocab_size = len(vocab) + 1
        return self

    def prob_char(self, ctx: str, ch: str) -> float:
        p = (self.counts[0].get(ch, 0) + self.alpha) / (self.total_uni + self.alpha * self.vocab_size)
        for k in range(2, self.order + 1):
            sub_ctx = ctx[-(k - 1) :]
            tot = self.ctx_totals[k - 1].get(sub_ctx, 0)
            if tot > 0:
                t = self.ctx_types[k - 1][sub_ctx]
                c = self.counts[k - 1].get(sub_ctx + ch, 0)
                p = (c + t * p) / (tot + t)
        return p

    def score(self, text: str) -> float:
        seq = (self.BOS * (self.order - 1)) + text + self.EOS
        ll = 0.0
        for i in range(self.order - 1, len(seq)):
            ctx = seq[i - self.order + 1 : i]
            ll += math.log(max(self.prob_char(ctx, seq[i]), 1e-30))
        return ll


class WordBigramLM:
    """Interpolated word bigram LM with c/(c+t) interpolation weight and additive-0.1 unigrams."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.uni = Counter()
        self.bi = Counter()
        self.ctx_tot = Counter()
        self.ctx_types = Counter()
        self.total_tok = 0
        self.vocab: set[str] = set()
        self.BOS = "<BOS>"
        self.EOS = "<EOS>"

    def fit(self, texts: Iterable[str]) -> "WordBigramLM":
        for t in texts:
            toks = t.split()
            for w in toks:
                self.vocab.add(w)
            seq = [self.BOS, *toks, self.EOS]
            for i in range(1, len(seq)):
                prev, cur = seq[i - 1], seq[i]
                self.uni[cur] += 1
                self.total_tok += 1
                pair = (prev, cur)
                if self.bi[pair] == 0:
                    self.ctx_types[prev] += 1
                self.bi[pair] += 1
                self.ctx_tot[prev] += 1
        self.vocab.add(self.EOS)
        return self

    def scores(self, text: str) -> tuple[float, float, float]:
        toks = text.split()
        seq = [self.BOS, *toks, self.EOS]
        v_size = len(self.vocab) + 1
        ll = 0.0
        unks = 0
        for i in range(1, len(seq)):
            prev, cur = seq[i - 1], seq[i]
            if cur not in self.vocab:
                unks += 1
            p_uni = (self.uni.get(cur, 0) + self.alpha) / (self.total_tok + self.alpha * v_size)
            c_tot = self.ctx_tot.get(prev, 0)
            if c_tot > 0:
                t = self.ctx_types[prev]
                lam = c_tot / (c_tot + t)
                p = lam * (self.bi.get((prev, cur), 0) / c_tot) + (1.0 - lam) * p_uni
            else:
                p = p_uni
            ll += math.log(max(p, 1e-30))
        n_tok = len(toks) + 1
        return (ll / n_tok, ll / (len(text) + 1), unks / n_tok)
