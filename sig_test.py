"""Paired significance tests between two runs' per-sample dumps (.npz
produced by analyze_slots.py).

Compares full model vs ablation on the SAME test samples:
  - paired bootstrap (10k resamples) on the metric difference
  - Wilcoxon signed-rank on per-sample reciprocal ranks (MRR)

Example:
    python sig_test.py analysis/Hvideo_E_full.npz analysis/Hvideo_E_noslots.npz
"""
import argparse
import numpy as np


def bootstrap_diff(a, b, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(a[idx].mean() - b[idx].mean())
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # two-sided bootstrap p: fraction of resamples crossing zero
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return a.mean() - b.mean(), lo, hi, min(p, 1.0)


def wilcoxon(a, b):
    try:
        from scipy.stats import wilcoxon as _w
        stat, p = _w(a, b, zero_method="zsplit")
        return float(p)
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("full", help="npz of the full model")
    ap.add_argument("ablation", help="npz of the ablation run")
    args = ap.parse_args()
    A = np.load(args.full)
    B = np.load(args.ablation)

    print(f"full: {args.full}   ablation: {args.ablation}")
    for key, name in [("hit5", "Recall@5"), ("hit10", "Recall@10"),
                      ("hit20", "Recall@20"), ("rr", "MRR@20")]:
        d, lo, hi, p = bootstrap_diff(A[key], B[key])
        sig = "SIGNIFICANT" if p < 0.05 else "n.s."
        print(f"{name:10s} diff={d:+.4f}  95%CI=[{lo:+.4f},{hi:+.4f}]  "
              f"p_boot={p:.4f}  {sig}")
    p_w = wilcoxon(A["rr"], B["rr"])
    print(f"Wilcoxon on reciprocal ranks: p={p_w:.4g}")


if __name__ == "__main__":
    main()
