"""Slot-collapse & specialisation analysis for a trained M3-SSR checkpoint.

Given a saved *.best.pt checkpoint, runs the TEST set once and reports:
  1. slot pairwise cosine similarity (collapse indicator: ->1 means collapse)
  2. WTA winner distribution + usage entropy (winner = slot with min CE)
  3. account-gate weight stats + gate entropy
  4. winner-slot distribution per target-popularity bucket (specialisation)
Also dumps per-sample hit@5/10/20 and reciprocal ranks to .npz for
paired significance tests (see sig_test.py).

Example:
    python analyze_slots.py --dataset Hvideo_E \
        --ckpt runs/Hvideo_E_m3ssr_d64_s42.best.pt --out analysis/Hvideo_E_full
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from dataset import build_loaders
from models.backbones import build_model


def entropy(p):
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Hvideo_E",
                    choices=["Hvideo_E", "Hvideo_V", "Hamazon_M", "Hamazon_B"])
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--ckpt", required=True, help="path to *.best.pt")
    ap.add_argument("--out", required=True, help="output prefix (no extension)")
    ap.add_argument("--max_len", type=int, default=50)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--n_users", type=int, default=4)
    ap.add_argument("--ablation", default="none")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data, _, te_loader = build_loaders(
        args.data_root, args.dataset, max_len=args.max_len,
        batch_size=args.batch_size)

    model = build_model("m3ssr", data.n_items, d_model=args.d_model,
                        max_len=args.max_len, n_layers=args.n_layers,
                        dropout=args.dropout,
                        n_accounts=data.n_accounts,
                        interactions=data.interactions,
                        n_users=args.n_users,
                        ablation=args.ablation).to(device)
    model.setup_graph(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device,
                                     weights_only=True))
    model.eval()

    # item popularity from train interactions -> tertile buckets
    pop = np.zeros(data.n_items + 1, dtype=np.int64)      # 1-based items
    for _, it in data.interactions:
        pop[it + 1] += 1
    order = np.argsort(-pop[1:])                          # item idx 0-based
    bucket = np.full(data.n_items + 1, 2, dtype=np.int64) # default tail
    third = max(len(order) // 3, 1)
    bucket[order[:third] + 1] = 0                         # head
    bucket[order[third:2 * third] + 1] = 1                # mid

    K = args.n_users
    win_counts = np.zeros(K, dtype=np.int64)
    win_by_bucket = np.zeros((3, K), dtype=np.int64)
    cos_sum = np.zeros((K, K))
    gate_sum = np.zeros(K)
    gate_ent, n = [], 0
    hits = {k: [] for k in (5, 10, 20)}
    rr = []

    with torch.no_grad():
        for seq, tgt, acc in te_loader:
            seq, tgt, acc = seq.to(device), tgt.to(device), acc.to(device)
            h_final, h_a, users = model.encode(seq, acc)  # users [B,K,D]
            B = seq.size(0)
            n += B

            # --- per-slot CE -> WTA winners -------------------------------
            per_slot = []
            for k in range(users.size(1)):
                lu = model.out(torch.cat([users[:, k], h_a], dim=-1))
                per_slot.append(F.cross_entropy(lu, tgt - 1, reduction='none'))
            per_slot = torch.stack(per_slot, dim=0)       # [K, B]
            winners = per_slot.argmin(dim=0).cpu().numpy()  # [B]
            for b, w in enumerate(winners):
                win_counts[w] += 1
                win_by_bucket[bucket[tgt[b].item()], w] += 1

            # --- slot cosine similarity ------------------------------------
            u = F.normalize(users, dim=-1)
            cos_sum += torch.bmm(u, u.transpose(1, 2)).sum(0).cpu().numpy()

            # --- gate stats -------------------------------------------------
            g = torch.softmax(model.gate(h_a), dim=-1)    # [B, K]
            gate_sum += g.sum(0).cpu().numpy()
            gate_ent.extend((-(g * (g + 1e-12).log()).sum(-1)).cpu().numpy())

            # --- per-sample ranking metrics (for significance tests) -------
            logits = model(seq, acc)
            top = logits.topk(20, dim=1).indices + 1
            for b in range(B):
                rl = top[b].tolist()
                t = tgt[b].item()
                r = rl.index(t) + 1 if t in rl else 10 ** 9
                for k in (5, 10, 20):
                    hits[k].append(1.0 if r <= k else 0.0)
                rr.append(1.0 / r if r <= 20 else 0.0)

    cos_mat = cos_sum / max(n, 1)
    off_diag = [cos_mat[i, j] for i in range(K) for j in range(K) if i != j]
    win_dist = win_counts / max(win_counts.sum(), 1)
    gate_mean = gate_sum / max(n, 1)

    report = {
        "dataset": args.dataset, "ckpt": args.ckpt, "n_test": n,
        "slot_cosine_mean_offdiag": float(np.mean(off_diag)),
        "slot_cosine_max_offdiag": float(np.max(off_diag)),
        "winner_distribution": win_dist.tolist(),
        "winner_usage_entropy": entropy(win_dist),
        "winner_usage_entropy_max": math.log(K),
        "gate_mean": gate_mean.tolist(),
        "gate_entropy_mean": float(np.mean(gate_ent)),
        "gate_entropy_max": math.log(K),
        "winner_by_popularity_bucket": {
            "head": (win_by_bucket[0] / max(win_by_bucket[0].sum(), 1)).tolist(),
            "mid": (win_by_bucket[1] / max(win_by_bucket[1].sum(), 1)).tolist(),
            "tail": (win_by_bucket[2] / max(win_by_bucket[2].sum(), 1)).tolist(),
        },
        "Recall@5": float(np.mean(hits[5])),
        "MRR@20(from_rr)": float(np.mean(rr)),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump(report, f, indent=2)
    np.savez(args.out + ".npz", hit5=hits[5], hit10=hits[10],
             hit20=hits[20], rr=rr, winners=winners)

    # --- figure ------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(9, 7))
        axes[0, 0].bar(range(K), win_dist)
        axes[0, 0].set_title("WTA winner distribution")
        axes[0, 0].set_xlabel("slot"); axes[0, 0].set_ylabel("fraction")
        im = axes[0, 1].imshow(cos_mat, vmin=-1, vmax=1, cmap="RdBu_r")
        axes[0, 1].set_title("slot cosine similarity")
        fig.colorbar(im, ax=axes[0, 1])
        bottom = np.zeros(3)
        for k in range(K):
            col = win_by_bucket[:, k] / np.maximum(win_by_bucket.sum(1), 1)
            axes[1, 0].bar(["head", "mid", "tail"], col, bottom=bottom,
                           label=f"slot {k}")
            bottom += col
        axes[1, 0].set_title("winner mix by target popularity")
        axes[1, 0].legend(fontsize=8)
        axes[1, 1].hist(gate_ent, bins=40)
        axes[1, 1].axvline(math.log(K), color="r", ls="--", label="max")
        axes[1, 1].set_title("per-sample gate entropy")
        axes[1, 1].legend()
        fig.tight_layout()
        fig.savefig(args.out + ".png", dpi=200)
        print("figure:", args.out + ".png")
    except Exception as e:
        print("figure skipped:", e)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
