"""Training & evaluation loop. Full-ranking metrics: Recall / NDCG / MRR @ K."""
import json
import os
import time
import torch
import torch.nn.functional as F


def evaluate(model, loader, device, ks=(5, 10, 20)):
    model.eval()
    max_k = max(ks)
    hits = {k: 0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    mrr = {k: 0.0 for k in ks}
    n = 0
    with torch.no_grad():
        for seq, tgt, acc in loader:
            seq, tgt, acc = seq.to(device), tgt.to(device), acc.to(device)
            logits = model(seq, acc)                            # [B, n_items]
            top = logits.topk(max_k, dim=1).indices + 1    # back to 1-based
            for b in range(seq.size(0)):
                t = tgt[b].item()
                n += 1
                rank_list = top[b].tolist()
                if t in rank_list:
                    r = rank_list.index(t) + 1
                    for k in ks:
                        if r <= k:
                            hits[k] += 1
                            ndcg[k] += 1.0 / torch.log2(torch.tensor(r + 1.0)).item()
                            mrr[k] += 1.0 / r
    out = {}
    for k in ks:
        out[f"Recall@{k}"] = hits[k] / max(n, 1)
        out[f"NDCG@{k}"] = ndcg[k] / max(n, 1)
        out[f"MRR@{k}"] = mrr[k] / max(n, 1)
    return out


def train(model, tr_loader, te_loader, device, epochs=50, lr=1e-3,
          patience=10, log_path=None, eval_every=2, dense=False,
          weight_decay=0.0):
    """log_path: prefix without extension -> writes {log_path}.json / {log_path}.best.pt"""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_epoch, bad = -1.0, -1, 0
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for seq, tgt, acc in tr_loader:
            seq, tgt, acc = seq.to(device), tgt.to(device), acc.to(device)
            if hasattr(model, "compute_loss"):        # e.g. DisenReason with aux loss
                loss = model.compute_loss(seq, tgt, acc)
            elif dense and hasattr(model, "full_logits"):
                logits = model.full_logits(seq)       # [B, L, n_items]
                mask = tgt != 0                        # tgt here is per-position
                loss = F.cross_entropy(
                    logits[mask], tgt[mask] - 1)
            else:
                logits = model(seq, acc)
                loss = F.cross_entropy(logits, tgt - 1)   # logits are 0-based over items
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * seq.size(0)
        msg = f"epoch {ep:3d} | loss {tot/len(tr_loader.dataset):.4f} | {time.time()-t0:.1f}s"
        if ep % eval_every == 0 or ep == epochs:
            m = evaluate(model, te_loader, device)
            msg += " | " + " ".join(f"{k}={v:.4f}" for k, v in m.items())
            score = m["NDCG@10"]
            history.append({"epoch": ep, **m})
            if score > best:
                best, best_epoch, bad = score, ep, 0
                torch.save(model.state_dict(), log_path + ".best.pt" if log_path else "best.pt")
            else:
                bad += eval_every
        print(msg, flush=True)
        if log_path:
            with open(log_path + ".json", "w") as f:
                json.dump(history, f, indent=2)
        if bad >= patience:
            print(f"early stop, best NDCG@10={best:.4f} @epoch {best_epoch}")
            break
    return best, best_epoch
