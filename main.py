"""Entry point. Example:
    python main.py --dataset Hvideo_E --model sasrec --epochs 50
    python main.py --dataset Hvideo_E --model mambarec --epochs 50
"""
import argparse
import os
import torch

from dataset import build_loaders
from models.backbones import build_model
from trainer import train, evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Hvideo_E",
                   choices=["Hvideo_E", "Hvideo_V", "Hamazon_M", "Hamazon_B"])
    p.add_argument("--model", default="sasrec")
    p.add_argument("--data_root", default="data")
    p.add_argument("--max_len", type=int, default=50)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--beta", type=float, default=0.5, help="aux loss weight (m3ssr/disenreason)")
    p.add_argument("--n_users", type=int, default=4, help="latent user slots (m3ssr)")
    p.add_argument("--ablation", default="none",
                   help="none | fft | transformer | noslots | nogcn | combos like fft+nogcn")
    p.add_argument("--patience", type=int, default=1000)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--dense", action="store_true")
    p.add_argument("--test_hist", action="store_true",
                   help="prepend train history to test inputs (LightGC2N-style)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="runs")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data, tr_loader, te_loader = build_loaders(
        args.data_root, args.dataset, max_len=args.max_len,
        batch_size=args.batch_size, dense=args.dense)
    print("stats:", data.stats())

    model = build_model(args.model, data.n_items, d_model=args.d_model,
                        max_len=args.max_len, n_layers=args.n_layers,
                        dropout=args.dropout,
                        n_accounts=data.n_accounts,
                        interactions=data.interactions,
                        beta=args.beta, n_users=args.n_users,
                        ablation=args.ablation).to(device)
    if hasattr(model, "setup_graph"):
        model.setup_graph(device)
    print(f"params: {sum(x.numel() for x in model.parameters())/1e6:.2f}M")

    os.makedirs(args.out, exist_ok=True)
    tag = f"_{args.ablation}" if args.ablation != "none" else ""
    log = os.path.join(args.out, f"{args.dataset}_{args.model}_d{args.d_model}_s{args.seed}{tag}")
    best, best_ep = train(model, tr_loader, te_loader, device,
                          epochs=args.epochs, lr=args.lr,
                          patience=args.patience, log_path=log, eval_every=args.eval_every, dense=args.dense,
                          weight_decay=args.weight_decay)

    model.load_state_dict(torch.load(log + ".best.pt", weights_only=True))
    final = evaluate(model, te_loader, device)
    print("FINAL (best ckpt):", {k: round(v, 4) for k, v in final.items()})
    with open(log + ".final.json", "w") as f:
        import json
        json.dump(final, f, indent=2)


if __name__ == "__main__":
    main()
