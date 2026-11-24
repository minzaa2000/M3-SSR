# M3-SSR: Competitive Latent-User Disentanglement with Wavelet-Mamba Networks for Shared-Account Sequential Recommendation

Source code for the experiments reported in the manuscript.

## Overview

M3-SSR is a shared-account sequential recommender built on three components:

1. a learnable multi-resolution wavelet disentanglement module (Haar-initialized two-tap filters),
2. a linear-time selective state-space (Mamba) sequence encoder,
3. competitive latent-user slots trained with winner-take-all (WTA) supervision, a pairwise diversity regularizer, and an account-conditioned gate.

The repository also contains our faithful reproduction of DisenReason and unified SASRec / MambaRec baselines, all evaluated under an identical protocol (full ranking over the entire item set, chronological 80/20 split, max sequence length 50).

## Repository structure

```
├── dataset.py            # Data loading (left-padded, PAD=0, item indices shifted by +1)
├── trainer.py            # Training loop + full-ranking evaluation (Recall/NDCG/MRR @5/10/20)
├── main.py               # Entry point
├── models/
│   ├── backbones.py      # SASRec, MambaRec (unified interface)
│   ├── disenreason.py    # DisenReason reproduction (LightGCN init + rFFT bands + progressive reasoning)
│   └── m3ssr.py          # M3-SSR (wavelet frontend + Mamba encoder + competitive slots)
├── analyze_slots.py      # Slot-specialisation diagnostics (cosine similarity, winner/gate entropy,
│                         # popularity-bucket analysis, per-sample scores -> .json/.npz/.png)
├── sig_test.py           # Paired bootstrap (10k resamples) + Wilcoxon signed-rank test on two .npz files
├── run_all.sh            # Batch runner for four datasets
└── requirements.txt
```

## Setup

Tested on a single RTX 4090, PyTorch 2.1+, CUDA 12.1, Python 3.10.

```bash
pip install -r requirements.txt
python -c "from mamba_ssm import Mamba; import torch; \
  m=Mamba(d_model=64).cuda(); print(m(torch.randn(2,100,64).cuda()).shape)"
```

## Data

We use the four public shared-account benchmarks Hvideo-E, Hvideo-V, Hamazon-M and Hamazon-B, introduced by Ma et al. (2019, π-Net, SIGIR) and adopted by subsequent SSR work. The preprocessed splits follow the LightGC2N (AAAI 2025) / DisenReason protocol. Place each dataset as:

```
data/<Dataset>/{train_data.txt, test_data.txt, item_dict.txt}
```

Each line of `train_data.txt` is `account_id<TAB>item1 item2 ...`; in `test_data.txt` the last item is the held-out target.

## Reproducing the results

### Table 2 (overall performance, identical protocol)

```bash
python main.py --dataset Hvideo_E --model sasrec --seed 42
python main.py --dataset Hvideo_E --model mambarec --seed 42                 # Mamba4Rec baseline
python main.py --dataset Hvideo_E --model disenreason --d_model 64 --epochs 100
python main.py --dataset Hvideo_E --model m3ssr --ablation nogcn --seed 42   # main model (plain embeddings)
```

Repeat for `Hvideo_V`, `Hamazon_M`, `Hamazon_B`. Seed-robustness rows (Table 4) use `--seed 0/1/42`.

### Table 3 / Table 5 (ablations and graph-initialized variant)

Ablation switches (combinable with `+`):

| Flag | Meaning |
|---|---|
| `fft` | fixed equal-width rFFT bands replace the learnable wavelet |
| `transformer` | Transformer encoder replaces the Mamba backbone |
| `noslots` | remove latent-user slots and auxiliary losses |
| `nogcn` | plain learned embeddings replace LightGCN initialization |
| `uniform` | uniform slot supervision (mean CE) replaces WTA (min CE) |
| `nodiv` | remove the pairwise diversity regularizer |
| `nogate` | uniform 1/K mixing replaces the account-conditioned gate |

Example:

```bash
python main.py --dataset Hvideo_E --model m3ssr --ablation uniform+nogcn --seed 42
```

### Table 6 / Fig. 5 (slot-specialisation diagnostics)

```bash
python analyze_slots.py --dataset Hvideo_E --ablation nogcn \
  --ckpt runs/Hvideo_E_m3ssr_d64_s42_nogcn.best.pt --out analysis/Hvideo_E_full_ng
python analyze_slots.py --dataset Hvideo_E --ablation uniform+nogcn \
  --ckpt runs/Hvideo_E_m3ssr_d64_s42_uniform+nogcn.best.pt --out analysis/Hvideo_E_uniform_ng
python sig_test.py analysis/Hvideo_E_full_ng.npz analysis/Hvideo_E_uniform_ng.npz
```

Each run writes `runs/<tag>.json` (training curves), `.best.pt` (best checkpoint) and `.final.json` (test metrics of the best checkpoint).
