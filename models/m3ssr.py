"""M3-SSR: Adaptive Wavelet-Mamba with Latent User Disentanglement
for Shared-account Sequential Recommendation.

Pipeline (drop-in replacement of DisenReason, same I/O contract):
  1. LightGCN collaborative embeddings (shared account-item graph, reused
     from models.disenreason for a fair comparison)
  2. Multi-scale adaptive wavelet decomposition (dyadic Haar DWT along the
     sequence axis) -> per-scale sub-sequences -> learnable softmax fusion.
     Replaces DisenReason's FIXED equal-width rFFT bands (its own
     acknowledged limitation) with a multi-resolution dyadic filter bank.
  3. Mamba backbone over the fused sequence (long-range selective SSM,
     replacing the shallow progressive-residual reasoning stage).
  4. Latent user disentanglement: K learnable slot queries attend over the
     Mamba outputs -> K latent-user representations, gated by the account
     embedding h_a.
  5. Prediction head (same as DisenReason): Linear(2d, n_items) over
     [h_final ; h_a]; loss = L_rec + beta * mean_slot_CE (L_aux analogue).

Inputs follow framework convention: seq LongTensor [B, L] (0 = PAD,
items 1..n, LEFT-padded -> last real token at index -1), acc LongTensor [B].
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.disenreason import LightGCN, FreqDisentangle


# --------------------------------------------------------------------------
# Mamba backbone: use the official CUDA kernel when available (AutoDL);
# fall back to a compact pure-PyTorch selective scan (CPU smoke tests only).
# --------------------------------------------------------------------------
try:
    from mamba_ssm import Mamba as _CudaMamba
    _HAS_MAMBA = True
except Exception:
    _HAS_MAMBA = False


class _TorchMamba(nn.Module):
    """Minimal pure-PyTorch selective-scan SSM (Mamba-like), CPU fallback.
    Same interface as mamba_ssm.Mamba: forward(x[B, L, D]) -> [B, L, D]."""
    def __init__(self, d_model, d_state=16, expand=2, **kw):
        super().__init__()
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner)
        self.A = nn.Parameter(torch.randn(self.d_inner, d_state) * 0.1)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        B, L, _ = x.shape
        xz = self.in_proj(x)
        xi, z = xz.chunk(2, dim=-1)                      # [B, L, I]
        xi = F.silu(xi)
        bc_d = self.x_proj(xi)                           # [B, L, 2S+1]
        Bm = bc_d[..., :self.d_state]
        Cm = bc_d[..., self.d_state:2 * self.d_state]
        dt = F.softplus(self.dt_proj(bc_d[..., -1:]))    # [B, L, I]
        A = -F.softplus(self.A)                          # [I, S]
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        ys = []
        for t in range(L):
            dA = torch.exp(dt[:, t, :, None] * A)        # [B, I, S]
            dB = dt[:, t, :, None] * Bm[:, t, None, :]   # [B, I, S]
            h = dA * h + dB * xi[:, t, :, None]
            ys.append((h * Cm[:, t, None, :]).sum(-1))   # [B, I]
        y = torch.stack(ys, dim=1) + self.D * xi
        return self.out_proj(y * F.silu(z))


def make_mamba(d_model, **kw):
    if _HAS_MAMBA:
        return _CudaMamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
    return _TorchMamba(d_model)


# --------------------------------------------------------------------------
# Multi-scale adaptive wavelet decomposition (Haar DWT, dyadic filter bank)
# --------------------------------------------------------------------------
class WaveletDisentangle(nn.Module):
    """phi(.) replacement: J-level dyadic Haar DWT along the sequence axis ->
    J+1 sub-band sequences (cA_J, cD_J ... cD_1) reconstructed to length L ->
    learnable softmax fusion (weights conditioned on the sequence summary).

    Contrast with DisenReason: fixed equal-width rFFT bands -> here bands are
    octave-spaced (multi-resolution), which matches bursty shared-account
    behaviour at multiple time scales.
    """
    def __init__(self, d_model, levels=3):
        super().__init__()
        self.levels = levels
        # V2: learnable 2-tap filters per level, Haar-initialised.
        # h0 = low-pass, h1 = high-pass; reconstruction uses the same
        # filters (QMF-style), so at init this is exactly the Haar DWT.
        r = math.sqrt(0.5)
        self.h0 = nn.ParameterList(
            [nn.Parameter(torch.tensor([r, r])) for _ in range(levels)])
        self.h1 = nn.ParameterList(
            [nn.Parameter(torch.tensor([r, -r])) for _ in range(levels)])
        self.fc = nn.Linear(d_model, levels + 1)         # adaptive fusion

    def _dwt(self, x, lvl):                               # x: [B, L, D], L even
        s = x[:, 0::2] * self.h0[lvl][0] + x[:, 1::2] * self.h0[lvl][1]
        d = x[:, 0::2] * self.h1[lvl][0] + x[:, 1::2] * self.h1[lvl][1]
        return s, d

    def _idwt(self, s, d, n, lvl):                        # -> [B, n, D]
        even = s * self.h0[lvl][0] + d * self.h1[lvl][0]
        odd = s * self.h0[lvl][1] + d * self.h1[lvl][1]
        out = torch.stack([even, odd], dim=2).flatten(1, 2)
        return out[:, :n]

    def _bands(self, x):                                  # x: [B, L, D]
        B, L, D = x.shape
        pad = (2 ** self.levels - L % 2 ** self.levels) % 2 ** self.levels
        if pad:                                           # right-pad by replicating last
            x = torch.cat([x, x[:, -1:].expand(B, pad, D)], dim=1)
        approx, details = x, []
        for lvl in range(self.levels):                    # lvl 0 = finest
            approx, det = self._dwt(approx, lvl)
            details.append(det)                           # details[0]=finest
        # reconstruct each band alone at full resolution (coarse -> fine).
        # band 0 = approximation cA_J; band i (1..J) = details[i-1]
        bands = []
        for i in range(self.levels + 1):
            cur = approx if i == 0 else torch.zeros_like(approx)
            for lvl in range(self.levels):                # coarse -> fine
                idx = self.levels - 1 - lvl               # detail at this step
                d = details[idx] if (i >= 1 and idx == i - 1) \
                    else torch.zeros_like(details[idx])
                cur = self._idwt(cur, d, cur.size(1) * 2, idx)
            bands.append(cur[:, :L])
        return torch.stack(bands, dim=0)                  # [J+1, B, L, D]

    def forward(self, x):                                 # x: [B, L, D]
        bands = self._bands(x)
        w = torch.softmax(self.fc(x.mean(dim=1)), dim=-1)     # [B, J+1]
        fused = (bands * w.t().unsqueeze(-1).unsqueeze(-1)).sum(0)
        return fused


# --------------------------------------------------------------------------
# Latent user disentanglement via slot attention
# --------------------------------------------------------------------------
class LatentUserSlots(nn.Module):
    """K learnable queries cross-attend over the encoded sequence ->
    K latent-user representations (masked mean over pad positions)."""
    def __init__(self, d_model, n_users=4, n_heads=4):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(n_users, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h, key_padding_mask):               # h: [B, L, D]
        B = h.size(0)
        q = self.slots.unsqueeze(0).expand(B, -1, -1)
        u, _ = self.attn(q, h, h, key_padding_mask=key_padding_mask)
        return self.norm(u + q)                           # [B, K, D]


class M3SSR(nn.Module):
    def __init__(self, n_items, d_model=64, max_len=50, n_accounts=None,
                 interactions=None, gcn_layers=3, wavelet_levels=3,
                 n_users=4, n_mamba_layers=2, beta=0.5, dropout=0.2,
                 ablation='none', **kw):
        # ablation: 'none' | 'fft' (fixed-band FFT instead of wavelet) |
        #           'transformer' (Transformer instead of Mamba) |
        #           'noslots' (no latent-user disentanglement) |
        #           'nogcn' (plain learned embeddings instead of LightGCN) |
        #           'uniform' (uniform slot supervision: mean CE over slots
        #                      instead of winner-take-all min) |
        #           'nodiv' (drop the slot-diversity regulariser) |
        #           'nogate' (uniform 1/K slot mixing instead of the
        #                     account-conditioned gate)
        super().__init__()
        assert n_accounts is not None and interactions is not None, \
            "M3SSR needs n_accounts and interactions for LightGCN"
        self.n_items = n_items
        self.beta = beta
        self.gcn = LightGCN(n_accounts, n_items, d_model, gcn_layers)
        self.ablation = ablation
        # ablation may combine flags with '+', e.g. 'fft+nogcn' runs the
        # fft variant on the nogcn (plain-embedding) base model.
        abl = set() if ablation == 'none' else set(ablation.split('+'))
        self.abl = abl
        if 'fft' in abl:
            self.wavelet = FreqDisentangle(d_model, bandwidth=5, n_bands=5)
        else:
            self.wavelet = WaveletDisentangle(d_model, wavelet_levels)
        self.drop = nn.Dropout(dropout)
        if 'transformer' in abl:
            tl = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=2, dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True, norm_first=True)
            self.mambas = nn.ModuleList(
                [nn.TransformerEncoder(tl, num_layers=1) for _ in range(n_mamba_layers)])
        else:
            self.mambas = nn.ModuleList([make_mamba(d_model) for _ in range(n_mamba_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_mamba_layers)])
        self.slots = LatentUserSlots(d_model, n_users)
        if 'nogcn' in abl:
            self.item_emb_plain = nn.Embedding(n_items + 1, d_model, padding_idx=0)
            self.acc_emb_plain = nn.Embedding(n_accounts, d_model)
            nn.init.xavier_uniform_(self.item_emb_plain.weight[1:])
            nn.init.xavier_uniform_(self.acc_emb_plain.weight)
        self.gate = nn.Linear(d_model, n_users)           # account-conditioned mix
        self.final_norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(2 * d_model, n_items)
        self._interactions = interactions
        self._n_accounts = n_accounts

    def setup_graph(self, device):
        self.gcn.build_graph(self._interactions, self._n_accounts,
                             self.n_items, device)

    def encode(self, seq, acc):
        if 'nogcn' in self.abl:
            h_a = self.acc_emb_plain(acc)
            x = self.item_emb_plain(seq)
        else:
            H_A, H_V = self.gcn.propagate()
            h_a = H_A[acc]                                # [B, D]
            x = H_V[seq]                                  # [B, L, D]
        x = self.drop(self.wavelet(x))                    # multi-scale fusion
        pad = (seq == 0)
        for mamba, norm in zip(self.mambas, self.norms):
            if 'transformer' in self.abl:
                L = x.size(1)
                causal = torch.full((L, L), float("-inf"), device=x.device).triu(1)
                m = causal.unsqueeze(0).expand(x.size(0), L, L).clone()
                m.masked_fill_(pad.unsqueeze(1), float("-inf"))
                full = torch.isinf(m).all(dim=-1)
                m[full] = 0.0
                m = m.repeat_interleave(2, dim=0)
                x = norm(x + self.drop(mamba(x, mask=m)))
            else:
                x = norm(x + self.drop(mamba(x)))
        x = self.final_norm(x)
        pivot = x[:, -1, :]                               # left-padded
        if 'noslots' in self.abl:
            return self.drop(pivot), h_a, []
        users = self.slots(x, pad)                        # [B, K, D]
        if 'nogate' in self.abl:
            g = users.new_full((users.size(0), users.size(1)),
                               1.0 / users.size(1))       # uniform mixing
        else:
            g = torch.softmax(self.gate(h_a), dim=-1)     # [B, K]
        u_mix = (users * g.unsqueeze(-1)).sum(1)          # [B, D]
        h_final = self.drop(pivot + u_mix)
        return h_final, h_a, users

    def forward(self, seq, acc=None):
        h_final, h_a, _ = self.encode(seq, acc)
        return self.out(torch.cat([h_final, h_a], dim=-1))

    def compute_loss(self, seq, tgt, acc):
        h_final, h_a, users = self.encode(seq, acc)
        logits = self.out(torch.cat([h_final, h_a], dim=-1))
        l_rec = F.cross_entropy(logits, tgt - 1)
        if len(users) == 0:                               # noslots ablation
            return l_rec
        # V2: Winner-Take-All slot loss -- only the best-predicting slot of
        # each sample is penalised, forcing slots to specialise (per-sample
        # min over slots, then batch mean).
        per_slot = []
        for k in range(users.size(1)):                    # shared output head
            lu = self.out(torch.cat([users[:, k], h_a], dim=-1))
            per_slot.append(F.cross_entropy(lu, tgt - 1, reduction='none'))
        per_slot = torch.stack(per_slot, dim=0)           # [K, B]
        if 'uniform' in self.abl:
            # uniform supervision: every slot is penalised on every sample
            # (the pre-WTA L_aux analogue) -> no pressure to specialise.
            l_aux = per_slot.mean(dim=0).mean()
        else:
            # V2: Winner-Take-All slot loss -- only the best-predicting slot
            # of each sample is penalised, forcing slots to specialise
            # (per-sample min over slots, then batch mean).
            l_aux = per_slot.min(dim=0).values.mean()     # [K,B]->[B]
        loss = l_rec + self.beta * l_aux
        if 'nodiv' in self.abl:
            return loss
        # slot diversity regulariser: penalise pairwise cosine similarity
        # between slot outputs so they cover different latent users.
        u = F.normalize(users, dim=-1)                    # [B, K, D]
        sim = torch.bmm(u, u.transpose(1, 2))             # [B, K, K]
        K = users.size(1)
        off = sim.sum(dim=(1, 2)) - torch.diagonal(sim, dim1=1, dim2=2).sum(-1)
        l_div = off.mean() / (K * (K - 1))
        return loss + 0.1 * l_div
