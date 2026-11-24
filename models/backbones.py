"""SASRec baseline (PyTorch), interface shared by all models in this framework.

All models must implement:
    __init__(n_items, d_model, max_len, **kw)
    forward(seq) -> logits over items   # seq: LongTensor [B, L], 0 = PAD,
                                        # item indices are 1..n_items
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseRec(nn.Module):
    def __init__(self, n_items, d_model=64, max_len=30, dropout=0.2):
        super().__init__()
        self.n_items = n_items
        self.d_model = d_model
        self.max_len = max_len
        # +1: index 0 reserved for PAD
        self.item_emb = nn.Embedding(n_items + 1, d_model, padding_idx=0)
        nn.init.xavier_uniform_(self.item_emb.weight[1:])
        self.dropout = nn.Dropout(dropout)

    def predict_last(self, seq_emb, seq):
        """seq_emb: [B, L, D] -> use last non-pad position, score against item table.
        Sequences are LEFT-padded, so the last real token is always at index L-1."""
        last = seq_emb[:, -1, :]                              # [B, D]
        return last @ self.item_emb.weight[1:].T              # [B, n_items]


class SASRec(BaseRec):
    def __init__(self, n_items, d_model=64, max_len=30, n_layers=2, n_heads=2, dropout=0.2, **kw):
        super().__init__(n_items, d_model, max_len, dropout)
        self.pos_emb = nn.Embedding(max_len, d_model)
        nn.init.trunc_normal_(self.pos_emb.weight, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)

    def encode(self, seq):
        B, L = seq.shape
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.dropout(self.item_emb(seq) + self.pos_emb(pos))
        # Combined causal + pad-key mask. A pad query row whose keys are ALL
        # masked would produce softmax(-inf) = NaN, which then propagates
        # everywhere via 0 * NaN in the next layer -> unmask those rows
        # (their outputs are never read: left-padding => last pos is real).
        pad = (seq == 0)                                       # True = PAD
        causal = torch.full((L, L), float("-inf"), device=seq.device).triu(1)
        m = causal.unsqueeze(0).expand(B, L, L).clone()
        m.masked_fill_(pad.unsqueeze(1), float("-inf"))        # mask pad keys
        fully_masked = torch.isinf(m).all(dim=-1)
        m[fully_masked] = 0.0
        nh = self.encoder.layers[0].self_attn.num_heads
        m = m.repeat_interleave(nh, dim=0)                     # [B*nh, L, L]
        x = self.encoder(x, mask=m)
        return self.final_norm(x)

    def full_logits(self, seq):
        """[B, L, n_items] logits at every position (dense supervision)."""
        return self.encode(seq) @ self.item_emb.weight[1:].T

    def forward(self, seq, acc=None):
        return self.predict_last(self.encode(seq), seq)


class MambaRec(BaseRec):
    """Pure Mamba encoder baseline (Mamba4Rec-style), used for the
    'w/o frequency, w/o disentangle' ablation and efficiency curves."""
    def __init__(self, n_items, d_model=64, max_len=30, n_layers=2, dropout=0.2, **kw):
        super().__init__(n_items, d_model, max_len, dropout)
        from mamba_ssm import Mamba
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)

    def encode(self, seq):
        x = self.dropout(self.item_emb(seq))
        for mamba, norm in zip(self.layers, self.norms):
            x = norm(x + mamba(x))
        return self.final_norm(x)

    def full_logits(self, seq):
        return self.encode(seq) @ self.item_emb.weight[1:].T

    def forward(self, seq, acc=None):
        return self.predict_last(self.encode(seq), seq)


from models.disenreason import DisenReason
from models.m3ssr import M3SSR

MODELS = {"sasrec": SASRec, "mambarec": MambaRec, "disenreason": DisenReason,
          "m3ssr": M3SSR}


def build_model(name, n_items, **kw):
    return MODELS[name](n_items, **kw)
