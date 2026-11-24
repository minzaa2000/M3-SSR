"""DisenReason (arXiv:2603.03782) PyTorch reimplementation.

Faithful to Sec.5 + Algorithm 1:
  Stage 1: LightGCN collaborative embeddings -> sequence lookup -> rFFT ->
           equal-width sub-band partition (bandwidth B) -> per-band zero-pad
           + irFFT -> softmax FC adaptive fusion -> pivot = last hidden state
  Stage 2: progressive residual reasoning with learnable position emb r_p,
           phi reuses the disentanglement operator (degenerates to a linear
           map on a single vector); terminate when cos(u_t, u_{t-1}) > alpha
  Loss: L_rec + beta * L_aux (aux CE per reasoning step, shared output layer)

Inputs follow framework convention: seq LongTensor [B,L] (0=PAD, items 1..n),
acc LongTensor [B] account indices, targets 1..n.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    """LightGCN over the account-item bipartite graph (Eq.4-6)."""
    def __init__(self, n_accounts, n_items, d_model, n_layers=3):
        super().__init__()
        self.n_layers = n_layers
        self.acc_emb = nn.Embedding(n_accounts, d_model)
        self.item_emb = nn.Embedding(n_items + 1, d_model, padding_idx=0)
        nn.init.xavier_uniform_(self.acc_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight[1:])
        self.register_buffer("norm_adj", None, persistent=False)

    def build_graph(self, interactions, n_accounts, n_items, device):
        """interactions: list of (account_idx, item_idx 0-based)."""
        import scipy.sparse as sp
        rows = [a for a, _ in interactions]
        cols = [i for _, i in interactions]
        m, n = n_accounts, n_items
        R = sp.coo_matrix(([1.0] * len(rows), (rows, cols)), shape=(m, n))
        # symmetric normalized adjacency of [[0,R],[R^T,0]]
        adj = sp.bmat([[None, R], [R.T, None]], format="coo")
        deg = torch.tensor(adj.sum(1), dtype=torch.float32).squeeze().clip(min=1.0) ** -0.5
        i = torch.tensor(adj.row, dtype=torch.long)
        j = torch.tensor(adj.col, dtype=torch.long)
        v = torch.tensor(adj.data, dtype=torch.float32)
        v = (v * deg[i] * deg[j]).float()
        sp_adj = torch.sparse_coo_tensor(torch.stack([i, j]), v,
                                         (m + n, m + n)).coalesce().to(device)
        self.norm_adj = sp_adj

    def propagate(self):
        E = torch.cat([self.acc_emb.weight, self.item_emb.weight[1:]], 0)
        out = 0.0
        cur = E
        for _ in range(self.n_layers):
            cur = torch.sparse.mm(self.norm_adj, cur)
            out = out + cur
        out = out / self.n_layers                       # Eq.6 averages layers 1..L
        H_A, H_V = out[: self.acc_emb.num_embeddings], out[self.acc_emb.num_embeddings:]
        H_V = torch.cat([torch.zeros(1, H_V.size(1), device=H_V.device), H_V], 0)
        return H_A, H_V                                 # item table padded at 0


class FreqDisentangle(nn.Module):
    """phi(.) of the paper: rFFT -> Z equal-width sub-bands -> irFFT ->
    softmax-FC weighted fusion (Eq.8-12). Works on [B, L, D] sequences;
    on [B, 1, D] it degenerates to the stage-2 reasoning map."""
    def __init__(self, d_model, bandwidth=5, n_bands=5):
        super().__init__()
        self.B = bandwidth
        self.Z = n_bands
        self.fc = nn.Linear(d_model, n_bands)           # adaptive fusion weights

    def forward(self, x):                               # x: [B, L, D]
        B, L, D = x.shape
        Fspec = torch.fft.rfft(x, dim=1)                # [B, f, D], f = L//2+1
        f = Fspec.size(1)
        bands = []
        for z in range(self.Z):
            lo, hi = z * self.B, min((z + 1) * self.B, f)
            spec = torch.zeros_like(Fspec)
            if lo < f:
                spec[:, lo:hi] = Fspec[:, lo:hi]
            bands.append(torch.fft.irfft(spec, n=L, dim=1))   # [B, L, D]
        bands = torch.stack(bands, dim=0)               # [Z, B, L, D]
        w = torch.softmax(self.fc(x.mean(dim=1)), dim=-1)     # [B, Z]
        fused = (bands * w.t().unsqueeze(-1).unsqueeze(-1)).sum(0)  # [B, L, D]
        return fused


class DisenReason(nn.Module):
    def __init__(self, n_items, d_model=64, max_len=50, n_accounts=None,
                 interactions=None, gcn_layers=3, bandwidth=5, n_bands=5,
                 alpha=0.5, beta=0.5, max_reason_steps=10, **kw):
        super().__init__()
        assert n_accounts is not None and interactions is not None, \
            "DisenReason needs n_accounts and interactions for LightGCN"
        self.n_items = n_items
        self.beta = beta
        self.alpha = alpha
        self.max_steps = max_reason_steps
        self.gcn = LightGCN(n_accounts, n_items, d_model, gcn_layers)
        self.phi = FreqDisentangle(d_model, bandwidth, n_bands)
        self.r_p = nn.Parameter(torch.randn(d_model) * 0.02)
        self.out = nn.Linear(2 * d_model, n_items)      # W_A: [h; h_A] -> items
        self.out_items = n_items
        self._interactions = interactions
        self._n_accounts = n_accounts

    def setup_graph(self, device):
        self.gcn.build_graph(self._interactions, self._n_accounts,
                             self.n_items, device)

    def _pivot_and_users(self, seq, acc):
        H_A, H_V = self.gcn.propagate()
        h_a = H_A[acc]                                   # [B, D]
        H_seq = H_V[seq]                                 # [B, L, D]
        H_bar = self.phi(H_seq)                          # fused sequence
        # sequences are LEFT-padded: last real token is at index -1
        pivot = H_bar[:, -1, :]

        # Stage 2: progressive residual reasoning
        r = pivot + self.r_p
        users, prev = [], None
        for t in range(self.max_steps):
            u = self.phi(r.unsqueeze(1)).squeeze(1)      # degenerate phi on vector
            users.append(u)
            r = r - u                                    # residual
            if prev is not None:
                sim = F.cosine_similarity(u, prev, dim=-1)
                # per-sample early stop is non-differentiable bookkeeping;
                # use batch-wise stop when ALL samples exceed alpha (eval)
                if not self.training and bool((sim > self.alpha).all()):
                    break
            prev = u
        h_final = torch.stack(users).mean(0)
        return h_final, h_a, users

    def forward(self, seq, acc=None):
        h_final, h_a, _ = self._pivot_and_users(seq, acc)
        return self.out(torch.cat([h_final, h_a], dim=-1))     # [B, n_items]

    def compute_loss(self, seq, tgt, acc):
        h_final, h_a, users = self._pivot_and_users(seq, acc)
        logits = self.out(torch.cat([h_final, h_a], dim=-1))
        l_rec = F.cross_entropy(logits, tgt - 1)
        l_aux = 0.0
        for u in users:                                  # Eq.18-19, shared W_u = W_A
            lu = self.out(torch.cat([u, h_a], dim=-1))
            l_aux = l_aux + F.cross_entropy(lu, tgt - 1)
        return l_rec + self.beta * l_aux / len(users)
