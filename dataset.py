"""Data loading for Shared-account Sequential Recommendation (SSR).

Data format (from LightGC2N / pi-Net protocol):
    train_data.txt / test_data.txt : each line = account_id \t item1 item2 item3 ...
    item_dict.txt                  : each line = item_index \t raw_item_id

Protocol (identical to DisenReason / LightGC2N):
    - train_data.txt provides the input sequence of each account
    - test_data.txt provides (history, target) pairs; we rank the target
      against ALL items (full ranking), metrics: Recall / NDCG / MRR @ 5,10,20
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def load_item_dict(path):
    """Return dict raw_id -> idx (0-based)."""
    m = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                idx, raw = int(parts[0]), parts[1]
                m[raw] = idx
    return m


def load_sequences(path, item_map, max_len):
    """Return list of (account_id, item_idx_list)."""
    seqs = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            acc, items = parts[0], parts[1:]
            idxs = [item_map[i] for i in items if i in item_map]
            if len(idxs) == 0:
                continue
            seqs.append((acc, idxs[-max_len:]))
    return seqs


class SSRData:
    """Protocol per DisenReason Sec.6.1.1:
    - sequences are pre-split 80/20 into train_data.txt / test_data.txt
    - for every sequence, last item = target, rest = input
    - max sequence length 50
    """
    def __init__(self, root, name, max_len=50, test_hist=False):
        self.name = name
        self.max_len = max_len
        d = os.path.join(root, name)
        self.item_map = load_item_dict(os.path.join(d, "item_dict.txt"))
        self.n_items = len(self.item_map)

        raw_train = load_sequences(os.path.join(d, "train_data.txt"), self.item_map, max_len + 1)
        raw_test = load_sequences(os.path.join(d, "test_data.txt"), self.item_map, max_len + 1)

        # account index mapping (needed by account-aware models, e.g. DisenReason)
        accs = sorted({a for a, _ in raw_train} | {a for a, _ in raw_test})
        self.acc_map = {a: i for i, a in enumerate(accs)}
        self.n_accounts = len(accs)

        # train samples: (account_idx, input_seq, target)
        self.train_samples = []
        for a, s in raw_train:
            if len(s) < 2:
                continue
            self.train_samples.append((self.acc_map[a], s[:-1], s[-1]))

        # test samples: same protocol, input = test_seq[:-1], target = last
        # test_hist=True: prepend the account's TRAIN history to the test input
        # (LightGC2N/pi-Net evaluation feeds the full observed history)
        train_hist = {}
        if test_hist:
            for a, s in raw_train:
                train_hist[self.acc_map[a]] = s
        self.test_samples = []
        for a, s in raw_test:
            if len(s) < 2:
                continue
            ai = self.acc_map[a]
            inp = s[:-1]
            if test_hist:
                inp = (train_hist.get(ai, []) + inp)[-max_len:]
            self.test_samples.append((ai, inp, s[-1]))

        # account-item interaction pairs (for LightGCN graph)
        self.interactions = []
        for a, s in raw_train:
            ai = self.acc_map[a]
            self.interactions.extend((ai, it) for it in set(s))

    def stats(self):
        return dict(dataset=self.name, accounts=self.n_accounts,
                    items=self.n_items, train=len(self.train_samples),
                    test=len(self.test_samples),
                    interactions=len(self.interactions))


class TrainDataset(Dataset):
    """Yields (input_seq_padded, target, account_idx). Item indices +1, PAD=0."""
    def __init__(self, samples, max_len):
        self.samples = samples
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        acc, seq, tgt = self.samples[i]
        seq = seq[-(self.max_len - 1):]
        pad = (self.max_len - 1) - len(seq)
        x = [0] * pad + [s + 1 for s in seq]
        return (torch.tensor(x, dtype=torch.long),
                torch.tensor(tgt + 1, dtype=torch.long),
                torch.tensor(acc, dtype=torch.long))


class DenseTrainDataset(Dataset):
    """SASRec-style dense supervision: predict next item at EVERY position.
    input = s[:-1] (left-padded), target = s[1:] (left-padded), PAD=0."""
    def __init__(self, samples, max_len):
        self.samples = samples
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        # samples store (acc, input_seq, target); full = input_seq + [target]
        acc, seq, tgt = self.samples[i]
        s = [x + 1 for x in seq] + [tgt + 1]
        s = s[-self.max_len:]
        x, y = s[:-1], s[1:]
        pad = (self.max_len - 1) - len(x)
        x = [0] * pad + x
        y = [0] * pad + y
        return (torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long),
                torch.tensor(acc, dtype=torch.long))


class EvalDataset(Dataset):
    def __init__(self, samples, max_len):
        self.samples = samples
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        acc, seq, tgt = self.samples[i]
        seq = seq[-(self.max_len - 1):]
        pad = (self.max_len - 1) - len(seq)
        x = [0] * pad + [s + 1 for s in seq]
        return (torch.tensor(x, dtype=torch.long),
                torch.tensor(tgt + 1, dtype=torch.long),
                torch.tensor(acc, dtype=torch.long))


def build_loaders(root, name, max_len=50, batch_size=256, num_workers=2, dense=False,
                  test_hist=False):
    data = SSRData(root, name, max_len, test_hist=test_hist)
    tr_cls = DenseTrainDataset if dense else TrainDataset
    tr = tr_cls(data.train_samples, max_len)
    te = EvalDataset(data.test_samples, max_len)
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, drop_last=True)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers)
    return data, tr_loader, te_loader
