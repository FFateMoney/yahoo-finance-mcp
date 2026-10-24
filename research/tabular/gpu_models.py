"""F140 MLP and F141 FT-Transformer (Gorishniy et al. 2021) on the same panel and walk-forward as cpu_models.py.
Fixed hyper-parameters from 回测/表格模型_预登记.md. Needs only numpy + torch (CUDA if available).
Inputs: data/panel.npz (or data/panel_X.npz + data/panel_meta.npz). Writes out/pred_mlp.npy, out/pred_ftt.npy.
Usage: python research/tabular/gpu_models.py [mlp|ftt ...]      (default: both)
"""
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from wf import daily_ic, folds, load_panel, save_pred

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
P = load_panel()
X, y, di, sec = P["Xf"], P["y"], P["di"], P["sec"].astype(np.int64)
NF, NS = X.shape[1], int(sec.max()) + 1
MISS = np.isnan(X)
XV = np.where(MISS, 0.0, X - 0.5).astype(np.float32)  # centred percentile, 0 when missing

CFG = {
    "mlp": dict(seeds=5, lr=1e-3, wd=1e-4, batch=4096, epochs=20, patience=3),
    "ftt": dict(seeds=3, lr=3e-4, wd=1e-5, batch=2048, epochs=15, patience=3),
}


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(NS, 8)
        layers, d = [], 2 * NF + 8
        for h in (256, 128, 64):
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.2)]
            d = h
        self.net = nn.Sequential(*layers, nn.Linear(d, 1))

    def forward(self, x, m, s):
        return self.net(torch.cat([x, m.float(), self.emb(s)], 1)).squeeze(1)


class FTT(nn.Module):
    """Feature tokenizer (per-feature linear token, learned token when missing) + sector token + CLS, 3 transformer blocks."""
    def __init__(self, d=64, layers=3, heads=8, drop=0.1):
        super().__init__()
        self.w = nn.Parameter(torch.randn(NF, d) * 0.1)
        self.b = nn.Parameter(torch.zeros(NF, d))
        self.miss = nn.Parameter(torch.zeros(NF, d))
        self.sec = nn.Embedding(NS, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        blk = nn.TransformerEncoderLayer(d, heads, 4 * d // 3 * 2, drop, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(blk, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, x, m, s):
        tok = x.unsqueeze(2) * self.w + self.b
        tok = torch.where(m.unsqueeze(2), self.miss.expand_as(tok), tok)
        z = torch.cat([self.cls.expand(len(x), -1, -1), tok, self.sec(s).unsqueeze(1)], 1)
        return self.head(self.enc(z)[:, 0]).squeeze(1)


def tensors(rows):
    return (torch.from_numpy(XV[rows]).to(DEV), torch.from_numpy(MISS[rows]).to(DEV),
            torch.from_numpy(sec[rows]).to(DEV), torch.from_numpy(y[rows]).to(DEV))


@torch.no_grad()
def predict(model, T, batch=16384):
    model.eval()
    out = []
    for i in range(0, len(T[0]), batch):
        with torch.autocast(DEV.type, enabled=DEV.type == "cuda"):
            out.append(model(T[0][i:i + batch], T[1][i:i + batch], T[2][i:i + batch]).float())
    return torch.cat(out).cpu().numpy()


def fit(kind, seed, tr, va):
    c = CFG[kind]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = (MLP() if kind == "mlp" else FTT()).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["wd"])
    scaler = torch.amp.GradScaler(enabled=DEV.type == "cuda")
    Ttr, Tva = tensors(tr), tensors(va)
    best, best_state, bad = np.inf, None, 0
    for ep in range(c["epochs"]):
        model.train()
        perm = torch.randperm(len(tr), device=DEV)
        for i in range(0, len(tr), c["batch"]):
            b = perm[i:i + c["batch"]]
            with torch.autocast(DEV.type, enabled=DEV.type == "cuda"):
                loss = ((model(Ttr[0][b], Ttr[1][b], Ttr[2][b]).float() - Ttr[3][b]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        vl = float(((predict(model, Tva) - y[va]) ** 2).mean())
        if vl < best - 1e-7:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= c["patience"]:
                break
    model.load_state_dict(best_state)
    return model, ep + 1, best


def run(kind):
    pred = np.full(len(y), np.nan)
    for tr, va, pr in folds(P):
        t, ps, info = time.time(), [], []
        Tpr = tensors(pr)
        for sd in range(CFG[kind]["seeds"]):
            m, ep, vl = fit(kind, sd + 1, tr, va)
            p = predict(m, Tpr)
            ps.append((p - p.mean()) / (p.std() + 1e-12))
            info.append(f"{ep}ep/{vl:.5f}")
        pred[pr] = np.mean(ps, 0)
        ic = daily_ic(pred[pr], y[pr], di[pr]).mean()
        print(f"{kind} fold {P['dates'][di[pr[0]]]} train {len(tr)} [{' '.join(info)}] block IC {ic:+.4f} {time.time() - t:.0f}s", flush=True)
        save_pred(kind, pred)  # partial results survive an interruption
    print(kind, "done")


if __name__ == "__main__":
    print("device", DEV, torch.cuda.get_device_name(0) if DEV.type == "cuda" else "")
    for k in sys.argv[1:] or ["mlp", "ftt"]:
        run(k)
