"""Lightweight spatio-temporal forecasters (PyTorch)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adj(A: torch.Tensor) -> torch.Tensor:
    """Symmetric normalization of A+I."""
    A = A + torch.eye(A.size(0), device=A.device, dtype=A.dtype)
    deg = A.sum(-1).clamp(min=1e-6)
    D_inv_sqrt = torch.diag(deg.pow(-0.5))
    return D_inv_sqrt @ A @ D_inv_sqrt


class TemporalEncoder(nn.Module):
    def __init__(self, in_steps: int = 8, hidden: int = 24):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.proj = nn.Linear(in_steps * hidden, hidden)

    def forward(self, x):
        # x: B, T, N, 1
        B, T, N, _ = x.shape
        h = x.permute(0, 2, 3, 1).reshape(B * N, 1, T)  # BN,1,T
        h = self.conv(h)  # BN,H,T
        h = h.reshape(B * N, -1)
        h = self.proj(h)  # BN,H
        return h.view(B, N, -1)


class ForecastHead(nn.Module):
    def __init__(self, hidden: int = 24, out_steps: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_steps),
        )

    def forward(self, h):
        # h: B,N,H -> B, out, N, 1
        y = self.mlp(h)  # B,N,out
        return y.permute(0, 2, 1).unsqueeze(-1)


class TemporalOnly(nn.Module):
    def __init__(self, in_steps=8, hidden=24, out_steps=4):
        super().__init__()
        self.enc = TemporalEncoder(in_steps, hidden)
        self.head = ForecastHead(hidden, out_steps)

    def forward(self, x, A_hat=None):
        return self.head(self.enc(x))


class STGNN(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        in_steps=8,
        hidden=24,
        out_steps=4,
        mode: str = "fixed",  # fixed | adaptive | hybrid
        emb_dim: int = 12,
        topk: int = 15,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.mode = mode
        self.topk = topk
        self.n_nodes = n_nodes
        self.enc = TemporalEncoder(in_steps, hidden)
        self.gW = nn.Linear(hidden, hidden)
        self.head = ForecastHead(hidden, out_steps)
        if mode in ("adaptive", "hybrid"):
            self.E1 = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)
            self.E2 = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)
        if mode == "hybrid":
            self.alpha = nn.Parameter(torch.tensor(float(alpha)))

    def adaptive_adj(self):
        A = F.relu(self.E1 @ self.E2.t())
        # top-k sparsify (keep gradients through soft values of selected edges)
        if self.topk is not None and self.topk < A.size(0):
            vals, idx = torch.topk(A, self.topk, dim=-1)
            mask = torch.zeros_like(A)
            mask.scatter_(-1, idx, 1.0)
            A = A * mask
        A = A / A.sum(-1, keepdim=True).clamp(min=1e-6)  # row-stochastic ~ softmax-like
        # closer to assignment: Softmax(ReLU(E1 E2^T)); use softmax on sparsified logits
        return A

    def adaptive_adj_softmax(self):
        logits = F.relu(self.E1 @ self.E2.t())
        if self.topk is not None and self.topk < logits.size(0):
            vals, idx = torch.topk(logits, self.topk, dim=-1)
            mask = torch.full_like(logits, -1e9)
            mask.scatter_(-1, idx, vals)
            logits = mask
        return F.softmax(logits, dim=-1)

    def forward(self, x, A_fixed_hat: torch.Tensor):
        Z = self.enc(x)  # B,N,H
        if self.mode == "fixed":
            A_hat = A_fixed_hat
        elif self.mode == "adaptive":
            A_hat = self.adaptive_adj_softmax()
        else:
            A_ad = self.adaptive_adj_softmax()
            a = torch.sigmoid(self.alpha)
            A_hat = a * A_fixed_hat + (1 - a) * A_ad
        # H = ReLU(A_hat Z W)
        H = torch.matmul(A_hat, Z)
        H = F.relu(self.gW(H))
        return self.head(H)

    def get_learned_adj(self):
        if self.mode == "fixed":
            return None
        return self.adaptive_adj_softmax().detach().cpu()
