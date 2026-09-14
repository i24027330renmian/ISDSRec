import math
import torch
from torch import nn
import torch.nn.functional as F

DAY = 86400.0

class GapEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1,64), nn.GELU(), nn.Linear(64,latent_dim))
        self.boundary = nn.Parameter(torch.zeros(latent_dim))
        nn.init.normal_(self.boundary, std=0.02)

    def forward(self, standardized_gap, is_true_boundary):
        out = self.net(standardized_gap.unsqueeze(-1))
        return torch.where(is_true_boundary.unsqueeze(-1), self.boundary.view(1,1,-1), out)

class TemporalInputEncoder(nn.Module):
    def __init__(self, max_seq_len=50, latent_dim=128):
        super().__init__()
        self.gap_encoder = GapEncoder(latent_dim)
        self.position = nn.Embedding(max_seq_len, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, item_repr, standardized_gap, true_boundary, valid_mask):
        B,L,_ = item_repr.shape
        pos = torch.arange(L, device=item_repr.device).view(1,L).expand(B,L)
        gap = self.gap_encoder(standardized_gap, true_boundary)
        x = self.norm(item_repr + gap + self.position(pos))
        return x * valid_mask.unsqueeze(-1)

def recency_prior(normalized_log_recency, gamma):
    return torch.exp(-float(gamma) * normalized_log_recency)

def build_temporal_tensors(histories_ts, max_seq_len, stats, device):
    """Compute temporal variables on full observable prefix before truncation.

    Histories are Python lists. The returned tensors are left padded.
    """
    B = len(histories_ts)
    Lmax = max_seq_len
    gaps = torch.zeros(B,Lmax, device=device)
    rec = torch.zeros(B,Lmax, device=device)
    valid = torch.zeros(B,Lmax, dtype=torch.bool, device=device)
    boundary = torch.zeros(B,Lmax, dtype=torch.bool, device=device)
    raw_gap = torch.zeros(B,Lmax, device=device)
    raw_ts = torch.zeros(B,Lmax, device=device)
    lengths = torch.zeros(B, dtype=torch.long, device=device)

    for b, ts0 in enumerate(histories_ts):
        ts0 = [float(x) for x in ts0]
        n = len(ts0); keep_start=max(0,n-Lmax); kept=ts0[keep_start:]
        l=len(kept); lengths[b]=l; off=Lmax-l
        if l==0: continue
        ref=ts0[-1]
        for k, orig_idx in enumerate(range(keep_start,n)):
            col=off+k; valid[b,col]=True; raw_ts[b,col]=ts0[orig_idx]
            if orig_idx==0:
                boundary[b,col]=True; gap_days=0.0
            else:
                gap_days=max(0.0,(ts0[orig_idx]-ts0[orig_idx-1])/stats.get('day_seconds',DAY))
            rec_days=max(0.0,(ref-ts0[orig_idx])/stats.get('day_seconds',DAY))
            raw_gap[b,col]=gap_days
            gaps[b,col]=(math.log1p(gap_days)-stats['gap_log_mean'])/(stats['gap_log_std']+1e-8)
            rec[b,col]=math.log1p(rec_days)/(stats['recency_log_std']+1e-8)
    return dict(standardized_gap=gaps, normalized_log_recency=rec, valid_mask=valid,
                true_boundary=boundary, raw_gap_days=raw_gap, raw_timestamps=raw_ts, lengths=lengths)
