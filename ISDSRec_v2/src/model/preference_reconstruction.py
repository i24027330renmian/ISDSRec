import torch
from torch import nn

class PreferenceReconstruction(nn.Module):
    def __init__(self,latent_dim=128,behavior_mean=None,behavior_std=None):
        super().__init__(); self.attn_proj=nn.Linear(latent_dim,latent_dim); self.attn_vector=nn.Linear(latent_dim,1,bias=False)
        self.temporal_bias=nn.Parameter(torch.tensor(0.0)); self.stat_encoder=nn.Sequential(nn.Linear(6,latent_dim),nn.GELU()); self.fusion_gate=nn.Linear(latent_dim,1)
        mean=torch.zeros(5) if behavior_mean is None else torch.tensor(behavior_mean,dtype=torch.float)
        std=torch.ones(5) if behavior_std is None else torch.tensor(behavior_std,dtype=torch.float)
        self.register_buffer('behavior_mean',mean); self.register_buffer('behavior_std',std.clamp_min(1e-8))
    def build_behavior(self,valid_mask,raw_ts,recency_prior):
        vals=[]
        for b in range(valid_mask.size(0)):
            idx=torch.where(valid_mask[b])[0]; n=len(idx)
            if n==0: vals.append(torch.zeros(6,device=valid_mask.device)); continue
            t=raw_ts[b,idx]; span=(t[-1]-t[0]).clamp_min(0.)/86400. if n>1 else torch.tensor(0.,device=t.device)
            if n>1: mg=((t[1:]-t[:-1]).clamp_min(0.)/86400.).mean(); gv=torch.tensor(1.,device=t.device)
            else: mg=torch.tensor(0.,device=t.device); gv=torch.tensor(0.,device=t.device)
            rr=recency_prior[b,idx]; rm=rr.mean(); rs=torch.sqrt(((rr-rm)**2).mean())
            vals.append(torch.stack([torch.log1p(torch.tensor(float(n),device=t.device)),torch.log1p(span),mg,gv,rm,rs]))
        return torch.stack(vals)
    def standardize_behavior(self,q):
        cont=q[:,[0,1,2,4,5]]; cont=(cont-self.behavior_mean)/self.behavior_std
        return torch.stack([cont[:,0],cont[:,1],cont[:,2],q[:,3],cont[:,3],cont[:,4]],dim=1)
    def forward(self,recovered,recency_prior,valid_mask,raw_ts,use_attention_bias=True,use_gate_prior_stats=True):
        logits=self.attn_vector(torch.tanh(self.attn_proj(recovered))).squeeze(-1)
        if use_attention_bias: logits=logits+self.temporal_bias*recency_prior
        logits=logits.masked_fill(~valid_mask,float('-inf')); weights=torch.softmax(logits,dim=1)
        long=(weights.unsqueeze(-1)*recovered).sum(dim=1); short=recovered[:,-1,:]
        q=self.build_behavior(valid_mask,raw_ts,recency_prior)
        if not use_gate_prior_stats: q=q.clone(); q[:,4:6]=0.
        rho=torch.sigmoid(self.fusion_gate(self.stat_encoder(self.standardize_behavior(q))))
        return rho*long+(1-rho)*short,{'attention':weights,'fusion_gate':rho.squeeze(-1),'behavior':q}
