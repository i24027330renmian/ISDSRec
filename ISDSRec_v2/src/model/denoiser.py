import torch
from torch import nn

class DiffusionDenoiser(nn.Module):
    def __init__(self,T=100,latent_dim=128,num_layers=2,heads=4,ff_dim=512,dropout=0.1):
        super().__init__(); self.step_embedding=nn.Embedding(T+1,latent_dim); self.recency_projection=nn.Linear(1,latent_dim)
        layer=nn.TransformerEncoderLayer(d_model=latent_dim,nhead=heads,dim_feedforward=ff_dim,dropout=dropout,activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,num_layers=num_layers); self.noise_head=nn.Linear(latent_dim,latent_dim); self.state_head=nn.Linear(latent_dim,latent_dim)
    def encode(self,x,steps,recency_prior,valid_mask,use_step=True,use_recency=True):
        h=x
        if use_step: h=h+self.step_embedding(steps).unsqueeze(1)
        if use_recency: h=h+self.recency_projection(recency_prior.unsqueeze(-1))
        h=self.encoder(h,src_key_padding_mask=~valid_mask); return h*valid_mask.unsqueeze(-1)
    def forward(self,x,steps,recency_prior,valid_mask,use_step=True,use_recency=True):
        h=self.encode(x,steps,recency_prior,valid_mask,use_step,use_recency); return self.noise_head(h)*valid_mask.unsqueeze(-1)
    def reconstruct_state(self,x,steps,recency_prior,valid_mask,use_step=False,use_recency=True):
        h=self.encode(x,steps,recency_prior,valid_mask,use_step,use_recency); return self.state_head(h)*valid_mask.unsqueeze(-1)
