import torch
from torch import nn
import torch.nn.functional as F

class SemanticProjection(nn.Module):
    def __init__(self, semantic_dim=3584, latent_dim=128):
        super().__init__()
        self.proj = nn.Linear(semantic_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)
    def forward(self, e_llm):
        return self.norm(self.proj(e_llm))

class GatedSemanticCollaborativeFusion(nn.Module):
    """Dimension-wise gated fusion: g*ID + (1-g)*semantic."""
    def __init__(self, num_items, semantic_dim=3584, latent_dim=128):
        super().__init__()
        self.id_embedding = nn.Embedding(num_items, latent_dim)
        self.semantic_projection = SemanticProjection(semantic_dim, latent_dim)
        self.gate = nn.Linear(2*latent_dim, latent_dim)

    def forward(self, item_ids, cached_semantics):
        e_id = self.id_embedding(item_ids)
        e_sem = cached_semantics[item_ids]
        h_sem = self.semantic_projection(e_sem)
        g = torch.sigmoid(self.gate(torch.cat([e_id, h_sem], dim=-1)))
        return g * e_id + (1.0-g) * h_sem

    def all_item_representations(self, cached_semantics):
        ids = torch.arange(self.id_embedding.num_embeddings, device=cached_semantics.device)
        return self.forward(ids, cached_semantics)
