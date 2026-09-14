import torch

class InteractionSpecificDiffusion:
    def __init__(self, T=100, beta_start=1e-4, beta_end=2e-2, beta_up=0.05, lambda_noise=0.5):
        self.T=int(T); self.beta_start=float(beta_start); self.beta_end=float(beta_end)
        self.beta_up=float(beta_up); self.lambda_noise=float(lambda_noise)

    def base_betas(self, device):
        return torch.linspace(self.beta_start, self.beta_end, self.T, device=device)

    def alpha_bar(self, recency_prior, steps, valid_mask=None, mode='interaction_specific'):
        """Return position-specific cumulative signal retention at requested steps.

        Eq. 17: beta_eff = beta_t * (1 + lambda_noise * (1-r)); clip at beta_up.
        `steps` is [B] with values 1..T.
        """
        B,L = recency_prior.shape
        betas = self.base_betas(recency_prior.device).view(1,self.T,1)
        r = recency_prior.unsqueeze(1)
        eff = torch.clamp(betas * (1.0 + self.lambda_noise*(1.0-r)), max=self.beta_up)
        abar_all = torch.cumprod(1.0-eff, dim=1)  # [B,T,L] via broadcast
        idx = (steps.long()-1).clamp(0,self.T-1).view(B,1,1).expand(B,1,L)
        abar = abar_all.expand(B,-1,-1).gather(1,idx).squeeze(1)
        if mode == 'noise_matched_uniform':
            if valid_mask is None: mean = abar.mean(dim=1, keepdim=True)
            else:
                denom=valid_mask.sum(dim=1,keepdim=True).clamp_min(1)
                mean=(abar*valid_mask).sum(dim=1,keepdim=True)/denom
            abar=mean.expand_as(abar)
        return abar.clamp(1e-8, 1.0)

    @staticmethod
    def diffuse(x0, alpha_bar, noise, valid_mask):
        a=torch.sqrt(alpha_bar).unsqueeze(-1)
        b=torch.sqrt(1.0-alpha_bar).unsqueeze(-1)
        xt=a*x0+b*noise
        return xt*valid_mask.unsqueeze(-1)

    @staticmethod
    def recover(xt, alpha_bar, predicted_noise, valid_mask):
        a=torch.sqrt(alpha_bar).unsqueeze(-1).clamp_min(1e-8)
        b=torch.sqrt(1.0-alpha_bar).unsqueeze(-1)
        x0=(xt-b*predicted_noise)/a
        return x0*valid_mask.unsqueeze(-1)
