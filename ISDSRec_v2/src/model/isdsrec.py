import torch
from torch import nn
import torch.nn.functional as F
from .semantic_fusion import GatedSemanticCollaborativeFusion
from .temporal_encoding import TemporalInputEncoder,recency_prior
from .diffusion_schedule import InteractionSpecificDiffusion
from .denoiser import DiffusionDenoiser
from .preference_reconstruction import PreferenceReconstruction

class ISDSRec(nn.Module):
    def __init__(self,num_items,cached_semantics,stats,cfg,ablation=None):
        super().__init__(); self.cfg=cfg; self.stats=stats; self.ablation=ablation or {}; m=cfg['model']; d=int(m.get('latent_dim',128))
        self.semantic=GatedSemanticCollaborativeFusion(num_items,int(cached_semantics.shape[1]),d); self.temporal=TemporalInputEncoder(int(m.get('max_seq_len',50)),d)
        self.diffusion=InteractionSpecificDiffusion(m.get('diffusion_steps',100),m.get('beta_start',1e-4),m.get('beta_end',2e-2),m.get('beta_up',0.05),m.get('lambda_noise',0.5))
        self.denoiser=DiffusionDenoiser(m.get('diffusion_steps',100),d,2,4,512,0.1)
        self.preference=PreferenceReconstruction(d,stats.get('behavior_continuous_mean',[0]*5),stats.get('behavior_continuous_std',[1]*5))
        self.register_buffer('cached_semantics',cached_semantics.float(),persistent=False); self.gamma=float(m.get('gamma',0.8)); self.lambda_diff=float(m.get('lambda_diff',0.4)); self.T=int(m.get('diffusion_steps',100)); self.t_inf=int(m.get('t_inf',40))
    def item_repr(self,item_ids): return self.semantic(item_ids,self.cached_semantics)
    def _ablation_id(self): return self.ablation.get('ablation_id','full_isdsrec')
    def _flags(self):
        ov=self.ablation.get('override',self.ablation); sched=ov.get('scheduling',{}); rec=ov.get('recency_prior_usage',{})
        return {'schedule_mode':sched.get('mode','interaction_specific'),'scheduler_transform':sched.get('scheduler_prior_transform'),
                'prior_source':sched.get('prior_source','elapsed_time_recency'),'use_denoiser_recency':bool(rec.get('denoiser_conditioning',True)),
                'use_attention_bias':bool(rec.get('preference_attention_bias',True)),'use_gate_prior_stats':bool(rec.get('gate_prior_mean',True) and rec.get('gate_prior_std',True)),
                'diffusion_enabled':ov.get('diffusion',{}).get('enabled',True)}
    def _scheduler_prior(self,r,valid):
        f=self._flags(); tr=f['scheduler_transform']
        if f['prior_source']=='normalized_distance_to_final_position':
            raise NotImplementedError('Position-Based Schedule Error')
        if tr=='random_permutation':
            out=r.clone()
            for b in range(r.size(0)):
                idx=torch.where(valid[b])[0]; out[b,idx]=r[b,idx[torch.randperm(len(idx),device=r.device)]]
            return out
        if tr=='reverse_positional_order':
            out=r.clone()
            for b in range(r.size(0)):
                idx=torch.where(valid[b])[0]; out[b,idx]=torch.flip(r[b,idx],[0])
            return out
        return r
    def encode_history(self,item_ids,temporal,steps=None,force_noise=None):
        valid=temporal['valid_mask']; x0=self.temporal(self.item_repr(item_ids),temporal['standardized_gap'],temporal['true_boundary'],valid)
        r=recency_prior(temporal['normalized_log_recency'],self.gamma)*valid; f=self._flags(); aid=self._ablation_id()
        eps=torch.zeros_like(x0); eps_hat=torch.zeros_like(x0); recon_loss=x0.new_tensor(0.)
        if aid=='clean_input_transformer':
            if steps is None: steps=torch.zeros(item_ids.size(0),dtype=torch.long,device=item_ids.device)
            recovered=self.denoiser.reconstruct_state(x0,steps,r,valid,use_step=False,use_recency=True); abar=torch.ones_like(r)
        elif aid=='heteroscedastic_dae':
            ov=self.ablation.get('override',{}); weight=ov.get('losses',{}).get('reconstruction_mse_weight','NOT_STATED_IN_MANUSCRIPT')
            if isinstance(weight,str): raise NotImplementedError('Heteroscedastic DAE reconstruction-MSE weight Error.')
            if steps is None: steps=torch.full((item_ids.size(0),),self.t_inf,dtype=torch.long,device=item_ids.device)
            abar=self.diffusion.alpha_bar(r,steps,valid,'interaction_specific'); eps=torch.randn_like(x0) if force_noise is None else force_noise
            xt=self.diffusion.diffuse(x0,abar,eps,valid); recovered=self.denoiser.reconstruct_state(xt,steps,r,valid,use_step=False,use_recency=True)
            denom=(valid.unsqueeze(-1).sum()*x0.size(-1)).clamp_min(1); recon_loss=(((recovered-x0)**2)*valid.unsqueeze(-1)).sum()/denom
        elif not f['diffusion_enabled']:
            recovered=x0; abar=torch.ones_like(r)
        else:
            if steps is None: steps=torch.randint(1,self.T+1,(item_ids.size(0),),device=item_ids.device)
            sr=self._scheduler_prior(r,valid); abar=self.diffusion.alpha_bar(sr,steps,valid,mode=f['schedule_mode']); eps=torch.randn_like(x0) if force_noise is None else force_noise
            xt=self.diffusion.diffuse(x0,abar,eps,valid); eps_hat=self.denoiser(xt,steps,r,valid,True,f['use_denoiser_recency']); recovered=self.diffusion.recover(xt,abar,eps_hat,valid)
        user,aux=self.preference(recovered,r,valid,temporal['raw_timestamps'],f['use_attention_bias'],f['use_gate_prior_stats'])
        return user,dict(x0=x0,recovered=recovered,noise=eps,predicted_noise=eps_hat,alpha_bar=abar,recency_prior=r,reconstruction_loss=recon_loss,**aux)
    def forward(self,batch,steps=None):
        if self._ablation_id()=='temporal_gte_sasrec': raise NotImplementedError('Temporal-GTE SASRec Error')
        user,aux=self.encode_history(batch['history_items'],batch['temporal'],steps); cand=self.item_repr(batch['candidate_items']); logits=(cand*user.unsqueeze(1)).sum(-1)
        rec=F.cross_entropy(logits,torch.zeros(logits.size(0),dtype=torch.long,device=logits.device)); valid=batch['temporal']['valid_mask'].unsqueeze(-1); denom=(valid.sum()*aux['noise'].size(-1)).clamp_min(1)
        diff=(((aux['noise']-aux['predicted_noise'])**2)*valid).sum()/denom
        if self._ablation_id()=='clean_input_transformer': total=rec; diff=diff*0
        elif self._ablation_id()=='heteroscedastic_dae':
            weight=float(self.ablation['override']['losses']['reconstruction_mse_weight']); total=rec+weight*aux['reconstruction_loss']; diff=diff*0
        else: total=rec+self.lambda_diff*diff
        return {'loss':total,'recommendation_loss':rec,'diffusion_loss':diff,'reconstruction_loss':aux['reconstruction_loss'],'logits':logits,'aux':aux}
    @torch.no_grad()
    def user_representation(self,batch,t_inf=None,noise=None):
        if self._ablation_id() in {'clean_input_transformer','heteroscedastic_dae'}: return self.encode_history(batch['history_items'],batch['temporal'],steps=None,force_noise=noise)
        t=self.t_inf if t_inf is None else int(t_inf); B=batch['history_items'].size(0)
        if t==0:
            saved=self.ablation; ov=dict(saved.get('override',saved)); ov['diffusion']={'enabled':False}; self.ablation={'override':ov}
            u,a=self.encode_history(batch['history_items'],batch['temporal'],torch.zeros(B,dtype=torch.long,device=batch['history_items'].device)); self.ablation=saved; return u,a
        return self.encode_history(batch['history_items'],batch['temporal'],torch.full((B,),t,dtype=torch.long,device=batch['history_items'].device),noise)
    @torch.no_grad()
    def all_item_representations(self): return self.semantic.all_item_representations(self.cached_semantics)
