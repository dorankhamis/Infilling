import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch import Tensor
from torch.autograd import Variable
from torch.distributions.normal import Normal
from torch.distributions import kl

from architectures.components.mlp import PositionwiseFeedForward
from architectures.components.attention import (MultiHeadedAttention,
                                                MultiHeadedAttention_TwoBatchDim)
from architectures.components.layer_norm import LayerNorm, SublayerConnection
from architectures.components.positional_encoding import PositionalEncoding
from architectures.components.nn_utils import subsequent_mask, clones


EPS = torch.tensor(1e-10)

# epoch = 300
# batch = dg.get_batch(
    # batch_size,
    # const_l=True,
    # batch_type='train',
    # min_gap_logprob=logprob_vargap_seq[epoch],
    # mean_gap_length=gap_length_seq[epoch],
    # gap_length_sd=gap_sd_seq[epoch],
    # shortrange=cfg.shortrange,
    # longrange=cfg.longrange
# )
# batch = send_batch_to_device(batch, device)

# cfg.features_d = 6
# cfg.embed_v = 16
# batch['observed_mask'] = (~batch['our_gaps']) * (~batch['existing_gaps'])

def sample_normal(mu, logsig):
    sample = Normal(mu, torch.exp(logsig)).rsample()
    return sample

def KL_std_norm(mu, logsig):
    # KL = kl.kl_divergence(Normal(mu, torch.exp(logsig)), Normal(0, 1))
    sig_sq = torch.square(torch.exp(logsig))
    KL_div = 0.5 * ( -(torch.log(sig_sq + EPS) + 1).sum() + 
        sig_sq.sum() + 
        torch.square(mu).sum()
    )
    return KL_div
    
def make_recon_loglik(use_non_gaps=True):
    
    mse = nn.MSELoss(reduction='sum')

    def loglik_gap_masked(batch, pred_mu, pred_sigma=None, pred_ensemble=None):
        # define masks 
        fake_gap_mask = (batch['our_gaps'])*(~batch['existing_gaps'])
        nongap_mask = (~batch['our_gaps'])*(~batch['existing_gaps'])
        
        # loss on manufactured gaps
        if not pred_sigma is None:
            neg_loglik = -Normal(pred_mu[fake_gap_mask], pred_sigma[fake_gap_mask]).log_prob(batch['targets'][fake_gap_mask]).sum()
            if not pred_ensemble is None:
                for ee in range(pred_ensemble.shape[1]):
                    neg_loglik += mse(pred_ensemble[:,ee,:,:][fake_gap_mask], batch['targets'][fake_gap_mask])
        else:
            neg_loglik = 3*mse(pred_mu[fake_gap_mask], batch['targets'][fake_gap_mask])
        
        # loss on rest of time series minus real gaps
        if use_non_gaps:
            if not pred_sigma is None:
                neg_loglik += -Normal(pred_mu[nongap_mask], pred_sigma[nongap_mask]).log_prob(batch['targets'][nongap_mask]).sum()
                if not pred_ensemble is None:
                    for ee in range(pred_ensemble.shape[1]):
                        neg_loglik += mse(pred_ensemble[:,ee,:,:][nongap_mask], batch['targets'][nongap_mask])
            else:
                neg_loglik += 3*mse(pred_mu[nongap_mask], batch['targets'][nongap_mask])
                
        return neg_loglik
        
    return loglik_gap_masked
            

class VarEmbed(nn.Module):

    def __init__(self, cfg):
        super(VarEmbed, self).__init__()
        
        self.relu = nn.ReLU()
        self.pre_O = nn.Linear(1 + cfg.features_aux, 2*cfg.embed_v)
        self.pre_M = nn.Linear(cfg.features_aux, 2*cfg.embed_v)        
        self.embed_r_O = nn.Linear(2*cfg.embed_v, cfg.embed_v)
        self.embed_r_M = nn.Linear(2*cfg.embed_v, cfg.embed_v)

    def forward(self, x, aux, mask_O):
        ## concat aux series onto inputs
        xj = torch.cat([x, aux], dim=1) # (B, C, L)
        
        ## embed to latent space
        r_O = self.pre_O(xj.transpose(-2, -1)) # (B, L, C)
        r_O = self.relu(r_O)
        r_O = self.embed_r_O(r_O)
        
        r_M = self.pre_M(aux.transpose(-2, -1)) # (B, L, C)
        r_M = self.relu(r_M)
        r_M = self.embed_r_M(r_M)
        
        ## sum using masks
        r = (r_O * mask_O.transpose(-2, -1) + 
            r_M * ~(mask_O.transpose(-2, -1)))
        return r


class DistParams(nn.Module):

    def __init__(self, cfg):
        super(DistParams, self).__init__()
        self.mu_linear = nn.Linear(cfg.embed_v, cfg.embed_v)
        self.logsig_linear = nn.Linear(cfg.embed_v, cfg.embed_v)
        
    def forward(self, r):
        ## calculate mean and standard deviation from a var embedding r
        return self.mu_linear(r), self.logsig_linear(r) 
        

class Encoder(nn.Module):

    def __init__(self, cfg):
        super(Encoder, self).__init__()
        
        self.v_encs = nn.ModuleList([
            VarEmbed(cfg) for i in range(cfg.features_d)
        ])
        
        self.dist_pars = nn.ModuleList([
            DistParams(cfg) for i in range(cfg.features_d)
        ])
        
        # var attention
        self.sublayer_v = SublayerConnection(cfg.embed_v, cfg.dropout)
        self.var_attn = MultiHeadedAttention_TwoBatchDim(cfg.attn_heads, cfg.embed_v)

        # temporal attention
        self.sublayer_t = SublayerConnection(cfg.embed_v, cfg.dropout)
        self.temp_attn = MultiHeadedAttention(cfg.attn_heads, cfg.embed_v)


    def forward(self, batch):
        
        ## variable embedding
        r_list = [layer(
            batch['inputs'][:,i:(i+1),:],
            batch['aux_inputs'],
            batch['observed_mask'][:,i:(i+1),:]
            ) for i,layer in enumerate(self.v_encs)
        ]
        
        ## attention across variables?        
        r_tens = torch.stack(r_list, dim=-2) # (B, L, V, C)
        r_tens = self.sublayer_v(r_tens, 
            lambda r_tens: self.var_attn(r_tens, r_tens, r_tens, None)
        )
        
        ## create mu, sigma for each time and join variables
        mu_list = []
        logsig_list = []
        for i in range(len(r_list)):
            # do separately for each variable to allow different learning
            mu_i, logsig_i = self.dist_pars[i](r_tens[:,:,i,:])
            mu_list.append(mu_i)
            logsig_list.append(logsig_i)
        
        mu = torch.cat(mu_list, dim=-1) # (B, L, V, C)
        logsig = torch.cat(logsig_list, dim=-1)
        return mu, logsig


class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        
        self.model_size = cfg.features_d * cfg.embed_v
        self.sublayer_t = SublayerConnection(self.model_size, cfg.dropout)
        self.pe = PositionalEncoding(self.model_size, cfg.dropout)
        self.temporal_attn = MultiHeadedAttention(cfg.attn_heads, self.model_size)
        self.decode_state = nn.Linear(self.model_size, cfg.features_d)

    def forward(self, z_sample, mask=None):
        if mask is None:
            ## create causal mask
            nbatches = z_sample.shape[0]
            seq_len = z_sample.shape[1]
            mask = (subsequent_mask(seq_len)
                .expand(nbatches, seq_len, seq_len)
                .to(z_sample.device)
            )
                                  
        ## causal attention across time series on sampled z
        z_sample = self.pe(z_sample)
        z_sample = self.sublayer_t(z_sample, 
            lambda z_sample: self.temporal_attn(z_sample, z_sample, z_sample, mask)
        )
        
        ## possibly attach auxiliary information?
        
        ## output time series x(t)
        x_pred = self.decode_state(z_sample).transpose(-2, -1) # (B, C, L)        
        return x_pred # (B, C, L)
        
    def decode_from_prior(self, seq_len):
        zeros = torch.zeros((seq_len, 1, self.model_size))
        z0_sample = sample_normal(zeros, zeros) # log(1) = 0
        return self.forward(z0_sample)

        
class TemporalEmbedder(nn.Module):
    def __init__(self, cfg):
        super(TemporalEmbedder, self).__init__()
        
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.recon_loss = make_recon_loglik(use_non_gaps=True)
        self.mask = None
        self.model_size = cfg.features_d * cfg.embed_v

    def encode(self, batch):
        z_mu, z_logsig = self.encoder(batch)
        return z_mu, z_logsig
        
    def decode_ensemble(self, z_mu, z_logsig, ensemble_size):
        pred = []
        for ee in range(ensemble_size):
            z_sample = sample_normal(z_mu, z_logsig) # (B, L, C)
            pred.append(self.decoder(z_sample, mask=self.mask))
        pred = torch.stack(pred, dim=1) # (B, E, C, L)
        
        if ensemble_size>1:
            pred_sigma = torch.std(pred, dim=1)
            pred_mu = torch.mean(pred, dim=1)
        else:
            pred_mu = pred
            pred_sigma = None
            
        return pred_mu, pred_sigma, pred
        
    def decode(self, z_mu, z_logsig):
        z_sample = sample_normal(z_mu, z_logsig)
        pred = self.decoder(z_sample, mask=self.mask)
        return pred

    def forward(self, batch,
                use_latent_mean=False,
                calc_elbo=True,
                use_ensemble_in_loss=True,
                ensemble_size=1):
        
        z_mu, z_logsig = self.encode(batch)
        
        # pre-make mask so we only have to do it once
        nbatches = z_mu.shape[0]
        seq_len = z_mu.shape[1]
        self.mask = (subsequent_mask(seq_len)
            .expand(nbatches, seq_len, seq_len)
            .to(z_mu.device)
        )

        if calc_elbo:
            # KL divergence term
            kl_term = KL_std_norm(z_mu, z_logsig)

        pred = None
        pred_sigma = None
        if use_latent_mean:
            pred_mu = self.decoder(z_mu, mask=self.mask)
        elif ensemble_size>1:
            pred_mu, pred_sigma, pred = self.decode_ensemble(
                z_mu, z_logsig, ensemble_size
            )
        else:
            pred_mu = self.decode(z_mu, z_logsig)

        if calc_elbo:
            # reconstruction loss
            if ensemble_size>1 and use_ensemble_in_loss:
                neg_loglik = self.recon_loss(batch, pred_mu, pred_sigma=pred_sigma, pred_ensemble=pred)
            else:
                neg_loglik = self.recon_loss(batch, pred_mu, pred_sigma=pred_sigma)
            ELBO = (-neg_loglik) - kl_term
        
        return {'pred_mu':pred_mu, 'pred_sigma':pred_sigma, 'pred_ensemble':pred,
                'nll':neg_loglik, 'kl':kl_term, 'ELBO':ELBO}
        
        
            
        
        
