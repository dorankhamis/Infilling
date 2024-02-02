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
# cfg.features_traj = 3
# cfg.features_dec = 96

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

    def __init__(self, model_size):
        super(DistParams, self).__init__()
        self.mu_linear = nn.Linear(model_size, model_size)
        self.logsig_linear = nn.Linear(model_size, model_size)
        
    def forward(self, r):
        ## calculate mean and standard deviation from a var embedding r
        return self.mu_linear(r), self.logsig_linear(r) 
        
def get_slopes(n):
    # https://github.com/ofirpress/attention_with_linear_biases/tree/master
    def get_slopes_power_of_2(n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start*ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)                   #In the paper, we only train models that have 2^a heads for some a. This function has
    else:                                                 #some good properties that only occur when the input is a power of 2. To maintain that even
        closest_power_of_2 = 2**math.floor(math.log2(n))  #when the number of heads is not a power of 2, we use this workaround. 
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]


class Encoder(nn.Module):

    def __init__(self, cfg):
        super(Encoder, self).__init__()
        
        # variable embeddings
        self.v_encs = nn.ModuleList([
            VarEmbed(cfg) for i in range(cfg.features_d)
        ])
        
        # var attention
        self.sublayer_v = SublayerConnection(cfg.embed_v, cfg.dropout)
        self.var_attn = MultiHeadedAttention_TwoBatchDim(cfg.attn_heads, cfg.embed_v)

        # temporal attention
        model_size = cfg.embed_v * cfg.features_d
        #self.pe = PositionalEncoding(model_size, cfg.dropout)
        self.sublayer_t = SublayerConnection(model_size, cfg.dropout)
        self.temporal_attn = MultiHeadedAttention(cfg.attn_heads, model_size)
        #self.trajectory = nn.Linear(model_size, cfg.features_traj)
        
        # RNN trajectory encoder
        self.rnn_enc = nn.GRU(model_size, cfg.features_traj, batch_first=True)
        
        # normal distribution parameter creaters
        self.dist_pars = nn.ModuleList([
            DistParams(cfg.embed_v + cfg.features_traj) for i in range(cfg.features_d)
        ])
        
        ## attention with linear biases instead of positional encoding!
        # https://github.com/ofirpress/attention_with_linear_biases/tree/master
        self.slopes = torch.Tensor(get_slopes(cfg.attn_heads))

    def forward(self, batch, mask=None):
        
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
        
        ## create deterministic trajectory in latent space
        if mask is None:
            ## create causal mask
            nbatches = r_tens.shape[0]
            seq_len = r_tens.shape[1]
            mask = (subsequent_mask(seq_len)
                .expand(nbatches, seq_len, seq_len)
                .to(r_tens.device)
            )
            
        ## attention with linear biases instead of positional encoding!
        # https://github.com/ofirpress/attention_with_linear_biases/tree/master
        alibi = (self.slopes.unsqueeze(1).unsqueeze(1) * 
            torch.arange(mask.shape[-1]).unsqueeze(0).unsqueeze(0)
            .expand(len(self.slopes), -1, -1)
        )
        alibi = alibi.view(len(self.slopes), 1, maxpos)
        alibi = alibi.repeat(1, 1, 1)  # batch_size, 1, 1
        # then add alibi to the attention mask, or simply use
        # alibi as the mask if we have no other masking
        # (though if we have a bool mask we need to make it into float mask!?
        # i.e. alter our attention function)
        mask = mask + alibi # e.g.
            
        
        r_traj = r_tens.view(r_tens.shape[0], r_tens.shape[1], r_tens.shape[2]*r_tens.shape[3]) # (B, L, V*C)
        #r_traj = self.pe(r_traj) # don't do this if using alibi
        r_traj = self.sublayer_t(r_traj,
            lambda r_traj: self.temporal_attn(r_traj, r_traj, r_traj, mask)
        )

        r_traj, hN_enc = self.rnn_enc(r_traj)
        
        ## aggregate r tensor in time
        r_agg = r_tens.mean(dim=1)
        
        # join final RNN hidden state
        r_agg = torch.cat([
            r_agg,
            hN_enc.transpose(0,1).expand(-1, r_agg.shape[1], -1)
        ], dim = -1)
        
        ## create mu, sigma
        mu_list = []
        logsig_list = []
        for i in range(len(r_list)):
            # do separately for each variable to allow different learning
            mu_i, logsig_i = self.dist_pars[i](r_agg[:,i,:])  # (B, C)
            mu_list.append(mu_i)
            logsig_list.append(logsig_i)
        
        ## join variables on channel dim
        mu = torch.cat(mu_list, dim=-1) # (B, C)
        logsig = torch.cat(logsig_list, dim=-1)
               
        return mu, logsig, r_traj


class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        
        in_size = cfg.features_d * (cfg.embed_v + cfg.features_traj) + cfg.features_traj
        self.dense = nn.Linear(in_size, cfg.features_dec)
        
        self.sublayer_t = SublayerConnection(cfg.features_dec, cfg.dropout)
        #self.pe = PositionalEncoding(cfg.features_dec, cfg.dropout)
        self.temporal_attn = MultiHeadedAttention(cfg.attn_heads, cfg.features_dec)
        self.decode_state = nn.Linear(cfg.features_dec, cfg.features_d)

        ## attention with linear biases instead of positional encoding!
        # https://github.com/ofirpress/attention_with_linear_biases/tree/master
        self.slopes = torch.Tensor(get_slopes(cfg.attn_heads))

    def forward(self, z_sample, r_traj, mask=None):
        if mask is None:
            ## create causal mask
            nbatches = r_traj.shape[0]
            seq_len = r_traj.shape[1]
            mask = (subsequent_mask(seq_len)
                .expand(nbatches, seq_len, seq_len)
                .to(z_sample.device)
            )
        
        ## combine random sample with latent trajectory
        x = torch.cat([r_traj, z_sample.unsqueeze(1).expand(-1, r_traj.shape[1], -1)], dim=-1)
        x = self.dense(x)
        
        ## attention with linear biases instead of positional encoding!
        # https://github.com/ofirpress/attention_with_linear_biases/tree/master
        alibi = (self.slopes.unsqueeze(1).unsqueeze(1) * 
            torch.arange(mask.shape[-1]).unsqueeze(0).unsqueeze(0)
            .expand(len(self.slopes), -1, -1)
        )
        alibi = alibi.view(len(self.slopes), 1, mask.shape[-1])
        alibi = alibi.repeat(1, 1, 1)  # batch_size, 1, 1
        # then add alibi to the attention mask, or simply use
        # alibi as the mask if we have no other masking
        # (though if we have a bool mask we need to make it into float mask!?
        # i.e. alter our attention function)
        mask = mask + alibi # e.g.
        
        ## causal attention through time series
        #x = self.pe(x) # don't do this if using alibi
        x = self.sublayer_t(x,
            lambda x: self.temporal_attn(x, x, x, mask)
        )        
                
        ## output time series x(t)
        x = self.decode_state(x).transpose(-2, -1) # (B, C, L)
        
        ## then do attention with input batch and potentially neighbours?
        ## (masking missing data point? ... though this is different for 
        ## each variable... but could split apart again and do variable-wise)
        
        return x # (B, C, L)

        
class AggregateEmbedder(nn.Module):
    def __init__(self, cfg):
        super(AggregateEmbedder, self).__init__()
        
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.recon_loss = make_recon_loglik(use_non_gaps=True)
        self.mask = None

    def encode(self, batch, mask=None):
        z_mu, z_logsig, r_traj = self.encoder(batch, mask=mask)
        return z_mu, z_logsig, r_traj
        
    def decode_ensemble(self, z_mu, z_logsig, r_traj, ensemble_size):
        pred = []
        for ee in range(ensemble_size):
            z_sample = sample_normal(z_mu, z_logsig) # (B, C)
            pred.append(self.decoder(z_sample, r_traj, mask=self.mask))
        pred = torch.stack(pred, dim=1) # (B, E, C, L)
        
        if ensemble_size>1:
            pred_sigma = torch.std(pred, dim=1)
            pred_mu = torch.mean(pred, dim=1)
        else:
            pred_mu = pred
            pred_sigma = None
            
        return pred_mu, pred_sigma, pred
        
    def decode(self, z_mu, z_logsig, r_traj):
        z_sample = sample_normal(z_mu, z_logsig) # (B, C)
        pred = self.decoder(z_sample, r_traj, mask=self.mask)
        return pred

    def forward(self, batch,
                use_latent_mean=False,
                calc_elbo=True,
                use_ensemble_in_loss=True,
                ensemble_size=1):

        # pre-make mask so we only have to do it once
        nbatches = batch['inputs'].shape[0]
        seq_len = batch['inputs'].shape[-1]
        self.mask = (subsequent_mask(seq_len)
            .expand(nbatches, seq_len, seq_len)
            .to(batch['inputs'].device)
        )
        
        z_mu, z_logsig, r_traj = self.encode(batch, mask=self.mask)
        
        if calc_elbo:
            # KL divergence term
            kl_term = KL_std_norm(z_mu, z_logsig)

        pred = None
        pred_sigma = None
        if use_latent_mean:
            pred_mu = self.decoder(z_mu, r_traj, mask=self.mask)
        elif ensemble_size>1:
            pred_mu, pred_sigma, pred = self.decode_ensemble(
                z_mu, z_logsig, r_traj, ensemble_size
            )
        else:
            pred_mu = self.decode(z_mu, z_logsig, r_traj)

        if calc_elbo:
            # reconstruction loss
            if ensemble_size>1 and use_ensemble_in_loss:
                neg_loglik = self.recon_loss(batch, pred_mu, pred_sigma=pred_sigma, pred_ensemble=pred)
            else:
                neg_loglik = self.recon_loss(batch, pred_mu, pred_sigma=pred_sigma)
            ELBO = (-neg_loglik) - kl_term
        
        return {'pred_mu':pred_mu, 'pred_sigma':pred_sigma,
                'pred_ensemble':pred, 'nll':neg_loglik,
                'kl':kl_term, 'ELBO':ELBO, 'r_trajectory':r_traj}
        
        
            
        
        
