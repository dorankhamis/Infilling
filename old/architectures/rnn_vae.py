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
from architectures.components.attention import MultiHeadedAttention
from architectures.components.layer_norm import LayerNorm, SublayerConnection
from architectures.components.positional_encoding import PositionalEncoding
from architectures.components.nn_utils import subsequent_mask, clones


EPS = torch.tensor(1e-10)

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
            

class Encoder(nn.Module):

    def __init__(self, cfg):
        super(Encoder, self).__init__()

        self.rnn_enc = nn.GRU(cfg.features_d + cfg.features_aux, cfg.embed_ds, batch_first=True)
        
        self.cross_attention_bool = cfg.enc_state_attention
        self.self_attention_bool = cfg.enc_hidden_attention
        #if self.cross_attention_bool or self.self_attention_bool:            
        self.sublayer = SublayerConnection(cfg.embed_ds, cfg.dropout)
        self.pe = PositionalEncoding(cfg.embed_ds, cfg.dropout)
        self.self_attn = MultiHeadedAttention(cfg.attn_heads, cfg.embed_ds)
        self.point_embed = nn.Linear(cfg.features_d + cfg.features_aux, cfg.embed_ds)
        self.cross_attn = MultiHeadedAttention(cfg.attn_heads, cfg.embed_ds)
        self.hidden_to_mu = nn.Linear(cfg.embed_ds, cfg.embed_ds)
        self.hidden_to_logsig = nn.Linear(cfg.embed_ds, cfg.embed_ds)

    def forward(self, batch, return_sample=False):
        ## concat aux series onto inputs
        x = torch.cat([batch['inputs'], batch['aux_inputs']], dim=1) # (B, C, L)
        x = torch.flip(x, dims=(1,)).transpose(-2,-1)
        
        ## flip time so we go from tN to t0, and the output is initial hidden state h0
        ht_enc, _ = self.rnn_enc(x)

        if self.cross_attention_bool:
            ## do cross attention back along the raw time series
            ## and self attention across the vector of hidden states
            x_embed = self.point_embed(x)
            ht_enc = self.pe(ht_enc)
            x_embed = self.pe(x_embed)
            ht_enc = self.sublayer(ht_enc, 
                lambda ht_enc: self.cross_attn(ht_enc, x_embed, x_embed, None)
            )
        if self.self_attention_bool:
            ht_enc = self.pe(ht_enc)
            ht_enc = self.sublayer(ht_enc, 
                lambda ht_enc: self.self_attn(ht_enc, ht_enc, ht_enc, None)
            )
        h0_enc = ht_enc[:,-1:,:]

        ## transform RNN hidden state to mean, standard deviation of z0,
        ## so that z0 ~ N(z0_mu, z0_sigma)
        z0_mu = self.hidden_to_mu(h0_enc)
        z0_logsig = self.hidden_to_logsig(h0_enc)
        
        if return_sample:
            return sample_normal(z0_mu, z0_logsig)
        else:
            return z0_mu, z0_logsig


class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        
        self.aux_embed = nn.Linear(cfg.features_dec, cfg.embed_dec)
        self.latent_to_hidden = nn.Linear(cfg.embed_ds, cfg.embed_ds)
        self.rnn_dec = nn.GRU(cfg.embed_dec, cfg.embed_ds, batch_first=True)
        self.hidden_size = cfg.embed_ds

        self.attention_bool = cfg.dec_hidden_attention
        #if self.attention_bool:
        self.sublayer = SublayerConnection(cfg.embed_ds, cfg.dropout)
        self.pe = PositionalEncoding(cfg.embed_ds, cfg.dropout)
        self.self_attn = MultiHeadedAttention(cfg.attn_heads, cfg.embed_ds)
        
        self.decode_state = nn.Linear(cfg.embed_ds, cfg.features_d//2)

    def forward(self, z0_sample, aux_inputs):
        ## embed the auxiliary data
        aux_t = self.aux_embed(aux_inputs.transpose(-2,-1))
        
        ## decode the z sample using a fresh RNN        
        h0_dec = self.latent_to_hidden(z0_sample)
        h0_dec = h0_dec.transpose(0,1) # RNN funniness, (nlayers, B, C)
        
        ht_dec, _ = self.rnn_dec(aux_t, h0_dec)

        if self.attention_bool:
            ## do self attention across the vector of hidden states
            ht_dec = self.pe(ht_dec)
            ht_dec = self.sublayer(ht_dec, 
                lambda ht_dec: self.self_attn(ht_dec, ht_dec, ht_dec, None)
            )
        
        ## and predict in state space
        x_pred = self.decode_state(ht_dec).transpose(-2, -1) # (B, C, L)
        
        return x_pred
        
    def decode_from_prior(self, aux_inputs):
        zeros = torch.zeros((aux_inputs.shape[0], 1, self.hidden_size)).to(aux_inputs.device)
        z0_sample = sample_normal(zeros, zeros) # log(1) = 0
        return self.forward(z0_sample, aux_inputs)

        
class RNNVAE(nn.Module):
    def __init__(self, cfg):
        super(RNNVAE, self).__init__()
        
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.recon_loss = make_recon_loglik(use_non_gaps=True)

    def encode(self, batch):
        z0_mu, z0_logsig = self.encoder(batch)
        return z0_mu, z0_logsig
        
    def decode_ensemble(self, z0_mu, z0_logsig, ensemble_size, aux_inputs):
        pred = []
        for ee in range(ensemble_size):
            z0_sample = sample_normal(z0_mu, z0_logsig)
            pred.append(self.decoder(z0_sample, aux_inputs))
        pred = torch.stack(pred, dim=1) # (B, E, C, L)
        
        if ensemble_size>1:
            pred_sigma = torch.std(pred, dim=1)
            pred_mu = torch.mean(pred, dim=1)
        else:
            pred_mu = pred
            pred_sigma = None
            
        return pred_mu, pred_sigma, pred
        
    def decode(self, z0_mu, z0_logsig, aux_inputs):
        z0_sample = sample_normal(z0_mu, z0_logsig)
        pred = self.decoder(z0_sample, aux_inputs)
        return pred

    def forward(self, batch,
                use_latent_mean=False,
                calc_elbo=True,
                use_ensemble_in_loss=True,
                ensemble_size=1):
        
        z0_mu, z0_logsig = self.encode(batch)

        if calc_elbo:
            # KL divergence term
            kl_term = KL_std_norm(z0_mu, z0_logsig)

        pred = None
        pred_sigma = None
        if use_latent_mean:
            pred_mu = self.decoder(z0_mu, batch['dec_inputs'])
        elif ensemble_size>1:
            pred_mu, pred_sigma, pred = self.decode_ensemble(z0_mu, z0_logsig, ensemble_size, batch['dec_inputs'])
        else:
            pred_mu = self.decode(z0_mu, z0_logsig, batch['dec_inputs'])

        if calc_elbo:
            # reconstruction loss
            if ensemble_size>1 and use_ensemble_in_loss:
                neg_loglik = self.recon_loss(batch, pred_mu, pred_sigma=pred_sigma, pred_ensemble=pred)
            else:
                neg_loglik = self.recon_loss(batch, pred_mu, pred_sigma=pred_sigma)
            ELBO = (-neg_loglik) - kl_term
        
        return {'pred_mu':pred_mu, 'pred_sigma':pred_sigma, 'pred_ensemble':pred,
                'nll':neg_loglik, 'kl':kl_term, 'ELBO':ELBO}
        
        
            
        
        
