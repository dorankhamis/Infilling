import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch import Tensor
from torch.autograd import Variable

from architectures.components.mlp import PositionwiseFeedForward
from architectures.components.attention import MultiHeadedAttention
from architectures.components.layer_norm import LayerNorm, SublayerConnection
from architectures.components.positional_encoding import PositionalEncoding
from architectures.components.nn_utils import subsequent_mask, clones

class PureAuxPred(nn.Module):
    def __init__(self, cfg):
        super(PureAuxPred, self).__init__()        
        self.aux_embedding1 = nn.Conv1d(cfg.features_aux, cfg.embed_aux, kernel_size=1)
        self.aux_embedding2 = nn.Conv1d(cfg.embed_aux, cfg.embed_aux, kernel_size=1)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(cfg.dropout)
        
        # prediction
        self.pred1 = nn.Linear(cfg.embed_aux, cfg.features_aux * 2) # as we output features not density
        self.pred2 = nn.Linear(cfg.features_aux * 2, cfg.features_d // 2) # as we output features not density
        
    def forward(self, x_in, aux_in):
        aux = self.aux_embedding1(aux_in)
        xaux = self.gelu(aux)
        aux = self.dropout(aux)
        
        aux = self.aux_embedding2(aux)
        aux = self.gelu(aux)
        aux = self.dropout(aux)

        # predict
        aux = self.pred1(aux.transpose(1,2)) # (B, L, C)
        aux = self.pred2(aux)
        return aux.transpose(1,2) # (B, C, L)

class AttentionBlock(nn.Module):
    def __init__(self, layer, N):
        super(AttentionBlock, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x):        
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class EncoderLayer(nn.Module):    
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x):        
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)


class NbrAttn(nn.Module):
    def __init__(self, ts_channels, ts_embed,
                 edge_aux_channels, aux_embed,
                 edge_aux_channels,
                 d_cross_attn, attn_heads,
                 dropout_rate):
        super(NbrAttn, self).__init__()
        self.process_ts = nn.Linear(ts_channels, ts_embed)
        self.process_node_aux = nn.Linear(node_aux_channels, aux_embed)
        self.process_edge_aux = nn.Linear(edge_aux_channels, d_cross_attn)
            
        self.embed_nbr = nn.Linear(ts_embed + aux_embed, d_cross_attn)
        self.sublayer = SublayerConnection(d_cross_attn, dropout_rate)
        self.pe = PositionalEncoding(d_cross_attn, dropout)
        self.cross_attn = MultiHeadedAttention(attn_heads, d_cross_attn,
                                               dropout=dropout_rate)
        
                                     
    def forward(self, x, nbr_data):                
        attn_out = []
        for b in range(len(nbr_data)): # loop over batches
            if len(nbr_data[b])==0: # no nbrs
                attn_out.append(x[b])
            else:      
                # take batch slice
                x_b = x[b:(b+1)]
                
                # create dummy
                nbr_attn_arrs = torch.zeros((0, x_b.shape[1], x_b.shape[2])) # requires grad = True?
                
                # do we want to do positional encoding of x_b?
                # (might be "double counting" PE from self attention?)
                x_b = self.pe(x_b.transpose(-2,-1))
                
                # loop over neighbour timeseries
                for i, this_nbr in enumerate(nbr_data[b]):
                    # process nbr timeseries and node/edge aux
                    nbr_b_i_ts = self.process_ts(nbr_data[b][i]['masked_data'].transpose(-2,-1)).transpose(-2,-1)
                    nbr_b_i_aux = self.process_node_aux(nbr_node[b][i]['node_aux'].transpose(-2,-1)).transpose(-2,-1)
                    nbr_b_i_edge = self.process_edge_aux(nbr_node[b][i]['edge_aux'].transpose(-2,-1)).transpose(-2,-1)
                    nbr_b_i = torch.cat([nbr_b_i_ts, nbr_b_i_aux], dim=1)
                    
                    # embed nbr data and do positional encoding
                    nbr_b_i = self.embed_nbr(nbr_b_i.transpose(-2,-1)).transpose(-2,-1)
                    nbr_b_i = self.pe(nbr_b_i.transpose(-2,-1))
                                            
                    # calculate cross-attention over nbr timeseries                    
                    this_out = self.sublayer(x_b, 
                        lambda x_b: self.cross_attn(x_b,
                                                    nbr_b_i,
                                                    nbr_b_i,
                                                    nbr_data[b][i]['attention_mask'])
                    )                    
                    # but how do we combine information from different neighbours?
                    # i.e. an attention over all neighbour time series'
                    # cross-attention outputs with the edge aux as intelligent weighting
                    
                    # maybe we accumulate the neighbour attention/sublayer output 
                    # and then multiply each by an embedded edge aux
                    # and then take the average as our new x???
                    
                    nbr_attn_arrs = torch.cat([nbr_attn_arrs, this_out * nbr_b_i_edge], dim=0)
                                
                attn_out.append(torch.mean(nbr_attn_arrs, dim=0)[0,...]) # remove batch dim
                
        # process outputs
        return torch.stack(attn_out, dim=0)


class AttnGapFill(nn.Module):
    def __init__(self, cfg):
        super(AttnGapFill, self).__init__()
        """ cfg is a SimpleNameSpace containing:
            features_d: the number of features
            embed_ds: embedding dimension
            dropout: dropout rate
            Natt_h: number of attention heads
            Natt_l: number of attention layers per head
            d_ff: hidden dimension in feed forward layers
        """        
        # embedding
        self.initial_embedding1 = nn.Conv1d(cfg.features_d, cfg.embed_ds, kernel_size=1)
        self.initial_embedding2 = nn.Conv1d(cfg.embed_ds, cfg.embed_ds, kernel_size=1)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(cfg.dropout)
        
        self.aux_embedding1 = nn.Conv1d(cfg.features_aux, cfg.embed_aux, kernel_size=1)
        self.aux_embedding2 = nn.Conv1d(cfg.embed_aux, cfg.embed_aux, kernel_size=1)
        
        # attention
        model_size = cfg.embed_ds + cfg.embed_aux
        self.pe = PositionalEncoding(model_size, cfg.dropout)        
        c = copy.deepcopy
        attn = MultiHeadedAttention(cfg.Natt_h, model_size)
        ff = PositionwiseFeedForward(model_size, cfg.d_ff, cfg.dropout)
        self.attblck = AttentionBlock(EncoderLayer(model_size, c(attn), c(ff), cfg.dropout), cfg.Natt_l)
        
        # prediction        
        self.pred1 = nn.Linear(model_size, cfg.features_d * 2) # as we output features not density
        self.pred2 = nn.Linear(cfg.features_d * 2, cfg.features_d // 2) # as we output features not density

    def forward(self, x_in, aux_in):
        """
        input x: (batch_size, 2*n_features [feature + density], length_time)            
        """
        # do initial embeddings
        x = self.initial_embedding1(x_in)
        x = self.gelu(x)
        x = self.dropout(x)
        
        x = self.initial_embedding2(x)
        x = self.gelu(x)
        x = self.dropout(x)
        
        aux = self.aux_embedding1(aux_in)
        aux = self.gelu(aux)
        aux = self.dropout(aux)
        
        aux = self.aux_embedding2(aux)
        aux = self.gelu(aux)
        aux = self.dropout(aux)
        
        # join on channel dim
        x = torch.cat([x, aux], dim=1)
        
        # add positional encoding
        x = self.pe(x.transpose(-2,-1)) # (B, L, C)
        
        # self attention across time series        
        x = self.attblck(x)
        
        # optional neighbour attentions
        
        
        
                
        # predict
        x = self.pred1(x)
        x = self.pred2(x)
        return x.transpose(1,2) # (B, C, L)
