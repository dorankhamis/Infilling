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


class NbrAttn1(nn.Module):
    def __init__(self, ts_channels, ts_embed,
                 node_aux_channels, aux_embed,
                 edge_aux_channels,
                 d_cross_attn, attn_heads,
                 dropout):
        super(NbrAttn1, self).__init__()
        self.process_ts = nn.Linear(ts_channels, ts_embed)
        self.process_node_aux = nn.Linear(node_aux_channels, aux_embed)
        self.process_edge_aux = nn.Linear(edge_aux_channels, d_cross_attn)
            
        self.embed_nbr = nn.Linear(ts_embed + aux_embed, d_cross_attn)
        self.sublayer = SublayerConnection(d_cross_attn, dropout_rate)
        self.pe = PositionalEncoding(d_cross_attn, dropout)
        self.cross_attn = MultiHeadedAttention(attn_heads, d_cross_attn)
        
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
                x_b = self.pe(x_b)
                
                # loop over neighbour timeseries
                for i, this_nbr in enumerate(nbr_data[b]):
                    # process nbr timeseries and node/edge aux
                    nbr_b_i_ts = self.process_ts(this_nbr['masked_data'].transpose(-2,-1))
                    nbr_b_i_aux = self.process_node_aux(this_nbr['node_aux'].transpose(-2,-1))
                    nbr_b_i_edge = self.process_edge_aux(this_nbr['edge_aux'].transpose(-2,-1))
                    nbr_b_i = torch.cat([nbr_b_i_ts, nbr_b_i_aux], dim=-1)
                                        
                    nbr_b_i = self.pe(nbr_b_i)
                                            
                    # calculate cross-attention over nbr timeseries                    
                    this_out = self.sublayer(x_b, 
                        lambda x_b: self.cross_attn(x_b,
                                                    nbr_b_i,
                                                    nbr_b_i,
                                                    nbr_data[b][i]['attention_mask'])
                    )                    
                    nbr_attn_arrs = torch.cat([nbr_attn_arrs, this_out * nbr_b_i_edge], dim=0)
                                
                attn_out.append(torch.mean(nbr_attn_arrs, dim=0)) # remove batch dim
                
        # process outputs
        return torch.stack(attn_out, dim=0)

class NbrAttn2(nn.Module):
    def __init__(self, ts_channels, ts_embed,
                 node_aux_channels, aux_embed,
                 edge_aux_channels,
                 d_cross_attn, attn_heads,
                 dropout):
        super(NbrAttn2, self).__init__()
        if ts_channels==1:
            self.process_ts_and_node1 = nn.Linear(ts_channels + node_aux_channels, d_cross_attn)
            self.process_ts_and_node2 = nn.Linear(d_cross_attn, d_cross_attn)
            self.prejoin = True
        else:
            self.process_ts1 = nn.Linear(ts_channels, ts_embed)
            self.process_ts2 = nn.Linear(ts_embed, ts_embed)
            self.process_node_aux1 = nn.Linear(node_aux_channels, aux_embed)
            self.process_node_aux2 = nn.Linear(aux_embed, aux_embed)
            self.prejoin = False
        
        self.process_edge_aux1 = nn.Linear(edge_aux_channels, d_cross_attn)
        self.process_edge_aux2 = nn.Linear(d_cross_attn, d_cross_attn)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(cfg.dropout)

        self.sublayer = SublayerConnection(d_cross_attn, dropout)
        self.pe = PositionalEncoding(d_cross_attn, dropout)
        self.cross_attn = MultiHeadedAttention(attn_heads, d_cross_attn)
        
    def forward(self, x, nbr_data):                
        attn_out = []
        for b in range(len(nbr_data)): # loop over batches
            if len(nbr_data[b])==0: # no nbrs
                attn_out.append(x[b])
            else:      
                # take batch slice
                x_b = x[b:(b+1)]
                
                # create dummies for accumulating neighbours
                nbr_attn_keys = torch.zeros((1, 0, x_b.shape[2])) # requires grad = True?
                nbr_attn_values = torch.zeros((1, 0, x_b.shape[2])) # requires grad = True?
                nbr_attn_masks = torch.zeros((1, x_b.shape[1], 0)) # requires grad = True?
                
                # do we want to do positional encoding of x_b?
                # (might be "double counting" PE from self attention?)
                x_b = self.pe(x_b)
                
                # loop over neighbour timeseries
                for i, this_nbr in enumerate(nbr_data[b]):
                    # process nbr timeseries and node/edge aux
                    if self.prejoin:
                        nbr_b_i = torch.cat([this_nbr['masked_data'], this_nbr['node_aux']], dim=1)
                        nbr_b_i = self.process_ts_and_node1(nbr_b_i.transpose(-2,-1))
                        nbr_b_i = self.relu(nbr_b_i)
                        nbr_b_i = self.dropout(nbr_b_i)
                        nbr_b_i = self.process_ts_and_node2(nbr_b_i)
                    else:
                        nbr_b_i_ts = self.process_ts1(this_nbr['masked_data'].transpose(-2,-1))
                        nbr_b_i_ts = self.relu(nbr_b_i_ts)
                        nbr_b_i_ts = self.dropout(nbr_b_i_ts)
                        nbr_b_i_ts = self.process_ts2(nbr_b_i_ts)
                        
                        nbr_b_i_aux = self.process_node_aux1(this_nbr['node_aux'].transpose(-2,-1))
                        nbr_b_i_aux = self.relu(nbr_b_i_aux)
                        nbr_b_i_aux = self.dropout(nbr_b_i_aux)
                        nbr_b_i_aux = self.process_node_aux2(nbr_b_i_aux)
                        
                        nbr_b_i = torch.cat([nbr_b_i_ts, nbr_b_i_aux], dim=-1)
                        
                    nbr_b_i_edge = self.process_edge_aux1(this_nbr['edge_aux'].transpose(-2,-1))
                    nbr_b_i_edge = self.relu(nbr_b_i_edge)
                    nbr_b_i_edge = self.dropout(nbr_b_i_edge)
                    nbr_b_i_edge = self.process_edge_aux2(nbr_b_i_edge)                    
                    nbr_b_i_key = nbr_b_i * nbr_b_i_edge # condition on the edge for keys
                    
                    nbr_b_i_key = self.pe(nbr_b_i_key)
                    nbr_b_i = self.pe(nbr_b_i)
                    
                    nbr_attn_keys = torch.cat([nbr_attn_keys, nbr_b_i_key], dim=1)
                    nbr_attn_values = torch.cat([nbr_attn_values, nbr_b_i], dim=1)
                    nbr_attn_masks = torch.cat([nbr_attn_masks, this_nbr['attention_mask']], dim=-1)                    
                                            
                # calculate cross-attention over nbr timeseries                    
                x_attn = sublayer(x_b, 
                    lambda x_b: cross_attn(x_b,
                                           nbr_attn_keys,
                                           nbr_attn_values,
                                           nbr_attn_masks)
                )
                attn_out.append(x_attn[0]) # remove batch dim
        
        # process outputs
        return torch.stack(attn_out, dim=0)


class NbrAttnGapFill(nn.Module):
    def __init__(self, cfg):
        super(NbrAttnGapFill, self).__init__()
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
        
        # nbr attention
        self.allvar_nbr_attn = NbrAttn2(
            cfg.features_d, cfg.embed_ds,
            cfg.features_aux, cfg.embed_aux,
            cfg.edge_aux_channels,
            model_size, cfg.Natt_h_nbr,
            cfg.dropout
        )
        
        self.precip_nbr_attn = NbrAttn2(
            1, 1,
            cfg.features_aux, cfg.embed_aux,
            cfg.edge_aux_channels,
            cfg.precip_embed, cfg.Natt_h_nbr,
            cfg.dropout
        )        
        self.embed_for_precip = nn.Linear(model_size, cfg.precip_embed)
        
        # prediction        
        self.pred1 = nn.Linear(model_size, cfg.features_d * 2) # as we output features not density
        self.pred2 = nn.Linear(cfg.features_d * 2, cfg.features_d // 2 - 1) # as we output features - precip
        self.pred_precip1 = nn.Linear(cfg.precip_embed, cfg.precip_embed//2) # as we output features not density
        self.pred_precip2 = nn.Linear(cfg.precip_embed//2, 1) # as we output features - precip

    def forward(self, x_in, aux_in, precip_nbr_data=None, allvar_nbr_data=None):
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
        
        # optional neighbour attentions, split precip from other vars
        x_precip = self.embed_for_precip(x)
        x_precip = self.precip_nbr_attn(x_precip, precip_nbr_data)
        
        x_allvar = self.allvar_nbr_attn(x, allvar_nbr_data)
                            
        # predict
        x_allvar = self.pred1(x_allvar)
        x_allvar = self.pred2(x_allvar)
        x_precip = self.pred_precip1(x_precip)
        x_precip = self.pred_precip2(x_precip)
        """
        Also output error estimates?
        """
        return x_allvar.transpose(1,2), x_precip.transpose(1,2) # (B, C, L)


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
                           
        # predict
        x = self.pred1(x)
        x = self.pred2(x)
        return x.transpose(1,2) # (B, C, L)

