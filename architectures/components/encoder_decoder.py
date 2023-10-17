import numpy as np
import torch
import torch.nn as nn
from .nn_utils import Identity

class EncoderDecoder(nn.Module):    
    def __init__(self, encoder, src_embed, decoder, 
                 tgt_embed=Identity(), generator=Identity()):
        """
            - src: (B, S, F) = (batch, seq_len, features)
            - src_mask: (S, S): causal masking
            - tgt_mask: (T, T): causal/no cheating masking
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, src_mask=None, tgt=None, tgt_mask=None):        
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
    
    def encode(self, src, src_mask=None):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask=None, tgt=None, tgt_mask=None):
        return self.decoder(memory, src_mask, self.tgt_embed(tgt), tgt_mask)
