from types import SimpleNamespace
import pandas as pd

model_cfg = SimpleNamespace(
    features_d = 6, #7*2,
    features_aux = 10,
    embed_ds = 96, #24
    embed_aux = 32, #14
    dropout = 0.05,
    lr = 3e-5, 
    enc_state_attention = True,
    enc_hidden_attention = False,
    dec_hidden_attention = True,
    attn_heads = 4,
    embed_v = 16,
    features_traj = 3,
    features_dec = 96,
    # nbr params
    shortrange = 10,
    longrange = 60,    
    # node_aux_channels == features_aux
    #edge_aux_channels = 16,
    #Natt_h_nbr = 2,
    #precip_embed = 32
)

normalisation = SimpleNamespace(    
    lwin_mu = 330.,
    lwin_sd = 35.,
    swin_norm = 500.,    
    temp_mu = 10.,
    temp_sd = 10.,
    p_mu = 1013.,
    p_sd = 25.,
    rh_mu = 85.,
    rh_sd = 12.,
    ws_mu = 4.,
    ws_sd = 2., 
    precip_norm = 100.,
    lat_norm = 90.,
    lon_norm = 180.,
    elev_norm = 300.,
    slope_norm = 4.,
    std_elev_norm = 15.
    
)
