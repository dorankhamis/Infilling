from types import SimpleNamespace
import pandas as pd

model_cfg = SimpleNamespace(
    features_d = 7*2,
    embed_ds = 192, 
    dropout = 0.1,
    Natt_h = 8,
    d_ff = 334,
    Natt_l = 4,
    lr = 5e-5,
    features_aux = 10,
    embed_aux = 64,
    shortrange = 10,
    longrange = 60
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
