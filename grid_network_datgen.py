import torch
import datetime
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import xarray as xr

from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from skimage.draw import line

from params import normalisation as nm
from utils import *

hj_base = '/gws/nopw/j04/hydro_jules/data/uk/'
hj_ancil_fldr = hj_base + '/ancillaries/'
chessmet_dir = hj_base + '/driving_data/chess/chess-met/daily/'
home_data_dir = '/home/users/doran/data_dump/'
nz_base = '/gws/nopw/j04/ceh_generic/netzero/'
nz_train_path = nz_base + '/downscaling/training_data/'
TWOPI = 2. * np.pi

def load_process_chess(year, month, chess_var, normalise=True):
    if type(chess_var)==str: chess_var = [chess_var]
    chess_var = list(set(chess_var))
    
    ## load
    st = chessmet_dir + f'/chess-met_'
    en = f'_gb_1km_daily_{year}{zeropad_strint(month)}*.nc'    
    fnames = [glob.glob(st+v+en)[0] for v in chess_var]
    chess_dat = xr.open_mfdataset(fnames)

    ## rescale
    # specific humidity to RH, requires psurf in Pa and T in K
    chess_dat.huss.values = (0.263 * chess_dat.psurf.values * chess_dat.huss.values * 
        np.exp((-17.67 * (chess_dat.tas.values - 273.16)) /
                (chess_dat.tas.values - 29.65)))
    # K to C
    chess_dat.tas.values = chess_dat.tas.values - 273.15
    # Pa to hPa
    chess_dat.psurf.values = chess_dat.psurf.values / 100.
    # kg m-2 s-1 to mm total
    chess_dat.precip.values = chess_dat.precip.values * 3600 * 24
    if normalise:
        return normalise_chess_data(chess_dat, chess_var)
    else:
        return chess_dat

def normalise_chess_data(chess_dat, chess_var):
    ## normalise
    if ('rlds' in chess_var) or chess_var is None:
        # incoming longwave radiation
        chess_dat.rlds.values = (chess_dat.rlds.values - nm.lwin_mu) / nm.lwin_sd
    if ('rsds' in chess_var) or chess_var is None:
        # incoming shortwave radiation
        #chess_dat.rsds.values = np.log(1. + chess_dat.rsds.values)
        #chess_dat.rsds.values = (chess_dat.rsds.values - nm.logswin_mu) / nm.logswin_sd
        chess_dat.rsds.values = chess_dat.rsds.values / nm.swin_norm
    if ('psurf' in chess_var) or ('huss' in chess_var) or ('rsds' in chess_var) or chess_var is None:
        # air pressure
        chess_dat.psurf.values = (chess_dat.psurf.values - nm.p_mu) / nm.p_sd
    if ('huss' in chess_var) or ('rlds' in chess_var) or chess_var is None:
        # relative humidity            
        chess_dat.huss.values = (chess_dat.huss.values - nm.rh_mu) / nm.rh_sd
    if ('tas' in chess_var) or ('huss' in chess_var) or chess_var is None:
        # temperature
        chess_dat.tas.values = (chess_dat.tas.values - nm.temp_mu) / nm.temp_sd
    if ('sfcWind' in chess_var) or chess_var is None:
        # wind speed            
        chess_dat.sfcWind.values = (chess_dat.sfcWind.values - nm.ws_mu) / nm.ws_sd
    if ('dtr' in chess_var):
        # daily temperature range
        chess_dat.dtr.values = chess_dat.dtr.values / nm.temp_sd
    return chess_dat
  
def extract_edges(knn_inds, mask=None):
    edges = []
    for i in range(knn_inds.shape[0]):
        for j in range(knn_inds.shape[1]):
            if mask is None:
                edges.append((i, knn_inds[i,j]))
                edges.append((knn_inds[i,j], i))
            else:
                if mask[i,j]:
                    edges.append((i, knn_inds[i,j]))
                    edges.append((knn_inds[i,j], i))
    edges = list(set(edges))
    return edges

'''
# Data structures for torch geometric PyG
data.x: Node feature matrix with shape [num_nodes, num_node_features]

data.edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long

data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]

data.y: Target to train against (may have arbitrary shape), e.g., node-level targets of shape [num_nodes, *] or graph-level targets of shape [1, *]

data.pos: Node position matrix with shape [num_nodes, num_dimensions]

'''

## subset to a network of pixels representing observation stations
chess_var = ['tas', 'psurf', 'rsds', 'rlds', 'sfcWind', 'huss', 'precip']
dummy_data = load_process_chess(2015, 5, chess_var, normalise=False)

dummy_data.


fine_grid = xr.open_dataset(home_data_dir + '/bng_grids/bng_1km_28km_parent_pixel_ids.nc')
fine_grid = fine_grid.load()
fine_grid = fine_grid.isel(y=np.arange(1057), x=np.arange(656)) # cut down to chess grid

_, all_y, all_x = np.where(~np.isnan(dummy_data.tas))
network_size = 100
network_inds = np.random.choice(np.arange(len(all_y)), size = network_size, replace=False)
network_meta = pd.DataFrame({
    'SITE_ID':network_inds,
    'easting':fine_grid.x[all_x[network_inds]].values / 1000., # in km
    'northing':fine_grid.y[all_y[network_inds]].values / 1000., # in km
    'y_ind':all_y[network_inds],
    'x_ind':all_x[network_inds]
})
del(dummy_data)

## display network nodes
fig, ax = plt.subplots()
ax.imshow(fine_grid.landfrac.values[::-1, :], alpha=0.6, cmap='Greys')
ax.scatter(network_meta.x_ind, len(fine_grid.y) - network_meta.y_ind,
           s=15, c='r', marker='s')
plt.show()

""" data.pos: Node position matrix with shape [num_nodes, num_dimensions] """
node_pos = network_meta[['y_ind', 'x_ind']].values

## calculate the adjacency matrix / knn / edge structures for remaining stations
nbrs = NearestNeighbors(n_neighbors=6).fit(network_meta[['northing','easting']])
nbr_dists, nbr_inds = nbrs.kneighbors(return_distance=True)       
""" data.edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long """
edges = extract_edges(nbr_inds)
edges = np.asarray(edges).T

## display network nodes and edges
from matplotlib.collections import LineCollection
points = np.stack([network_meta.x_ind.values,
                   len(fine_grid.y) - network_meta.y_ind.values], axis=1)

fig, ax = plt.subplots()
lc = LineCollection(points[edges.T], linewidths=0.4)
ax.imshow(fine_grid.landfrac.values[::-1, :], alpha=0.6, cmap='Greys')
plt.gca().add_collection(lc)
ax.scatter(points[:,0], points[:,1], s=20, c='r', marker='s')
plt.show()

if False:
    ## mask edges that connect too distant nodes
    plt.hist(nbr_dists.flatten(), bins=60)
    plt.show()
    dist_thresh = 100
    mask = nbr_dists < dist_thresh
    edges = extract_edges(nbr_inds, mask=mask)
    edges = np.asarray(edges).T

    fig, ax = plt.subplots()
    lc = LineCollection(points[edges.T], linewidths=0.4)
    ax.imshow(fine_grid.landfrac.values[::-1, :], alpha=0.6, cmap='Greys')
    plt.gca().add_collection(lc)
    ax.scatter(points[:,0], points[:,1], s=20, c='r', marker='s')
    plt.show()


## gather the ancillary grids for the whole map and for the stations
topography = xr.open_dataset(hj_ancil_fldr + 'uk_ihdtm_topography+topoindex_1km.nc').load()
topography.elev.values[np.isnan(topography.elev.values)] = 0
topography.stdev.values[np.isnan(topography.stdev.values)] = 0
topography.slope.values[np.isnan(topography.slope.values)] = 0
# aspect, sea NaNs are tricky, aspect is "straight up", stick with uniform noise
asp_mask = np.isnan(topography.aspect.values)
topography.aspect.values[asp_mask] =  np.random.uniform(
    low=0, high=360, size=len(np.where(asp_mask)[0])
)
topography = topography.drop_vars(['area', 'topi', 'stdtopi', 'fdepth'])

network_meta = network_meta.assign(ALTITUDE = topography.elev.values[all_y[network_inds], all_x[network_inds]])

## calculate the mean edge ancillaries representing the "separation"
## and "barriers" between neighbouring stations
edge_attrs = []
nm.elev_std_between_pixels = np.nanstd(topography.elev.values.flatten())
for i in range(edges.shape[1]):    
    edge = edges[:,i]
    ## get edge metadata
    pix_row, pix_col = line(int(network_meta.iloc[edge[0]].y_ind),
                            int(network_meta.iloc[edge[0]].x_ind),
                            int(network_meta.iloc[edge[1]].y_ind),
                            int(network_meta.iloc[edge[1]].x_ind))
    landfrac = fine_grid.landfrac.values[pix_row, pix_col]
    elevs = topography.elev.values[pix_row, pix_col]
    std_elevs = topography.stdev.values[pix_row, pix_col]
    slopes = topography.slope.values[pix_row, pix_col]
    aspect = topography.aspect.values[pix_row, pix_col]
    edge_aux = []
    
    # distance
    if edge[1] in nbr_inds[edge[0],:]:
        ni = np.where(nbr_inds[edge[0],:]==edge[1])[0][0]
        edge_aux += [nbr_dists[edge[0],ni] / nm.dist_norm]
    elif edge[0] in nbr_inds[edge[1],:]:
        ni = np.where(nbr_inds[edge[1],:]==edge[0])[0][0]
        edge_aux += [nbr_dists[edge[1],ni] / nm.dist_norm]
    else:
        print("False edge!")
        
    # topographic statistics
    edge_aux += [elevs.min() / min(elevs[0], elevs[-1])] # relative canyons
    edge_aux += [elevs.max() / max(elevs[0], elevs[-1])] # relative mountains
    edge_aux += [elevs.mean() / (0.5*(elevs[0]+elevs[-1]))] # relative average
    edge_aux += [abs(np.diff(elevs)).max() / nm.elev_std_between_pixels] # biggest jump
    edge_aux += [elevs.std() / nm.elev_std_between_pixels] # stddev along edge
    edge_aux += [std_elevs.max() / nm.std_elev_norm] # max stddev within pixels
    edge_aux += [landfrac.sum() / len(landfrac)] # water/land fraction along edge
    edge_aux += [slopes.max() / nm.slope_norm] # steepest section of edge    
    edge_aux += [(1 - np.cos(np.deg2rad(aspect[1:] - aspect[:-1]))).max()] # biggest aspect cosine distance
    edge_attrs.append(edge_aux)

""" data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features] """
edge_attrs = np.stack(edge_attrs, axis=0)

## visualise edge attrs
sns.clustermap(edge_attrs, cmap='plasma', col_cluster=False); plt.show()

## load node data (time series)
year = 2015
month = 5
data = load_process_chess(year, month, chess_var, normalise=True)

""" data.x: Node feature matrix with shape [num_nodes, num_node_features] """
timeseries = []
var_order = chess_var.copy()
for i in range(len(data.time)):
    timeseries.append(pd.DataFrame({v:data.isel(time=i)[v].values[all_y[network_inds], all_x[network_inds]] for v in chess_var}))
    # sort columns    
    timeseries[i] = timeseries[i][var_order].values
    
timeseries = np.stack(timeseries, axis=-1) # (N, V, T) == (node, var, time)

## introduce false gaps into the timeseries of the stations
gap_mask_timeseries = []
real_gap_mask_timeseries = []
masked_timeseries = []
timeseries = timeseries.transpose(0, 2, 1) # (N, T, V)
for i in range(timeseries.shape[0]):
    this_timeseries = pd.DataFrame(timeseries[i], columns=var_order, index=data.time)
    g, rg = create_gap_masks(
        this_timeseries,
        min_gap_logprob = -0.9,
        mean_gap_length = 10,
        gap_length_sd = 5
    )
    gap_mask_timeseries.append(g)
    real_gap_mask_timeseries.append(rg)
    masked_timeseries.append(insert_gaps(this_timeseries, gap_masks=g, real_gap_masks=rg))
    # sort columns
    masked_timeseries[i] = masked_timeseries[i][var_order].values    

del(g)
del(rg)

masked_timeseries = np.stack(masked_timeseries, axis=0) # (N, T, V)
masked_timeseries = masked_timeseries.transpose(0, 2, 1) # (N, V, T)
timeseries = timeseries.transpose(0, 2, 1) # (N, V, T)

# arrange masks into (var, time) arrays
gap_masks_arr = []
real_gap_masks_arr = []
for i in range(timeseries.shape[0]):
    gap_masks_arr.append(
        np.stack(
            [gap_mask_timeseries[i][v] for v in var_order], axis=1
        )
    )
    real_gap_masks_arr.append(
        np.stack(
            [real_gap_mask_timeseries[i][v] for v in var_order], axis=1
        )
    )
our_gaps = np.stack(gap_masks_arr, axis=0) # (N, T, V)
existing_gaps = np.stack(real_gap_masks_arr, axis=0) # (N, T, V)
our_gaps = our_gaps.transpose(0, 2, 1) # (N, V, T)
existing_gaps = existing_gaps.transpose(0, 2, 1) # (N, V, T)
# use: batch['observed_mask'] = (~batch['our_gaps']) * (~batch['existing_gaps']) on mask tensors
observed_mask = (~our_gaps) * (~existing_gaps)

## gather time-varying auxiliary data (time of year/day, cloud cover) and join to
## static aux data (elevation, latitude)
# shading
shading_aux_timeseries = []
for doy in (pd.to_datetime(data.time.values).dayofyear - 1): # zero indexed
    shading_ds = xr.open_dataset(nz_train_path + f'./terrain_shading/shading_mask_{doy}.nc')    
    shading_aux_timeseries.append(
        shading_ds.shading.values[:,network_meta.y_ind,network_meta.x_ind].mean(axis=0)
    )
    
# time of year
sin_aux_timeseries = np.asarray(np.sin(
    (pd.to_datetime(data.time.values).dayofyear - 1) / 365 * TWOPI),
    dtype=np.float32
)
cos_aux_timeseries = np.asarray(np.cos(
    (pd.to_datetime(data.time.values).dayofyear - 1) / 365 * TWOPI),
    dtype=np.float32
)

# collect node aux data across time series
auxiliary_timeseries = []
for i in range(len(sin_aux_timeseries)):
    auxiliary_timeseries.append(
        (network_meta[['easting', 'northing', 'ALTITUDE']]/1000.).assign(
            shading = shading_aux_timeseries[i],
            sin_toy = sin_aux_timeseries[i],
            cos_toy = sin_aux_timeseries[i]
        ).copy()
    )
auxiliary_timeseries = np.stack(auxiliary_timeseries, axis=-1) # (N, AV, T)

## create tensors for inputs
auxiliary_timeseries = torch.from_numpy(auxiliary_timeseries).to(torch.float32)
timeseries = torch.from_numpy(timeseries).to(torch.float32)
masked_timeseries = torch.from_numpy(masked_timeseries).to(torch.float32)
observed_mask = torch.from_numpy(observed_mask)
edge_attrs = torch.from_numpy(edge_attrs).to(torch.float32)
edges = torch.from_numpy(edges)


## infill gaps using network of nodes and temporal continuity!
'''
Model structure:
    i) VarEmbed (from Aggregate Embedder) within node, using only aux data
        where observed_mask is false
    ii) graph attention https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html#torch_geometric.nn.conv.GATv2Conv
        but make it aware of where observed mask was initially false?
    iii) temporal self-attention within node to condition on gap edges.
        might want to create non-symmetric attention masks to hide gaps
        from non-gaps, but not vice-versa
    iv) uncertainty outputs where we have gaps / aware of where we have gaps
'''

import math
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from torch.distributions.normal import Normal
from torch.distributions import kl
from torch_geometric.data import Data

from architectures.components.mlp import PositionwiseFeedForward
from architectures.components.attention import (MultiHeadedAttention,
                                                MultiHeadedAttention_TwoBatchDim)
from architectures.components.layer_norm import LayerNorm, SublayerConnection
from architectures.components.positional_encoding import PositionalEncoding
from architectures.components.nn_utils import subsequent_mask, clones

from types import SimpleNamespace

cfg = SimpleNamespace(
    features_d = timeseries.shape[1],
    node_features_aux = auxiliary_timeseries.shape[1],
    edge_features_aux = edge_attrs.shape[-1],
    embed_v = 2,
    embed_ds = 24,
    embed_aux = 12,
    dropout = 0.05,
    lr = 5e-4,
    attn_heads = 4
)


def get_slopes(n):
    def get_slopes_power_of_2(n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start*ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:  
        closest_power_of_2 = 2**math.floor(math.log2(n))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]

maxpos = timeseries.shape[-1]
attn_heads = cfg.attn_heads
slopes = torch.Tensor(get_slopes(attn_heads))
alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(maxpos).unsqueeze(0).unsqueeze(0).expand(attn_heads, -1, -1)
alibi = alibi.view(attn_heads, 1, maxpos)
alibi = alibi.repeat(1, 1, 1)

# create mask matrices of the correct size
mask = []
for i in range(maxpos):
    mask.append(alibi[:,:,np.hstack([np.arange(1, i+1)[::-1], np.arange(0, maxpos-i)])])
mask = torch.cat(mask, dim=1)

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)
    
def buffered_future_mask(self, tensor):
    dim = tensor.size(1)
    self._future_mask = torch.triu(
        fill_with_neg_inf(torch.zeros([self.args.tokens_per_sample, self.args.tokens_per_sample])), 1
    )
    self._future_mask = self._future_mask.unsqueeze(0) + self.alibi
    self._future_mask = self._future_mask.to(tensor)
    return self._future_mask[:tensor.shape[0]*self.args.decoder_attention_heads, :dim, :dim]

class VarEmbed(nn.Module):

    def __init__(self, cfg):
        super(VarEmbed, self).__init__()
        
        self.relu = nn.ReLU()
        self.pre_O = nn.Linear(1 + cfg.node_features_aux, 2*cfg.embed_v)
        self.pre_M = nn.Linear(cfg.node_features_aux, 2*cfg.embed_v)        
        self.embed_r_O = nn.Linear(2*cfg.embed_v, cfg.embed_v)
        self.embed_r_M = nn.Linear(2*cfg.embed_v, cfg.embed_v)

    def forward(self, x, aux, mask_O):
        ## concat aux series onto inputs
        xj = torch.cat([x, aux], dim=1) # (B, C, L)
        
        ## embed to latent space
        # observed mapping
        r_O = self.pre_O(xj.transpose(-2, -1)) # (B, L, C)
        r_O = self.relu(r_O)
        r_O = self.embed_r_O(r_O)
        # add indicator variable for observed
        r_O = torch.cat([
            r_O,
            torch.ones((r_O.shape[0], r_O.shape[1], 1), dtype=torch.float32).to(r_O.device)
        ], dim = -1)
        
        # auxiliary-driven mapping for data gaps
        r_M = self.pre_M(aux.transpose(-2, -1)) # (B, L, C)
        r_M = self.relu(r_M)
        r_M = self.embed_r_M(r_M)
        # add indicator variable for gap
        r_M = torch.cat([
            r_M,
            torch.zeros((r_M.shape[0], r_M.shape[1], 1), dtype=torch.float32).to(r_M.device)
        ], dim = -1)
        
        ## sum using masks
        r = (r_O * mask_O.transpose(-2, -1) + 
            r_M * ~(mask_O.transpose(-2, -1)))
        return r


class NodeEncoder(nn.Module):

    def __init__(self, cfg):
        super(NodeEncoder, self).__init__()
        
        # variable embeddings
        self.v_encs = nn.ModuleList([
            VarEmbed(cfg) for i in range(cfg.features_d)
        ])
        
        # # temporal attention
        # self.sublayer = SublayerConnection((cfg.embed_v + 1) * cfg.features_d, cfg.dropout)
        # self.temporal_attn = MultiHeadedAttention(cfg.attn_heads, (cfg.embed_v + 1) * cfg.features_d)
        
        self.linear = nn.Linear((cfg.embed_v + 1) * cfg.features_d, cfg.embed_ds)
        
    def forward(self,
                masked_timeseries,
                auxiliary_timeseries,
                observed_mask,
                mask=None):
        
        ## variable embedding
        r_list = [layer(
            masked_timeseries[:,i:(i+1),:],
            auxiliary_timeseries,
            observed_mask[:,i:(i+1),:]
            ) for i,layer in enumerate(self.v_encs)
        ]
        
        r_tens = torch.stack(r_list, dim=-2) # (B, L, V, C)        
        r_tens = r_tens.view(r_tens.shape[0],
                             r_tens.shape[1],
                             r_tens.shape[2]*r_tens.shape[3]) # (B, L, V*C)
        
        ## attention across the time series
        # r_tens = self.sublayer(r_tens,
            # lambda r_tens: self.temporal_attn(r_tens, r_tens, r_tens, mask)
        # )
        r_tens = self.linear(r_tens)
        return r_tens 

tind = 15
data = Data(x=r_tens[:,tind,:],
            edge_index=edges,
            edge_attr=edge_attrs,
            pos=node_pos)

class GraphConvAttn(torch.nn.Module):
    
    def __init__(self, cfg):
        super().__init__()        
        self.conv1 = gnn.GATv2Conv(cfg.embed_ds, 2*cfg.embed_ds,
                                   edge_dim=cfg.edge_features_aux)
        self.conv2 = gnn.GATv2Conv(2*cfg.embed_ds, 2*cfg.embed_ds,
                                   edge_dim=cfg.edge_features_aux)
        self.linear = nn.Linear(2*cfg.embed_ds, cfg.embed_ds)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, data):
        x, e_i, e_a = data.x, data.edge_index, data.edge_attr
        
        x = self.conv1(x, e_i, edge_attr=e_a)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, e_i, edge_attr=e_a)
        x = self.linear(x)

        return x
