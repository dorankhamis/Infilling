import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import xarray as xr
import datetime

from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from skimage.draw import line

from soil_moisture.data_classes.cosmos_data import CosmosMetaData
from soil_moisture.data_loaders import read_one_cosmos_site
from soil_moisture.utils import zeropad_strint
from solar_position_temporal import SolarPositionTemporal
from params import normalisation as norm
from utils import *

hj_base = '/gws/nopw/j04/hydro_jules/data/uk/'
era5_fldr = hj_base + '/driving_data/era5/28km_grid/'
chess_ancil_dir = '/gws/nopw/j04/hydro_jules/data/uk/ancillaries/'
ea_dir = '/home/users/doran/data_dump/EA_rain_gauges/'

TWOPI = 2. * np.pi

def get_EA_gauge_info():
    # load EA rain gauges
    TREAT_RAW_EA_INFO = False
    if TREAT_RAW_EA_INFO is True:    
        EA_site_info = pd.read_csv(ea_dir + 'sites_info.csv')    
        # go through all EA sites dropping NaN regions and
        # updating start/end dates to only mark present regions
        EA_site_info['START_DATE'] = pd.to_datetime(EA_site_info.START_DATE, format='%Y-%m-%d %H:%M:%S', utc=True)
        EA_site_info['END_DATE'] = pd.to_datetime(EA_site_info.END_DATE, format='%Y-%m-%d %H:%M:%S', utc=True)
        for ea_sid in EA_site_info.SITE_ID:
            try:
                sdat = pd.read_csv(f'{ea_dir}/data_15min/{ea_sid}.csv')
            except:
                continue
            sdat = sdat.dropna()
            sdat['DATE_TIME'] = pd.to_datetime(sdat['DATE_TIME'], format='%Y-%m-%d %H:%M:%S', utc=True)
            sind = EA_site_info[EA_site_info['SITE_ID']==ea_sid].index[0]
            EA_site_info.loc[sind, 'START_DATE'] = sdat.DATE_TIME.min()
            EA_site_info.loc[sind, 'END_DATE'] = sdat.DATE_TIME.max()
        EA_site_info.to_csv(ea_dir + 'sites_info_nonan.csv', index=False)
    else:
        EA_site_info = pd.read_csv(ea_dir + 'sites_info_nonan.csv')
        EA_site_info['START_DATE'] = pd.to_datetime(EA_site_info.START_DATE, format='%Y-%m-%d %H:%M:%S', utc=True)
        EA_site_info['END_DATE'] = pd.to_datetime(EA_site_info.END_DATE, format='%Y-%m-%d %H:%M:%S', utc=True)
    EA_site_info = EA_site_info[['SITE_ID', 'LATITUDE', 'LONGITUDE', 'START_DATE', 'END_DATE']]
    return EA_site_info

def find_chess_tile(lat, lon, latlon_ref):
    # assumes equal length lat/lon vectors in latlon_ref
    dist_diff = np.sqrt(np.square(latlon_ref.lat.values - lat) +
                        np.square(latlon_ref.lon.values - lon))
    chesstile_yx = np.where(dist_diff == np.min(dist_diff))
    return chesstile_yx

def tensorise_nbr_data(nbr_data):
    for ii in range(len(nbr_data)):
        # create pseudo batch dim
        nbr_data[ii]['masked_data'] = torch.unsqueeze(
            torch.from_numpy(
                nbr_data[ii]['masked_data'].values.astype(np.float32)
            ).T, 0
        )
        
        nbr_data[ii]['attention_mask'] = torch.unsqueeze(
            torch.from_numpy(
                nbr_data[ii]['attention_mask']
            ), 0
        ).to(torch.bool)
        
        nbr_data[ii]['node_aux'] = torch.unsqueeze(            
            torch.from_numpy(
                nbr_data[ii]['node_aux'].values.astype(np.float32)
            ).T, 0
        )
        
        # create pseudo batch and time dim
        # nbr_data[ii]['node_aux'] = torch.unsqueeze(
            # torch.unsqueeze(
                # torch.tensor(nbr_data[ii]['node_aux']).to(torch.float32),
                # 0
            # ), -1
        # )
        
        nbr_data[ii]['edge_aux'] = torch.unsqueeze(
            torch.unsqueeze(
                torch.tensor(nbr_data[ii]['edge_aux']).to(torch.float32),
                0
            ), -1
        )
    return nbr_data

def load_cloud_cover(time_dat, lat, lon):
    # total cloud cover
    t_prevpoint = time_dat.iloc[0].DATE_TIME - datetime.timedelta(minutes=30)
    t_nextpoint = time_dat.iloc[-1].DATE_TIME + datetime.timedelta(minutes=30)
    t_plus = pd.concat([pd.DataFrame({'DATE_TIME':t_prevpoint}, index=[0]),
                        time_dat,
                        pd.DataFrame({'DATE_TIME':t_nextpoint}, index=[time_dat.shape[0]])], axis=0)
    t_plus = t_plus.reset_index(drop=True)
    uniq_ymd = list(set([(t_plus.DATE_TIME.dt.year.iloc[i],
                          t_plus.DATE_TIME.dt.month.iloc[i],
                          t_plus.DATE_TIME.dt.day.iloc[i]) 
                             for i in range(t_plus.shape[0])]))
    tcc = xr.open_mfdataset(
        [era5_fldr + f'/tcc/era5_{ymd[0]}{zeropad_strint(ymd[1])}{zeropad_strint(ymd[2])}_tcc.nc' 
            for ymd in uniq_ymd])
    tcc_time = pd.to_datetime(tcc.time.values, utc=True)
    tcc_time.name = 'DATE_TIME'
    
    # interp tcc from 4 nearest neighbours
    latlon_grid = np.reshape(np.stack([tcc.lat.values, tcc.lon.values], axis=-1), (tcc.lat.shape[0]*tcc.lat.shape[1], 2))
    neigh = NearestNeighbors(n_neighbors=4).fit(latlon_grid)
    dists, inds = neigh.kneighbors(X=np.array([[site_meta.LATITUDE, site_meta.LONGITUDE]]))
    weights = (1 - (dists / dists.sum(axis=1))).flatten()
    
    X1 = np.where(np.ones((len(tcc.y), len(tcc.x))))
    X1 = np.hstack([X1[0][...,np.newaxis], X1[1][...,np.newaxis]])        
    tcc = np.stack([(tcc.tcc.values[:,X1[inds.flatten()][s][0], X1[inds.flatten()][s][1]] * weights[s]) for s in range(weights.shape[0])]).sum(axis=0) / weights.sum()
    tcc = pd.DataFrame({'cloud_cover':tcc}, index=tcc_time)
    # interpolate to the time_dat grid
    tcc = (t_plus.assign(idx = np.arange(t_plus.shape[0]))
        .set_index('DATE_TIME')
        .merge(tcc, how='left', on='DATE_TIME')
        .drop('idx', axis=1)
        .interpolate(method='linear')
    ).iloc[1:-1] # trim off extra time points
    return tcc

def extract_ireland_shetland_sites(site_meta):
    lat_up = 55.3
    lat_down = 53.0
    lon_right = -5.4
    lon_left = -8.3        
    ireland_sites = site_meta[
        ((site_meta['LATITUDE']>lat_down) & 
         (site_meta['LATITUDE']<lat_up) &
         (site_meta['LONGITUDE']>lon_left) &
         (site_meta['LONGITUDE']<lon_right))
    ]
    shetland_sites = site_meta[site_meta['LATITUDE']>59.5]
    return pd.concat([ireland_sites, shetland_sites], axis=0)    
    
def process_precip_nbr_data(precip_nbr_data, data):
    for i in range(len(precip_nbr_data)):
        precip_nbr_data[i]['real_gap_masks'] = create_real_gap_masks(precip_nbr_data[i]['timeseries'])
        precip_nbr_data[i]['masked_data'] = insert_gaps(precip_nbr_data[i]['timeseries'],
                                                        gap_masks=None,
                                                        real_gap_masks=precip_nbr_data[i]['real_gap_masks'])        
        # for single-variable precip timeseries we don't need
        # to input the 0/1 mask flags as we will be masking all the
        # missing times anyway, so just select PRECIP
        precip_nbr_data[i]['masked_data'] = precip_nbr_data[i]['masked_data'][['PRECIP']]
        
        # create attention mask
        precip_nbr_data[i]['attention_mask'] = np.ones((data.shape[0], precip_nbr_data[i]['timeseries'].shape[0]), dtype=bool)
        mask_inds = np.where(np.stack([precip_nbr_data[i]['real_gap_masks'][v] for v in ['PRECIP']], axis=0).sum(axis=0) == 1)[0]
        precip_nbr_data[i]['attention_mask'][:,mask_inds] = False
    return precip_nbr_data

def process_nonprecip_nbr_data(allvar_nbr_data, data):
    for i in range(len(allvar_nbr_data)):
        allvar_nbr_data[i]['real_gap_masks'] = create_real_gap_masks(allvar_nbr_data[i]['timeseries'])
        allvar_nbr_data[i]['masked_data'] = insert_gaps(allvar_nbr_data[i]['timeseries'],
                                                        gap_masks=None,
                                                        real_gap_masks=allvar_nbr_data[i]['real_gap_masks'])
        allvar_nbr_data[i]['masked_data'] = allvar_nbr_data[i]['masked_data'][allvar_nbr_data[i]['masked_data'].columns.sort_values()]
        
        # create attention mask
        allvar_nbr_data[i]['attention_mask'] = np.ones((data.shape[0], allvar_nbr_data[i]['timeseries'].shape[0]), dtype=bool)
        mask_inds = np.where(np.stack([allvar_nbr_data[i]['real_gap_masks'][v] for v in data.columns], axis=0).sum(axis=0) == len(data.columns))[0]
        allvar_nbr_data[i]['attention_mask'][:,mask_inds] = False
    return allvar_nbr_data

class gap_generator():
    def __init__(self, precip=False, use_nbrs=False):
        self.metadata = CosmosMetaData()
        
        ''' we need to remove the Ireland / Shetland sites as we 
        don't have the ancillary data for them currently '''
        rm_sites = extract_ireland_shetland_sites(self.metadata.site)
        self.metadata.site = self.metadata.site[~self.metadata.site['SITE_ID'].isin(rm_sites['SITE_ID'])]
        
        self.val_sites = self.metadata.site.sample(frac=0.33, random_state=2)['SITE_ID']
        self.sites = self.metadata.site[~self.metadata.site.index.isin(self.val_sites.index)]['SITE_ID']
        if precip:      
            self.use_vars = ['PRECIP']
        else:
            self.use_vars = ['WS', 'LWIN', 'PA', 'RH', 'SWIN', 'TA']
        self.max_period = 250
        self.min_period = 1
        self.spt = SolarPositionTemporal(timezone=0)
        self.use_clouds = False
        self.site_data = {}
        self.precip = precip
        self.use_nbrs = use_nbrs
        
        
        self.ea_site_info = get_EA_gauge_info()
        rm_sites = extract_ireland_shetland_sites(self.ea_site_info)
        self.ea_site_info = self.ea_site_info[~self.ea_site_info['SITE_ID'].isin(rm_sites['SITE_ID'])]
        
        self.attach_bng_coords() # to site metadata        
        
        self.topography = xr.open_dataset(chess_ancil_dir + 'uk_ihdtm_topography+topoindex_1km.nc').load()
        
        ## treat NaNs (i.e. water) in height grid        
        # 1: elev, sea NaNs should be elev==0        
        self.topography.elev.values[np.isnan(self.topography.elev.values)] = 0
        # 2: stdev, sea NaNs should be stdev==0        
        self.topography.stdev.values[np.isnan(self.topography.stdev.values)] = 0
        # 3: slope, sea NaNs should be slope==0
        self.topography.slope.values[np.isnan(self.topography.slope.values)] = 0
        # 4: aspect, sea NaNs are tricky, aspect is "straight up", stick with uniform noise
        asp_mask = np.isnan(self.topography.aspect.values)
        self.topography.aspect.values[asp_mask] =  np.random.uniform(
            low=0, high=360, size=len(np.where(asp_mask)[0])
        )
        
    def attach_bng_coords(self):
        latlon_ref = xr.open_dataset(chess_ancil_dir + 'chess_lat_lon.nc').load()
        ys = []
        xs = []
        for lat, lon in zip(self.ea_site_info.LATITUDE, self.ea_site_info.LONGITUDE):
            y_ind, x_ind = find_chess_tile(lat, lon, latlon_ref)
            ys.append(y_ind[0])
            xs.append(x_ind[0])
        self.ea_site_info['y_ind'] = ys
        self.ea_site_info['x_ind'] = xs
        self.ea_site_info['northing'] = latlon_ref.y.values[np.array(ys)]/1000.
        self.ea_site_info['easting'] = latlon_ref.x.values[np.array(xs)]/1000.

        ys = []
        xs = []
        for lat, lon in zip(self.metadata.site.LATITUDE, self.metadata.site.LONGITUDE):
            y_ind, x_ind = find_chess_tile(lat, lon, latlon_ref)
            ys.append(y_ind[0])
            xs.append(x_ind[0])
        self.metadata.site['y_ind'] = ys
        self.metadata.site['x_ind'] = xs
        self.metadata.site['northing'] = latlon_ref.y.values[np.array(ys)]/1000.
        self.metadata.site['easting'] = latlon_ref.x.values[np.array(xs)]/1000.
        latlon_ref.close()        
    
    def connect_to_neighbours(self, SID, shortrange=10, longrange=60):
        if self.precip:
            ## for precip (short distance)
            targ = self.metadata.site[self.metadata.site['SITE_ID']==SID][['northing','easting']]
            nbrs = NearestNeighbors().fit(self.ea_site_info[self.ea_site_info['SITE_ID']!=SID][['northing','easting']])
            dis_ea, ids_ea = nbrs.radius_neighbors(targ, radius=shortrange)
            sub_ea = self.ea_site_info.iloc[ids_ea[0]]
            
            nbrs = NearestNeighbors().fit(self.metadata.site[self.metadata.site['SITE_ID']!=SID][['northing','easting']])
            dis_cos, ids_cos = nbrs.radius_neighbors(targ, radius=shortrange)
            sub_cos = self.metadata.site.iloc[ids_cos[0]]
        else:
            ## for other vars (long distance)
            nbrs = NearestNeighbors().fit(self.metadata.site[self.metadata.site['SITE_ID']!=SID][['northing','easting']])
            dis_v_cos, ids_v_cos = nbrs.radius_neighbors(targ, radius=longrange)
            #sub_cos_othervars = self.metadata.site.iloc[ids_v_cos[0]]
            sub_cos = self.metadata.site.iloc[ids_v_cos[0]]
            sub_ea = None
        #return {'precip_nbrs':{'EA':sub_ea, 'COSMOS':sub_cos}, 'other_nbrs':{'COSMOS':sub_cos_othervars}}
        return {'EA':sub_ea, 'COSMOS':sub_cos}
    
    def extract_neighbour_data(self, nbr_dict, home_data, SID):
        home_meta = self.metadata.site[self.metadata.site['SITE_ID']==SID]
        
        if self.precip:
            ## precip neighbours
            precip_nbr_data = []
            #for sid in nbr_dict['precip_nbrs']['EA'].SITE_ID:
            for sid in nbr_dict['EA'].SITE_ID:
                ## get time series data
                try:
                    sdat = pd.read_csv(f'{ea_dir}/data_15min/{sid}.csv')
                except:
                    continue        
                    
                sdat['DATE_TIME'] = pd.to_datetime(sdat['DATE_TIME'], format='%Y-%m-%d %H:%M:%S', utc=True)
                sdat = (sdat.set_index('DATE_TIME')
                    .rename({'PRECIPITATION':'PRECIP'}, axis=1)
                    .resample('30T')
                    .sum()
                )
                # trim to desired period, keeping NaNs for missing times
                sdat = (home_data.reset_index()[['DATE_TIME']]
                    .merge(sdat.reset_index(), how='left', on='DATE_TIME')
                    .set_index('DATE_TIME')
                )

                # if there is no data at the nbr node, skip
                if sdat.dropna().shape[0]==0:
                    continue
                    
                precip_nbr_data.append({'site_id':sid})        
                precip_nbr_data[-1]['timeseries'] = sdat
                
                ## get nbr node and connecting edge metadata        
                node_aux, edge_aux = self.get_node_and_edge_aux_data(
                    sdat,
                    nbr_dict['precip_nbrs']['EA'],
                    home_meta,
                    sid
                )
                precip_nbr_data[-1]['node_aux'] = node_aux
                precip_nbr_data[-1]['edge_aux'] = edge_aux            

            #for sid in nbr_dict['precip_nbrs']['COSMOS'].SITE_ID:
            for sid in nbr_dict['COSMOS'].SITE_ID:
                ## get time series data
                if sid in self.site_data.keys():
                    sdat = self.site_data[sid].copy()
                else:
                    sdat = read_one_cosmos_site(sid)
                    sdat = sdat.subhourly[self.use_vars]
                    self.site_data[sid] = sdat.copy()
                
                # pull out just precip
                sdat = sdat[['PRECIP']]
                
                # trim to desired period, keeping NaNs for missing times
                sdat = (home_data.reset_index()[['DATE_TIME']]
                    .merge(sdat.reset_index(), how='left', on='DATE_TIME')
                    .set_index('DATE_TIME')
                )

                # if there is no data at the nbr node, skip
                if sdat.dropna().shape[0]==0:
                    continue
                    
                precip_nbr_data.append({'site_id':sid})        
                precip_nbr_data[-1]['timeseries'] = sdat
                
                ## get nbr node and connecting edge metadata        
                node_aux, edge_aux = self.get_node_and_edge_aux_data(
                    sdat,
                    nbr_dict['precip_nbrs']['COSMOS'],
                    home_meta,
                    sid
                )
                precip_nbr_data[-1]['node_aux'] = node_aux
                precip_nbr_data[-1]['edge_aux'] = edge_aux
            return precip_nbr_data
        else:
            ## all var neighbours
            allvar_nbr_data = []
            #for sid in nbr_dict['other_nbrs']['COSMOS'].SITE_ID:
            for sid in nbr_dict['COSMOS'].SITE_ID:
                ## get time series data
                if sid in self.site_data.keys():
                    sdat = self.site_data[sid].copy()
                else:
                    sdat = read_one_cosmos_site(sid)
                    sdat = sdat.subhourly[self.use_vars]
                    self.site_data[sid] = sdat.copy()
                
                # trim to desired period, keeping NaNs for missing times
                sdat = (home_data.reset_index()[['DATE_TIME']]
                    .merge(sdat.reset_index(), how='left', on='DATE_TIME')
                    .set_index('DATE_TIME')
                )

                # if there is no data at the nbr node, skip
                if sdat.dropna(how='all').shape[0]==0:
                    continue
                    
                allvar_nbr_data.append({'site_id':sid})        
                allvar_nbr_data[-1]['timeseries'] = sdat
                
                ## get nbr node and connecting edge metadata        
                node_aux, edge_aux = self.get_node_and_edge_aux_data(
                    sdat,
                    nbr_dict['other_nbrs']['COSMOS'],
                    home_meta,
                    sid
                )
                allvar_nbr_data[-1]['node_aux'] = node_aux
                allvar_nbr_data[-1]['edge_aux'] = edge_aux
            return allvar_nbr_data
        #return precip_nbr_data, allvar_nbr_data

    def get_node_and_edge_aux_data(self, nbr_timeseries, nbr_dict_meta, home_meta, site_id):
        ## get nbr node metadata        
        nbr_meta = nbr_dict_meta[nbr_dict_meta['SITE_ID']==site_id]
        #node_aux = [
        #    nbr_meta.LATITUDE.values[0] / norm.lat_norm,
        #    nbr_meta.LONGITUDE.values[0] / norm.lon_norm,
        #    self.topography.elev.values[int(nbr_meta.y_ind), int(nbr_meta.x_ind)] / norm.elev_norm
        #]
        elev = nbr_meta.ALTITUDE.values[0] if 'ALTITUDE' in nbr_meta.columns else self.topography.elev.values[int(nbr_meta.y_ind), int(nbr_meta.x_ind)]
        node_aux = self.create_auxiliary_data(
            nbr_timeseries,
            nbr_meta.LATITUDE.values[0],
            nbr_meta.LONGITUDE.values[0],
            elev
        )
        
        ## get edge metadata between home and nbr         
        pix_i, pix_j = line(int(home_meta.y_ind), int(home_meta.x_ind), int(nbr_meta.y_ind), int(nbr_meta.x_ind))
        elevs = self.topography.elev.values[pix_i, pix_j]
        std_elevs = self.topography.stdev.values[pix_i, pix_j]
        slopes = self.topography.slope.values[pix_i, pix_j]
        aspect = self.topography.aspect.values[pix_i, pix_j]
        edge_aux = []
        edge_aux += [f(elevs)/norm.elev_norm for f in [np.nanmean, np.nanmin, np.nanmax]]
        edge_aux += [f(std_elevs)/norm.std_elev_norm for f in [np.nanmean, np.nanmin, np.nanmax]]
        edge_aux += [f(slopes)/norm.slope_norm for f in [np.nanmean, np.nanmin, np.nanmax]]
        edge_aux += [np.sin(np.deg2rad(f(aspect))) for f in [np.nanmean, np.nanmin, np.nanmax]]
        edge_aux += [np.cos(np.deg2rad(f(aspect))) for f in [np.nanmean, np.nanmin, np.nanmax]]
        # add a distance measurement from length of line?
        distance = dist_lat_lon(
            home_meta.LATITUDE.values[0], home_meta.LONGITUDE.values[0],
            nbr_meta.LATITUDE.values[0], nbr_meta.LONGITUDE.values[0],
            R = 10.
        )
        edge_aux += [distance]
        # and the initial bearing between home and nbr
        # bearing_theta = init_bearing(
            # home_meta.LATITUDE.values[0], home_meta.LONGITUDE.values[0],
            # nbr_meta.LATITUDE.values[0], nbr_meta.LONGITUDE.values[0]
        # )
        # edge_aux += [bearing_theta / 3.] # normalise as goes between -pi, pi
        return node_aux, edge_aux
    
    def get_batch(self, batch_size,
                  const_l = True,
                  batch_type = 'train',
                  min_gap_logprob = -1.5,
                  mean_gap_length = 5,
                  gap_length_sd = 2.5,
                  shortrange = 10,
                  longrange = 60):

        inputs = []
        aux_inputs = []
        targets = []
        our_gaps = []
        existing_gaps = []
        batch_metadata = []
        #batch_precip_nbrs = []
        batch_allvar_nbrs = []
        
        if const_l:
            # keep data length the same within a batch so we 
            # don't have to pad and mix very long with very short
            l = np.random.randint(self.min_period, self.max_period+1)
        else:
            l = None
        
        for b in range(batch_size):
            # p_nbrs, av_nbrs
            m_data, data, gaps, real_gaps, meta, aux, nbrs = self.get_sample(
                l=l,
                samp_type=batch_type,
                min_gap_logprob=min_gap_logprob,
                mean_gap_length=mean_gap_length,
                gap_length_sd=gap_length_sd,                 
                shortrange=shortrange,
                longrange=longrange
            )
            inputs.append(torch.from_numpy(m_data.values.astype(np.float32)).T)
            aux_inputs.append(torch.from_numpy(aux.values.astype(np.float32)).T)
            targets.append(torch.from_numpy(data.values.astype(np.float32)).T)
            our_gaps.append(torch.from_numpy(gaps))
            existing_gaps.append(torch.from_numpy(real_gaps))
            batch_metadata.append(meta)
            #batch_precip_nbrs.append(p_nbrs)
            batch_allvar_nbrs.append(nbrs)
            
        inputs = torch.stack(inputs, dim=0) # B, C, T
        aux_inputs = torch.stack(aux_inputs, dim=0) # B, C, T
        targets = torch.stack(targets, dim=0) # B, C, T
        our_gaps = torch.stack(our_gaps, dim=0) # B, C, T
        existing_gaps = torch.stack(existing_gaps, dim=0) # B, C, T
        
        if self.use_nbrs:
            # tensorise neighbour data
            for b in range(batch_size):
                #batch_precip_nbrs[b] = tensorise_nbr_data(batch_precip_nbrs[b])
                batch_allvar_nbrs[b] = tensorise_nbr_data(batch_allvar_nbrs[b])
        else:
            batch_allvar_nbrs = None
        
        return {'inputs':inputs, 'aux_inputs':aux_inputs,
                'targets':targets, 'our_gaps':our_gaps,
                'existing_gaps':existing_gaps, 'metadata':batch_metadata,
                'allvar_nbrs':batch_allvar_nbrs} #'precip_nbrs':batch_precip_nbrs}
    
    def normalise(self, data):
        if 'TA' in data.columns:
            data['TA'] = (data['TA'] - 273.15 - norm.temp_mu) / norm.temp_sd
        if 'PA' in data.columns:
            data['PA'] = (data['PA'] - norm.p_mu) / norm.p_sd
        if 'LWIN' in data.columns:
            data['LWIN'] = (data['LWIN'] - norm.lwin_mu) / norm.lwin_sd
        if 'RH' in data.columns:
            data['RH'] = (data['RH'] - norm.rh_mu) / norm.rh_sd
        if 'WS' in data.columns:
            data['WS'] = (data['WS'] - norm.ws_mu) / norm.ws_sd
        if 'SWIN' in data.columns:
            data['SWIN'] = data['SWIN'] / norm.swin_norm
        if 'PRECIP' in data.columns:
            data['PRECIP'] = data['PRECIP'] / norm.precip_norm
        return data
    
    def create_auxiliary_data(self, data, lat, lon, elev):
        ## create auxiliary data        
        time_dat = data.reset_index()[['DATE_TIME']]
        # time of day
        sin_tod = np.sin((time_dat.DATE_TIME.dt.hour + time_dat.DATE_TIME.dt.minute / 60.) / 24. * TWOPI)
        cos_tod = np.cos((time_dat.DATE_TIME.dt.hour + time_dat.DATE_TIME.dt.minute / 60.) / 24. * TWOPI)
        # time of year
        sin_toy = np.sin((
            (time_dat.DATE_TIME.dt.dayofyear-1) * 24 +
            time_dat.DATE_TIME.dt.hour +
            time_dat.DATE_TIME.dt.minute / 60.) / (365*24) * TWOPI
        )
        cos_toy = np.cos((
            (time_dat.DATE_TIME.dt.dayofyear-1) * 24 + 
            time_dat.DATE_TIME.dt.hour +
            time_dat.DATE_TIME.dt.minute / 60.) / (365*24) * TWOPI
        )
        encoded_time = pd.DataFrame(
            {'sin_tod':sin_tod.values,
             'cos_tod':cos_tod.values,
             'sin_toy':sin_toy.values,
             'cos_toy':cos_toy.values},
            index=time_dat.DATE_TIME
        )
                
        # sun elevation, azimuth
        solar_angles = self.spt.calc_solar_angles(time_dat.DATE_TIME, lat, lon)
        solar_angles['solar_elevation'] /= 90.
        solar_angles['sin_solar_azimuth'] = np.sin(np.deg2rad(solar_angles['solar_azimuth_angle']))
        solar_angles['cos_solar_azimuth'] = np.cos(np.deg2rad(solar_angles['solar_azimuth_angle']))
        solar_angles = solar_angles.drop('solar_azimuth_angle', axis=1)
        
        if self.use_clouds:
            tcc = load_cloud_cover(time_dat, lat, lon)
            auxiliary_timeseries = pd.concat([solar_angles, 
                                              tcc,
                                              encoded_time], axis=1)            
        else:
            auxiliary_timeseries = pd.concat([solar_angles,                                               
                                              encoded_time], axis=1)

        auxiliary_timeseries = auxiliary_timeseries.assign(
            lat = lat / norm.lat_norm,
            lon = lon / norm.lon_norm,
            elev = elev / norm.elev_norm
        )
        return auxiliary_timeseries
    
    def get_sample(self, l=None,
                   samp_type='train',
                   min_gap_logprob = -1.5,
                   mean_gap_length = 5,
                   gap_length_sd = 2.5,
                   shortrange = 10,
                   longrange = 60):
        ## load and subset
        if l is None:
            l = np.random.randint(self.min_period, self.max_period+1)
            
        data = np.array([])
        while data.shape[0]<l:
            if samp_type=='train':
                SID = np.random.choice(self.sites)
            elif samp_type=='val':
                SID = np.random.choice(self.val_sites)
            if SID in self.site_data.keys():
                data = self.site_data[SID].copy()
            else:
                data = read_one_cosmos_site(SID)
                data = data.subhourly[self.use_vars]
                self.site_data[SID] = data.copy()            
            data = data.sort_index()
        
        ## randomly select a subset
        dt0 = np.random.randint(0, data.shape[0]-l-1)
        data = data.iloc[dt0:(dt0+l)]
        
        nbr_data = None
        if self.use_nbrs:
            ## find neighbours for cross-attention
            nbr_dict = self.connect_to_neighbours(
                SID, shortrange=shortrange, longrange=longrange
            )
            
            ## get neighbour data
            #precip_nbr_data, allvar_nbr_data = self.extract_neighbour_data(nbr_dict, data, SID)
            nbr_data = self.extract_neighbour_data(nbr_dict, data, SID)

        ## normalise
        data = self.normalise(data)
        if self.use_nbrs:
            # for i in range(len(precip_nbr_data)):
                # precip_nbr_data[i]['timeseries'] = self.normalise(precip_nbr_data[i]['timeseries'])
            # for i in range(len(allvar_nbr_data)):
                # allvar_nbr_data[i]['timeseries'] = self.normalise(allvar_nbr_data[i]['timeseries'])
            for i in range(len(nbr_data)):
                nbr_data[i]['timeseries'] = self.normalise(nbr_data[i]['timeseries'])
        
        ## create and apply gap masks
        gap_masks, real_gap_masks = create_gap_masks(
            data,
            min_gap_logprob = min_gap_logprob,
            mean_gap_length = mean_gap_length,
            gap_length_sd   = gap_length_sd
        )
        masked_data = insert_gaps(data, gap_masks=gap_masks, real_gap_masks=real_gap_masks)
        
        masked_data = masked_data[masked_data.columns.sort_values()]
        data = data[data.columns.sort_values()]
        
        if self.use_nbrs:
            ## find real gap masks for neighbour time series
            ## where neighbours have all vars as gaps, mask from attention
            if self.precip:
                process_precip_nbr_data(nbr_data, data)
            else:
                process_nonprecip_nbr_data(nbr_data, data)
       
        ## create auxiliary data
        home_site_meta = self.metadata.site.set_index('SITE_ID').loc[SID]
        aux_timeseries = self.create_auxiliary_data(
            data,
            home_site_meta.LATITUDE,
            home_site_meta.LONGITUDE,
            home_site_meta.ALTITUDE
        )

        ## sample metadata
        metadata = {}
        metadata['tot_our_gaps'] = {v:gap_masks[v].sum() for v in gap_masks.keys()}
        metadata['tot_real_gaps'] = {v:real_gap_masks[v].sum() for v in real_gap_masks.keys()}
        metadata['start_date'] = data.index[0]
        metadata['end_date'] = data.index[-1]
        metadata['site'] = SID
        metadata['var_order'] = list(data.columns)
        metadata['aux_var_order'] = list(aux_timeseries.columns)
        
        # arrange masks into (var, time) arrays
        gap_masks = np.stack([gap_masks[v] for v in metadata['var_order']], axis=0)
        real_gap_masks = np.stack([real_gap_masks[v] for v in metadata['var_order']], axis=0)
                
        return (masked_data, data, gap_masks, real_gap_masks, metadata,
                aux_timeseries, nbr_data) #precip_nbr_data, allvar_nbr_data)


if __name__=="__main__":
    dg = gap_generator(use_nbrs=True)
    
    (masked_data, data, gap_masks, real_gap_masks, metadata, # precip_nbr_data, allvar_nbr_data
        aux_timeseries, nbr_data) = dg.get_sample(
            l = 50,
            samp_type = 'train',
            min_gap_logprob = -0.5,
            mean_gap_length = 30,
            gap_length_sd = 2.5,
            shortrange = 10,
            longrange = 60
    )
