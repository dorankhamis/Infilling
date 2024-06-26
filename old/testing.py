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
chess_ancil_dir = hj_base + '/ancillaries/'
home_data_dir = '/home/users/doran/data_dump/'
midas_fldr = home_data_dir + '/MetOffice/midas_data/'
ea_fldr = home_data_dir + '/EA_rain_gauges/'

TWOPI = 2. * np.pi

def create_gap_masks(df, min_gap_logprob = -1.5,
                     mean_gap_length = 5, gap_length_sd = 2.5):
    # randomly create gaps in one/some/all variables
    mask_probs = np.logspace(0, min_gap_logprob, df.shape[1])
    mask_binary = np.random.binomial(1, p=mask_probs).astype(bool)
    use_vars = list(df.columns).copy()
    np.random.shuffle(use_vars)
    gap_masks = {} # where we are creating gaps
    real_gap_masks = {} # where gaps already exist
    unmasked = np.zeros(df.shape[0], dtype=bool)
    for i, var in enumerate(use_vars):
        real_gap_masks[var] = df[var].isna().values
        if not mask_binary[i]:
            gap_masks[var] = unmasked.copy()
        else:
            thismask = unmasked.copy()
            gap_start = np.random.randint(0, df.shape[0])
            #gap_len = np.random.randint(1, df.shape[0]+1)
            gap_len = int(np.clip(np.random.normal(mean_gap_length, gap_length_sd),
                                  a_min=1, a_max=df.shape[0]-gap_start))
            real_gap_len = len(thismask[gap_start:(gap_start+gap_len)])
            thismask[gap_start:(gap_start+gap_len)] = True
            gap_masks[var] = thismask
    return gap_masks, real_gap_masks

def create_real_gap_masks(df):
    real_gap_masks = {} # where gaps already exist    
    for i, var in enumerate(df.columns):
        real_gap_masks[var] = df[var].isna().values
    return real_gap_masks

def insert_gaps(df, gap_masks=None, real_gap_masks=None):
    # apply gap masks to create masked input data
    use_vars = list(df.columns).copy()
    for i, var in enumerate(use_vars):
        this_dat = df[[var]].copy()
        # mask imagined gaps
        if gap_masks is None:
            this_dat[var+'_d'] = np.ones(this_dat.shape[0], dtype=np.int32)
        else:
            this_dat.loc[gap_masks[var], var] = 0
            this_dat[var+'_d'] = (~gap_masks[var]).astype(np.int32)
        # also mask real gaps
        if not (real_gap_masks is None):
            this_dat.loc[real_gap_masks[var], var] = 0
            this_dat.loc[real_gap_masks[var], var+'_d'] = 0
        if i==0:
            masked_data = this_dat.copy()
        else:
            masked_data = pd.concat([masked_data, this_dat], axis=1)
    return masked_data

def get_EA_gauge_info():
    # load EA rain gauges
    TREAT_RAW_EA_INFO = False
    if TREAT_RAW_EA_INFO is True:    
        EA_site_info = pd.read_csv(ea_fldr + 'sites_info.csv')    
        # go through all EA sites dropping NaN regions and
        # updating start/end dates to only mark present regions
        EA_site_info['START_DATE'] = pd.to_datetime(EA_site_info.START_DATE, format='%Y-%m-%d %H:%M:%S', utc=True)
        EA_site_info['END_DATE'] = pd.to_datetime(EA_site_info.END_DATE, format='%Y-%m-%d %H:%M:%S', utc=True)
        for ea_sid in EA_site_info.SITE_ID:
            try:
                sdat = pd.read_csv(f'{ea_fldr}/data_15min/{ea_sid}.csv')
            except:
                continue
            sdat = sdat.dropna()
            sdat['DATE_TIME'] = pd.to_datetime(sdat['DATE_TIME'], format='%Y-%m-%d %H:%M:%S', utc=True)
            sind = EA_site_info[EA_site_info['SITE_ID']==ea_sid].index[0]
            EA_site_info.loc[sind, 'START_DATE'] = sdat.DATE_TIME.min()
            EA_site_info.loc[sind, 'END_DATE'] = sdat.DATE_TIME.max()
        EA_site_info.to_csv(ea_fldr + 'sites_info_nonan.csv', index=False)
    else:
        EA_site_info = pd.read_csv(ea_fldr + 'sites_info_nonan.csv')
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

def provide_cosmos_met_data(metadata, met_vars, sites=None, missing_val = -9999.0, forcenew=False):
    Path(home_data_dir+'/met_pickles/').mkdir(exist_ok=True, parents=True)
    fname = home_data_dir+'/met_pickles/cosmos_site_met.pkl'
    if forcenew is False and Path(fname).is_file():
        with open(fname, 'rb') as fo:
            metdat = pickle.load(fo)
        met_data = metdat.pop('data')        
    else:        
        met_data = {}
        if sites is None: sites = metadata.site['SITE_ID']
        # remove wind components and insert wind direction        
        for SID in sites:            
            data = read_one_cosmos_site_met(SID, missing_val = missing_val)            
            met_data[SID] = data.subhourly[et_vars]
        metdat = dict(data=met_data)
        with open(fname, 'wb') as fs:
            pickle.dump(metdat, fs)
    return met_data

def provide_midas_met_data(metadata, met_vars, sites=None, forcenew=False):
    Path(home_data_dir+'/met_pickles/').mkdir(exist_ok=True, parents=True)
    fname = home_data_dir+'/met_pickles/midas_site_met.pkl'
    if forcenew is False and Path(fname).is_file():
        with open(fname, 'rb') as fo:
            metdat = pickle.load(fo)
        met_data = metdat.pop('data')
    else:        
        met_data = {}
        if sites is None: sites = metadata['SITE_ID']       
        for SID in sites:
            if Path(midas_fldr + f'/{SID}.parquet').exists():
                data = pd.read_parquet(midas_fldr + f'/{SID}.parquet')
                try:                    
                    met_data[SID] = data[met_vars]
                except:
                    print(f'Met vars missing from {SID}')
        metdat = dict(data=met_data)
        with open(fname, 'wb') as fs:
            pickle.dump(metdat, fs)
    return met_data

def load_midas_elevation():
    # load midas site elevation
    midas_meta = pd.read_csv('/badc/ukmo-midas/metadata/SRCE/SRCE.DATA.COMMAS_REMOVED')
    mid_cols = ['SRC_ID', 'SRC_NAME','HIGH_PRCN_LAT',# - Latitude to 0.001 deg
                'HIGH_PRCN_LON',# - Longitude to 0.001 deg
                'LOC_GEOG_AREA_ID', 'REC_ST_IND', 'SRC_BGN_DATE',
                'SRC_TYPE','GRID_REF_TYPE', 'EAST_GRID_REF',
                'NORTH_GRID_REF','HYDR_AREA_ID', 'POST_CODE', 
                'SRC_END_DATE', 'ELEVATION', # - Metres
                'WMO_REGION_CODE', 'PARENT_SRC_ID', 'ZONE_TIME',
                'DRAINAGE_STREAM_ID', 'SRC_UPD_DATE',
                'MTCE_CTRE_CODE', 'PLACE_ID',
                'LAT_WGs84',# - SRC higher precision latitude to 5 dp - please see note below
                'LONG_WGS84',# - SRC higer precision longitude to 5dp - please see note below
                'SRC_GUID', 'SRC_GEOM', 'SRC_LOCATION_TYPE']
    midas_meta.columns = mid_cols[:24]
    return midas_meta[['SRC_ID', 'SRC_NAME', 'ELEVATION']]

class gap_generator():
    def __init__(self, shortrange=15, longrange=75):
        self.use_vars = ['WS', 'LWIN', 'PA', 'RH', 'SWIN', 'TA', 'PRECIP']
        self.max_period = 250
        self.min_period = 1
        self.spt = SolarPositionTemporal(timezone=0)        
        self.site_data = {}        
                
        # load child pixels on 1km BNG grid labelled by parent pixel IDs
        self.fine_grid = xr.open_dataset(home_data_dir + '/bng_grids/bng_1km_28km_parent_pixel_ids.nc')
        self.fine_grid = self.fine_grid.load()
        
        # load topography and treat NaNs
        self.topography = xr.open_dataset(chess_ancil_dir + 'uk_ihdtm_topography+topoindex_1km.nc').load()
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

        ## deal with site data
        # load site metadata/locations
        self.site_metadata = CosmosMetaData()
        self.site_metadata = self.site_metadata.site.assign(source = 'COSMOS')
        
        # load EA (SEPA?, NRW?) site data for precipitation
        ea_site_meta = (pd.read_csv(ea_fldr + 'sites_info.csv')
            .drop(['ELEVATION', 'PATH', 'START_DATE', 'END_DATE'], axis=1)
        )
        ea_site_elev = pd.read_csv(ea_fldr + '/site_elevation.csv')           
        ea_site_meta['SITE_ID'] = 'EA_' + ea_site_meta.SITE_ID.values.astype(str).astype(object)
        ea_site_meta = ea_site_meta.merge(ea_site_elev, on=['SITE_ID', 'LATITUDE', 'LONGITUDE'])
        ea_site_meta['ALTITUDE'] = ea_site_meta.ALTITUDE.astype(np.float32)
        ea_site_meta = ea_site_meta.loc[~(abs(ea_site_elev.ALTITUDE) > 2000)]

        # add to master meta
        self.site_metadata = pd.concat([self.site_metadata, ea_site_meta.assign(source = 'EA')], axis=0)
        self.site_metadata = self.site_metadata.reset_index().drop('index', axis=1)
        
        # numericise site elevation
        self.site_metadata['ALTITUDE'] = self.site_metadata.ALTITUDE.astype(np.float32)
                
        # remove sites for which we don't have static data (ireland, shetland)
        rm_sites = extract_ireland_shetland_sites(self.site_metadata)
        self.site_metadata = self.site_metadata[~self.site_metadata['SITE_ID'].isin(rm_sites['SITE_ID'])]
        
        # attach northing / easting to site metadata
        self.attach_bng_coords()

        # trim down columns
        self.site_metadata = self.site_metadata[
            ['SITE_NAME', 'SITE_ID', 'source',
             'LATITUDE', 'LONGITUDE', 'ALTITUDE',
             'y_ind', 'x_ind', 'northing', 'easting']
         ]
        
        # run knn between all sites with short and long range
        nbrs = NearestNeighbors().fit(self.site_metadata[['northing','easting']])
        self.dis_sr, self.ids_sr = nbrs.radius_neighbors(self.site_metadata[['northing','easting']], radius=shortrange)
        self.dis_lr, self.ids_lr = nbrs.radius_neighbors(self.site_metadata[['northing','easting']], radius=longrange)
        # note the ids are not orderd by distances and contain the target site itself
        
        self.sites = list(self.site_metadata.loc[self.site_metadata['source']=='COSMOS'].SITE_ID)

    def attach_bng_coords(self):
        latlon_ref = self.fine_grid
        ys = []
        xs = []
        for lat, lon in zip(self.site_metadata.LATITUDE, self.site_metadata.LONGITUDE):
            y_ind, x_ind = find_chess_tile(lat, lon, self.fine_grid)
            ys.append(y_ind[0])
            xs.append(x_ind[0])
        self.site_metadata['y_ind'] = ys
        self.site_metadata['x_ind'] = xs
        self.site_metadata['northing'] = self.fine_grid.y.values[np.array(ys)]/1000.
        self.site_metadata['easting'] = self.fine_grid.x.values[np.array(xs)]/1000.     
    
        
    def read_site_data(self, sid, source):
        if source=='EA':
            dat = pd.read_csv(f'{ea_fldr}/data_15min/{sid.replace("EA_","")}.csv')
            dat['DATE_TIME'] = pd.to_datetime(dat['DATE_TIME'], utc=True)
            return dat.set_index('DATE_TIME')
        if source=='COSMOS':
            if sid in self.site_data.keys():
                return self.site_data[sid].copy()
            else:
                data = read_one_cosmos_site(sid).subhourly[self.use_vars]
                self.site_data[sid] = data.copy()      
                return data
    
    
#########################

'''

For proof of concept, load up chess met data and subset 
pixels to act as met sites. introduce gaps and infill from 
neighbouring sites and neighbouring time points.
implement the nbr attention with key qualification based on
network edge traversal.  

'''
    
    

    
    l = 48
    samp_type = 'train'
    min_gap_logprob = -1.5
    mean_gap_length = 15
    gap_length_sd = 5
    shortrange = 20
    longrange = 75
    ## load and subset
    if l is None:
        l = np.random.randint(self.min_period, self.max_period+1)
        
    data = np.array([])
    while data.shape[0]<l:
        # if samp_type=='train':
            # SID = np.random.choice(self.sites)
        # elif samp_type=='val':
            # SID = np.random.choice(self.val_sites)
        SID = np.random.choice(self.sites)
        data = self.read_site_data(SID, 'COSMOS')           
        data = data.sort_index()

    ## randomly select a subset
    dt0 = np.random.randint(0, data.shape[0]-l-1)
    data = data.iloc[dt0:(dt0+l)]

    ## find neighbours for cross-attention
    targ_ind = self.site_metadata.loc[self.site_metadata['SITE_ID']==SID].index[0]
    sr_nbrs = np.setdiff1d(self.ids_sr[targ_ind], [targ_ind])
    lr_nbrs = np.setdiff1d(self.ids_lr[targ_ind], [targ_ind])
    
    sr_nbr_info = self.site_metadata.iloc[sr_nbrs]
    lr_nbr_info = self.site_metadata.iloc[lr_nbrs]
    lr_nbr_info = lr_nbr_info.loc[lr_nbr_info['source']!='EA'] # remove EA as don't want precip at long range

    sr_nbr_data = {}
    for i in range(sr_nbr_info.shape[0]):
        sid = sr_nbr_info.iloc[i].SITE_ID
        source = sr_nbr_info.iloc[i].source
        sdat = self.read_site_data(sid, source)
        sdat = (data.assign(dummy=1)[['dummy']]
            .merge(sdat, how='left', on='DATE_TIME')
            .drop('dummy', axis=1)
        )
        sr_nbr_data[sid] = sdat
        
    lr_nbr_data = {}    
    for i in range(lr_nbr_info.shape[0]):
        sid = lr_nbr_info.iloc[i].SITE_ID
        source = lr_nbr_info.iloc[i].source
        sdat = self.read_site_data(sid, source)
        sdat = (data.assign(dummy=1)[['dummy']]
            .merge(sdat, how='left', on='DATE_TIME')
            .drop('dummy', axis=1)
        )
        lr_nbr_data[sid] = sdat
        
        
    
    

    
    def extract_neighbour_data(self, nbr_dict, home_data, SID):
        home_meta = self.metadata.site[self.metadata.site['SITE_ID']==SID]
        
        if self.precip:
            ## precip neighbours
            precip_nbr_data = []
            #for sid in nbr_dict['precip_nbrs']['EA'].SITE_ID:
            for sid in nbr_dict['EA'].SITE_ID:
                ## get time series data
                try:
                    sdat = pd.read_csv(f'{ea_fldr}/data_15min/{sid}.csv')
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

if False:
    if __name__=="__main__":
        dg = gap_generator()
        
        
        l = 48
        samp_type = 'train'
        min_gap_logprob = -1.5
        mean_gap_length = 30
        gap_length_sd = 2.5
        shortrange = 10
        longrange = 60
        ## load and subset
        if l is None:
            l = np.random.randint(dg.min_period, dg.max_period+1)
            
        data = np.array([])
        while data.shape[0]<l:
            if samp_type=='train':
                SID = np.random.choice(dg.sites)
            elif samp_type=='val':
                SID = np.random.choice(dg.val_sites)
            if SID in dg.site_data.keys():
                data = dg.site_data[SID].copy()
            else:
                data = read_one_cosmos_site(SID)
                data = data.subhourly[dg.use_vars]
                dg.site_data[SID] = data.copy()            
            data = data.sort_index()

        ## randomly select a subset
        dt0 = np.random.randint(0, data.shape[0]-l-1)
        data = data.iloc[dt0:(dt0+l)]

        ## find neighbours for cross-attention
        # nbr_dict = dg.connect_to_neighbours(
            # SID, shortrange=shortrange, longrange=longrange
        # )
        #def connect_to_neighbours(self, SID, shortrange=10, longrange=60):

        def find_nbr_site_info(targ, SID, site_info, radius):
            info_sans_sid = site_info[site_info['SITE_ID']!=SID]
            nbrs = NearestNeighbors().fit(info_sans_sid[['northing','easting']])
            dis, ids = nbrs.radius_neighbors(targ, radius=radius)
            out_info = info_sans_sid.iloc[ids[0]]
            return out_info
            
        targ = dg.metadata.site[dg.metadata.site['SITE_ID']==SID][['northing','easting']]
        shortrange = 25
        longrange = 80
        # for precip (short distance)
        precip_ea_nbrs = find_nbr_site_info(targ, SID, dg.ea_site_info, shortrange)
        precip_cosmos_nbrs = find_nbr_site_info(targ, SID, dg.metadata.site, shortrange)
        
        # for precip (short distance)    
        othervars_cosmos_nbrs = find_nbr_site_info(targ, SID, dg.metadata.site, longrange)
        
            
        ## get neighbour data
        #precip_nbr_data, allvar_nbr_data = dg.extract_neighbour_data(nbr_dict, data, SID)
        #nbr_data = dg.extract_neighbour_data(nbr_dict, data, SID)

        #def extract_neighbour_data(self, nbr_dict, home_data, SID):
        home_meta = self.metadata.site[self.metadata.site['SITE_ID']==SID]
        
        if self.precip:
            ## precip neighbours
            precip_nbr_data = []
            #for sid in nbr_dict['precip_nbrs']['EA'].SITE_ID:
            for sid in nbr_dict['EA'].SITE_ID:
                ## get time series data
                try:
                    sdat = pd.read_csv(f'{ea_fldr}/data_15min/{sid}.csv')
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
            #return precip_nbr_data
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
            #return allvar_nbr_data
        #return precip_nbr_data, allvar_nbr_data

    ##################



        ## normalise
        data = dg.normalise(data)
        if dg.use_nbrs:
            # for i in range(len(precip_nbr_data)):
                # precip_nbr_data[i]['timeseries'] = dg.normalise(precip_nbr_data[i]['timeseries'])
            # for i in range(len(allvar_nbr_data)):
                # allvar_nbr_data[i]['timeseries'] = dg.normalise(allvar_nbr_data[i]['timeseries'])
            for i in range(len(nbr_data)):
                nbr_data[i]['timeseries'] = dg.normalise(nbr_data[i]['timeseries'])

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

        if dg.use_nbrs:
            ## find real gap masks for neighbour time series
            ## where neighbours have all vars as gaps, mask from attention
            if dg.precip:
                process_precip_nbr_data(nbr_data, data)
            else:
                process_nonprecip_nbr_data(nbr_data, data)

        ## create auxiliary data
        home_site_meta = dg.metadata.site.set_index('SITE_ID').loc[SID]
        aux_timeseries = dg.create_auxiliary_data(
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
                






        
        
        
        
        
        
        
        
        
        
        
