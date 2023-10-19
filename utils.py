import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import pkbar
import shutil
from pathlib import Path

SQRT2PI = np.sqrt(2 * np.pi)
EPS = 1e-9

def count_parameters(model, trainable=True):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters() if not p.requires_grad)

def dist_lat_lon(lat1, lon1, lat2, lon2, R = 10.):
    # Haversine formula
    # R = 6371 # km, Earth's radius, used as a scale factor here
    a = (np.sin(np.deg2rad(lat1 - lat2) / 2.)**2 + 
        np.cos(np.deg2rad(lat1)) * np.cos(np.deg2rad(lat2)) * 
        np.sin(np.deg2rad(lon1 - lon2) / 2.)**2
    )
    c = 2. * np.arctan2(np.sqrt(a), np.sqrt(1 - a))    
    return R * c

def init_bearing(lat1, lon1, lat2, lon2):
    return np.arctan2(
        np.sin(np.deg2rad(lon2 - lon1)) * np.cos(np.deg2rad(lat2)),
        np.cos(np.deg2rad(lat1)) * np.sin(np.deg2rad(lat2)) -
        np.sin(np.deg2rad(lat1)) * np.cos(np.deg2rad(lat2)) *
        np.cos(np.deg2rad(lon2 - lon1))
    )

def make_train_step(model, optimizer, loss_func, stop_on_nan=True):
    def train_step(batch):
        model.train()
        
        ## old
        # pred = model(batch['inputs'], batch['aux_inputs'])
        # loss = loss_func(pred, batch)
        
        pred, logsig, p_pred, p_logsig = model(
            batch['inputs'],
            batch['aux_inputs'],
            precip_nbr_data=batch['precip_nbrs'],
            allvar_nbr_data=batch['allvar_nbrs']
        )
        loss = loss_func(pred, logsig, p_pred, p_logsig, batch)
        
        if stop_on_nan:
            if np.isnan(loss.item()):
                return 'STOP'
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return train_step

def make_val_step(model, loss_func, stop_on_nan=True):
    def val_step(batch):
        model.eval()
        
        ## old
        # pred = model(batch['inputs'], batch['aux_inputs'])
        # loss = loss_func(pred, batch)
        
        pred, logsig, p_pred, p_logsig = model(
            batch['inputs'],
            batch['aux_inputs'],
            precip_nbr_data=batch['precip_nbrs'],
            allvar_nbr_data=batch['allvar_nbrs']
        )
        loss = loss_func(pred, logsig, p_pred, p_logsig, batch)
        
        if stop_on_nan:
            if np.isnan(loss.item()):
                return 'STOP'
        
        return loss.item()
    return val_step
    
    

def make_loss_func(weight_gaps=1., use_non_gaps=False,
                   enforce_gap_endpoints=True, weight_endpoints=1.):
    mse = nn.MSELoss(reduction='mean')
    def gap_mse_loss(pred, batch):
        fake_gap_mask = (batch['our_gaps'])*(~batch['existing_gaps'])
        gap_loss = weight_gaps * mse(pred[fake_gap_mask], batch['targets'][fake_gap_mask])
        
        if enforce_gap_endpoints:
            endpoint_loss = gap_loss * 0
            for b in range(len(batch['metadata'])):
                for var in np.setdiff1d(batch['metadata'][b]['var_order'], ['PRECIP']):                    
                    ii = np.where(np.array(batch['metadata'][b]['var_order'])==var)[0][0]
                    if batch['metadata'][b]['tot_our_gaps'][var]>0:
                        gap_inds = torch.where(batch['our_gaps'][b,ii,:])[0]
                        if (gap_inds[0]-1)>=0 and not batch['existing_gaps'][b,ii,gap_inds[0]-1]:
                            endpoint_loss += mse(pred[b,ii,gap_inds[0]-1], batch['targets'][b,ii,gap_inds[0]-1])
                        if (gap_inds[0]+1)<len(batch['our_gaps'][b,ii,:]) and not batch['existing_gaps'][b,ii,gap_inds[0]+1]:
                            endpoint_loss += mse(pred[b,ii,gap_inds[0]+1], batch['targets'][b,ii,gap_inds[0]+1])        
            gap_loss += weight_endpoints * endpoint_loss
            
        if use_non_gaps:
            nongap_mask = (~batch['our_gaps'])*(~batch['existing_gaps'])
            nongap_loss = mse(pred[nongap_mask], batch['targets'][nongap_mask])
            gap_loss += nongap_loss
        return gap_loss
        
    return gap_mse_loss

def normal_loglikelihood(mu, x, sigma):
    return 0.5 * ((x - mu)/(sigma + EPS)) * ((x - mu)/(sigma + EPS)) + torch.log(sigma * SQRT2PI + EPS)

def make_nll_loss(weight_gaps=1., use_non_gaps=False,
                  enforce_gap_endpoints=True, weight_endpoints=1.):
    
    def gap_nll_loss(pred, logsig, p_pred, p_logsig, batch):
        p_ind = batch['metadata'][0]['var_order'].index('PRECIP')
        pred = torch.cat([pred[:,:p_ind,:], p_pred, pred[:,p_ind:,:]], dim=1)
        sigma = torch.cat([logsig[:,:p_ind,:], p_logsig, logsig[:,p_ind:,:]], dim=1)
        sigma = torch.exp(sigma)
        
        fake_gap_mask = (batch['our_gaps'])*(~batch['existing_gaps'])
        gap_loss = weight_gaps * normal_loglikelihood(
            batch['targets'][fake_gap_mask],
            pred[fake_gap_mask],
            sigma[fake_gap_mask]
        ).mean()
        
        if enforce_gap_endpoints:
            endpoint_loss = gap_loss * 0
            num = 0
            for b in range(len(batch['metadata'])):
                #for var in np.setdiff1d(batch['metadata'][b]['var_order'], ['PRECIP']):
                for var in batch['metadata'][b]['var_order']:
                    ii = np.where(np.array(batch['metadata'][b]['var_order'])==var)[0][0]
                    if batch['metadata'][b]['tot_our_gaps'][var]>0:
                        gap_inds = torch.where(batch['our_gaps'][b,ii,:])[0]
                        if (gap_inds[0]-1)>=0 and not batch['existing_gaps'][b,ii,gap_inds[0]-1]:
                            endpoint_loss += normal_loglikelihood(
                                batch['targets'][b,ii,gap_inds[0]-1],
                                pred[b,ii,gap_inds[0]-1],
                                sigma[b,ii,gap_inds[0]-1]
                            ).mean()
                            num += 1
                        if ((gap_inds[0]+1)<len(batch['our_gaps'][b,ii,:]) and 
                            not batch['existing_gaps'][b,ii,gap_inds[0]+1]):
                            endpoint_loss += normal_loglikelihood(
                                batch['targets'][b,ii,gap_inds[0]+1],
                                pred[b,ii,gap_inds[0]+1],
                                sigma[b,ii,gap_inds[0]+1]
                            ).mean()
                            num += 1
            if num>0:
                gap_loss += weight_endpoints * endpoint_loss / num
            
        if use_non_gaps:
            nongap_mask = (~batch['our_gaps'])*(~batch['existing_gaps'])
            nongap_loss = normal_loglikelihood(
                batch['targets'][nongap_mask],
                pred[nongap_mask],
                sigma[nongap_mask]
            ).mean()            
            gap_loss += nongap_loss
        
        return gap_loss
        
    return gap_nll_loss

def update_checkpoint(epoch, model, optimizer, best_loss, losses, val_losses):
    return {'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss,            
            'losses': losses,
            'val_losses': val_losses}

def prepare_run(checkpoint):
    if checkpoint is None:
        curr_epoch = 0
        best_loss = np.inf
        losses = []
        val_losses = []
    else:
        curr_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']        
        try:
            losses = checkpoint['losses']
            val_losses = checkpoint['val_losses']
        except:
            losses = []
            val_losses = []
    return losses, val_losses, curr_epoch, best_loss

def save_checkpoint(state, is_best, outdir):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    f_path = outdir + '/checkpoint.pth'
    torch.save(state, f_path)
    if is_best:
        print ("=> Saving a new best")
        best_fpath = outdir + '/best_model.pth'
        shutil.copyfile(f_path, best_fpath)
    else:
        print ("=> Validation loss did not improve")

def load_checkpoint(checkpoint_fpath, model, optimizer, device):
    if device.type=='cpu':
        checkpoint = torch.load(checkpoint_fpath, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint.pop('state_dict'))
    if not optimizer is None:
        optimizer.load_state_dict(checkpoint.pop('optimizer'))  
    return model, optimizer, checkpoint

def setup_checkpoint(model, optimizer, device, load_prev_chkpnt,
                     model_outdir, log_dir, specify_chkpnt=None,
                     reset_chkpnt=False):
    if load_prev_chkpnt:
        if specify_chkpnt is None:
            loadmodel = model_outdir + 'best_model.pth'
            print('Loading best checkpoint...')
        else:
            ## to load different weights to begin        
            # specify_chkpnt of form "modelname/checkpoint.pth" or 
            # "OTHERMODEL/best_model.pth" or "OTHERMODEL/checkpoint.pth"
            loadmodel = f'{log_dir}/{specify_chkpnt}'
            print(f'Loading {log_dir}/{specify_chkpnt}...')
        try:
            model, optimizer, checkpoint = load_checkpoint(loadmodel, model, optimizer, device)
            print('Loaded checkpoint successfully')
            print(f'Best loss: {checkpoint["best_loss"]}')
            print(f'current epoch: {checkpoint["epoch"]}')
        except:
            print('Failed loading checkpoint')
            checkpoint = None
    else: 
      checkpoint = None
      loadmodel = None

    if reset_chkpnt is True:        
        checkpoint = None # adding this to reset best loss and loss trajectory
    
    return model, optimizer, checkpoint

def send_batch_to_device(batch, device):
    # pop complicated data structures
    meta = batch.pop('metadata')
    p_nbrs = False
    av_nbrs = False
    if 'precip_nbrs' in batch.keys():
        precip_nbrs = batch.pop('precip_nbrs')
        p_nbrs = True
    if 'allvar_nbrs' in batch.keys():
        allvar_nbrs = batch.pop('allvar_nbrs')
        av_nbrs = True
    # send easy tensors to device
    batch = {k : batch[k].to(device) for k in batch.keys()}
    batch['metadata'] = meta
    # deal with nbr dicts
    nbr_arrs = ['masked_data', 'attention_mask', 'node_aux', 'edge_aux']
    if p_nbrs:
        for ii in range(len(precip_nbrs)):
            for n in range(len(precip_nbrs[ii])):
                for arr in nbr_arrs:
                    precip_nbrs[ii][n][arr] = precip_nbrs[ii][n][arr].to(device)
        batch['precip_nbrs'] = precip_nbrs
    if av_nbrs:
        for ii in range(len(allvar_nbrs)):
            for n in range(len(allvar_nbrs[ii])):
                for arr in nbr_arrs:
                    allvar_nbrs[ii][n][arr] = allvar_nbrs[ii][n][arr].to(device)    
        batch['allvar_nbrs'] = allvar_nbrs    
    return batch
