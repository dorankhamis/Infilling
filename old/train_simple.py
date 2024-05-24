import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import pkbar
import shutil

from torch.optim import Adam
from pathlib import Path

from params import model_cfg as cfg
from architectures.gap_fill_attn import NbrAttnGapFill
from data_generator import gap_generator
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__=="__main__":
    batch_size = 8
    max_epochs = 1000
    train_batches_per_epoch = 400
    val_batches_per_epoch = 150
    
    log_dir = '/home/users/doran/projects/infilling/logs/'
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    precip = False
    use_nbrs = False
    sep_precip_branch = False
    model_name = f'infill_simple_p{precip}_n{use_nbrs}'
    model_outdir = f'{log_dir}/{model_name}/'    
    Path(model_outdir).mkdir(parents=True, exist_ok=True)    
    specify_chkpnt = f'{model_name}/checkpoint.pth' # if None, load best, otherwise "modelname/checkpoint.pth"
    reset_chkpnt = False
    load_prev_chkpnt = True
        
    gap_length_seq = np.round(np.linspace(1,48,max_epochs)).astype(np.int32)
    gap_sd_seq = gap_length_seq/2.
    gap_sd_seq[500:] = np.linspace(gap_sd_seq[500], 10, len(gap_sd_seq[500:]))
    logprob_vargap_seq = np.linspace(-2,-0.5,max_epochs)
    
    # create data generator and model
    dg = gap_generator(precip=precip, use_nbrs=use_nbrs)
    cfg.features_d = 2 * len(dg.use_vars)
    
    model = NbrAttnGapFill(cfg, precip_branch=sep_precip_branch) # whether we have a splitting model with additional precip branch
    model.to(device)
    
    # create optimizer and load checkpoint
    optimizer = Adam(model.parameters(), lr=cfg.lr)
    
    model, optimizer, checkpoint = setup_checkpoint(
        model, optimizer, device, load_prev_chkpnt,
        model_outdir, log_dir,
        specify_chkpnt=specify_chkpnt,
        reset_chkpnt=reset_chkpnt
    )
    
    # define loss function and create train steps
    loss_func = make_nll_loss(
        weight_gaps=5,
        use_non_gaps=True,
        enforce_gap_endpoints=True,
        weight_endpoints=1
    )
    train_step = make_train_step(model, optimizer, loss_func,
                                 sep_precip_branch=sep_precip_branch)
    val_step = make_val_step(model, loss_func,
                             sep_precip_branch=sep_precip_branch)    
    losses, val_losses, curr_epoch, best_loss = prepare_run(checkpoint)
        
    print("Trainable parameters: %d" % count_parameters(model))
    print("Non-trainable parameters: %d" % count_parameters(model, trainable=False))
    
    if False:
        epoch = curr_epoch
        
        batch = dg.get_batch(
            batch_size,
            const_l=True,
            batch_type='train',
            min_gap_logprob=logprob_vargap_seq[epoch],
            mean_gap_length=gap_length_seq[epoch],
            gap_length_sd=gap_sd_seq[epoch],
            shortrange=cfg.shortrange,
            longrange=cfg.longrange
        )
        batch = send_batch_to_device(batch, device)
        
        if sep_precip_branch:
            pred, logsig, p_pred, p_logsig = model(
                batch['inputs'],
                batch['aux_inputs'],
                precip_nbr_data=batch['precip_nbrs'],
                allvar_nbr_data=batch['allvar_nbrs']
            )
            loss = loss_func(pred, logsig, batch, p_pred, p_logsig)
            
            # join precip onto other vars
            p_ind = batch['metadata'][0]['var_order'].index('PRECIP')
            pred = torch.cat([pred[:,:p_ind,:], p_pred, pred[:,p_ind:,:]], dim=1)
            logsig = torch.cat([logsig[:,:p_ind,:], p_logsig, logsig[:,p_ind:,:]], dim=1)            
        else:
            pred, logsig = model(
                batch['inputs'],
                batch['aux_inputs'],
                allvar_nbr_data=batch['allvar_nbrs']
            )
            loss = loss_func(pred, logsig, batch)            
        print(loss)
        
        sigma = torch.exp(logsig)
               
        plt.plot(checkpoint['losses'])
        plt.plot(checkpoint['val_losses'])
        plt.show()
                
        for b in range(2):
            nplots = len(batch['metadata'][0]['var_order'])
            ncols = int(np.ceil(np.sqrt(nplots)))
            nrows = int(np.ceil(nplots / ncols))        
            fig, axs = plt.subplots(nrows, ncols)
            for i in range(nplots):
                axs[i//ncols, i%ncols].plot(batch['inputs'][b,2*i,:].numpy())
                axs[i//ncols, i%ncols].plot(batch['targets'][b,i,:].numpy())
                axs[i//ncols, i%ncols].plot(pred[b,i,:].detach().numpy())
                axs[i//ncols, i%ncols].fill_between(list(range(pred[b,i,:].shape[0])),
                                                    pred[b,i,:].detach().numpy()-sigma[b,i,:].detach().numpy(),
                                                    pred[b,i,:].detach().numpy()+sigma[b,i,:].detach().numpy(),
                                                    alpha=0.2)
                axs[i//ncols, i%ncols].title.set_text(batch['metadata'][0]['var_order'][i])        
            plt.show()
        
        nplots = len(batch['metadata'][0]['aux_var_order'])
        ncols = int(np.ceil(np.sqrt(nplots)))
        nrows = int(np.ceil(nplots / ncols))        
        fig2, axs2 = plt.subplots(nrows, ncols)
        for i in range(nplots):
            axs2[i//ncols, i%ncols].plot(batch['aux_inputs'][b,i,:].numpy())            
            axs2[i//ncols, i%ncols].title.set_text(batch['metadata'][0]['aux_var_order'][i])
        plt.show()

    
    ## fit the model
    model.train()
    for epoch in range(curr_epoch, max_epochs):
        kbar = pkbar.Kbar(target=train_batches_per_epoch,
                          epoch=epoch,
                          num_epochs=max_epochs,
                          width=15,
                          always_stateful=False)
        running_loss = []
        for bidx in range(1, train_batches_per_epoch+1):
            batch = dg.get_batch(batch_size,
                                 const_l=True,
                                 batch_type='train',
                                 min_gap_logprob=logprob_vargap_seq[epoch],
                                 mean_gap_length=gap_length_seq[epoch],
                                 gap_length_sd=gap_sd_seq[epoch],
                                 shortrange=cfg.shortrange,
                                 longrange=cfg.longrange)
            batch = send_batch_to_device(batch, device)
            loss = train_step(batch)
            
            if loss=='STOP': break
            
            print_values = [('loss', loss)]
            kbar.update(bidx, values=print_values)
            running_loss.append(loss)
        losses.append(np.mean(running_loss)) # append epoch average loss
        
        if loss=='STOP': 
            print('Stopping due to nan loss')
            break
        
        with torch.no_grad():
            kbarv = pkbar.Kbar(target=val_batches_per_epoch,
                               epoch=epoch,
                               num_epochs=max_epochs,
                               width=15,
                               always_stateful=False)
            running_loss = []
            for bidx in range(1, val_batches_per_epoch+1):
                batch = dg.get_batch(batch_size,
                                     const_l=True,
                                     batch_type='val',
                                     min_gap_logprob=logprob_vargap_seq[epoch],
                                     mean_gap_length=gap_length_seq[epoch],
                                     gap_length_sd=gap_sd_seq[epoch],
                                     shortrange=cfg.shortrange,
                                     longrange=cfg.longrange)
                batch = send_batch_to_device(batch, device)
                loss = val_step(batch)
                
                if loss=='STOP': break
                
                print_values = [('val_loss', loss)]
                kbarv.update(bidx, values=print_values)
                running_loss.append(loss)
            
            if loss=='STOP':
                print('Stopping due to nan loss')
                break
            
            val_losses.append(np.mean(running_loss))
            kbar.add(1, values=[('val_loss', val_losses[-1])])        
            is_best = bool(val_losses[-1] < best_loss)
            best_loss = min(val_losses[-1], best_loss)
            checkpoint = update_checkpoint(epoch, model, optimizer,
                                           best_loss, losses, val_losses)
            save_checkpoint(checkpoint, is_best, model_outdir)
        print("Done epoch %d" % (epoch+1))
