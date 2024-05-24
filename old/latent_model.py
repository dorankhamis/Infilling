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
from architectures.rnn_vae import RNNVAE
#from architectures.context_embed_attn import TemporalEmbedder
from architectures.aggregate_embedder import AggregateEmbedder
from data_generator import gap_generator
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
    - Remove (some of) the attention modules so we don't have
        positional encoding wiggles? Maybe look into a learnt
        positional encoding rather than sine waves
        Try Alibi linear biases attention, skeleton code already added
    - Experiment with aggregate/trajectory embedding size
    - Why does air pressure never seem to work... fully check that this
        is working properly
    - Is there a way to constrain the r_trajectory to be smooth as
        part of the loss function, potentially?
    - We expect larger uncertainty where there are data gaps, but 
        currently we are only drawing a "single" random variable
        so there is not going to be a temporal-dependence in the 
        uncertainty? Some way to draw from the same latent normal
        for the data gaps and the data-present parts, but with some
        relationship for scaling the drawing when doing gaps?
'''


if __name__=="__main__":
    ## initialiase data generator
    batch_size = 8
    max_epochs = 1000
    train_batches_per_epoch = 400
    val_batches_per_epoch = 150
    ensemble_size = 5
    
    log_dir = '/home/users/doran/projects/infilling/logs/'
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    precip = False
    use_nbrs = False
    feed_decoder = False
    #model_name = f'rnnvae_model2_p{precip}_feeddec{feed_decoder}'
    #model_name = f'temporal_embedder_p{precip}'
    model_name = f'aggregate_embedder'
    model_outdir = f'{log_dir}/{model_name}/'    
    Path(model_outdir).mkdir(parents=True, exist_ok=True)    
    specify_chkpnt = f'{model_name}/checkpoint.pth' # if None, load best, otherwise "modelname/checkpoint.pth"
    reset_chkpnt = False
    load_prev_chkpnt = True
        
    gap_length_seq = np.round(np.linspace(5,48,max_epochs)).astype(np.int32)
    gap_sd_seq = gap_length_seq/2.
    gap_sd_seq[500:] = np.linspace(gap_sd_seq[500], 10, len(gap_sd_seq[500:]))
    logprob_vargap_seq = np.linspace(-2,-0.5,max_epochs)
    
    # create data generator and model
    dg = gap_generator(precip=precip, use_nbrs=use_nbrs)
    
    # for RNNVAE model
    # cfg.features_d = 2 * len(dg.use_vars)
    # if feed_decoder:
        # cfg.features_dec = cfg.features_aux + cfg.features_d
        # cfg.embed_dec = 2 * cfg.embed_aux
    # else:
        # cfg.features_dec = cfg.features_aux
        # cfg.embed_dec = cfg.embed_aux

    # for TemporalEmbedder model
    cfg.features_d = 6 # vars in
    cfg.embed_v = 16 # embed dim, or model size
    cfg.features_traj = 3
    cfg.features_dec = 96

    #model = RNNVAE(cfg)
    #model = TemporalEmbedder(cfg)
    model = AggregateEmbedder(cfg)
    model.to(device)
    
    # create optimizer and load checkpoint
    optimizer = Adam(model.parameters(), lr=cfg.lr)
    
    model, optimizer, checkpoint = setup_checkpoint(
        model, optimizer, device, load_prev_chkpnt,
        model_outdir, log_dir,
        specify_chkpnt=specify_chkpnt,
        reset_chkpnt=reset_chkpnt
    )
    
    # create train/val steps
    train_step = make_rnnvae_train_step(
        model,
        optimizer,
        stop_on_nan=True,
        use_latent_mean=False,
        ensemble_size=ensemble_size
    )
    val_step = make_rnnvae_val_step(
        model,
        stop_on_nan=True,
        use_latent_mean=False,
        ensemble_size=ensemble_size
    )
    losses, val_losses, curr_epoch, best_loss = prepare_run(checkpoint)
        
    print("Trainable parameters: %d" % count_parameters(model))
    print("Non-trainable parameters: %d" % count_parameters(model, trainable=False))
    
    if False:
        plt.plot(checkpoint['losses'])
        plt.plot(checkpoint['val_losses'])
        plt.show()
        
        ## testing        
        epoch = curr_epoch
        
        batch = dg.get_batch(
            batch_size,
            const_l=True,
            batch_type='train',
            min_gap_logprob= -0.05,#logprob_vargap_seq[epoch],
            mean_gap_length= 50,#gap_length_seq[epoch],
            gap_length_sd=gap_sd_seq[epoch],
            shortrange=cfg.shortrange,
            longrange=cfg.longrange
        )
        batch = send_batch_to_device(batch, device)
        batch['observed_mask'] = (~batch['our_gaps']) * (~batch['existing_gaps'])
        batch['inputs'] = batch['inputs'][:,::2,:] # no longer want binary presence data
        
        # if feed_decoder:
            # batch['dec_inputs'] = torch.cat([batch['aux_inputs'], batch['inputs']], dim=-2)
        # else:
            # batch['dec_inputs'] = batch['aux_inputs']
    
        model.eval()
        ensemble_size = 10
        use_latent_mean = False
        out = model(
            batch,
            use_latent_mean=use_latent_mean,
            calc_elbo=True,
            ensemble_size=ensemble_size
        )
        
        r_traj = out['r_trajectory'].detach().numpy()
        b = 2
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(r_traj[b,:,0], r_traj[b,:,1], r_traj[b,:,2])
        plt.show()
        plt.plot(r_traj[b,:,0])
        plt.plot(r_traj[b,:,1])
        plt.plot(r_traj[b,:,2])
        plt.show()
                
        for b in range(batch['inputs'].shape[0]):
            nplots = len(batch['metadata'][0]['var_order'])
            ncols = int(np.ceil(np.sqrt(nplots)))
            nrows = int(np.ceil(nplots / ncols)) 
            
            # plot ensemble       
            fig, axs = plt.subplots(nrows, ncols)
            for i in range(nplots):
                axs[i//ncols, i%ncols].plot(batch['inputs'][b,i,:].numpy())                
                axs[i//ncols, i%ncols].plot(batch['targets'][b,i,:].numpy())
                for j in range(ensemble_size):
                    axs[i//ncols, i%ncols].plot(out['pred_ensemble'][b,j,i,:].detach().numpy(), c='k', alpha=0.5)                                    
                axs[i//ncols, i%ncols].title.set_text(batch['metadata'][0]['var_order'][i])        
            plt.show()
            
            # plot mean and std
            fig, axs = plt.subplots(nrows, ncols)
            for i in range(nplots):
                axs[i//ncols, i%ncols].plot(batch['inputs'][b,i,:].numpy())
                axs[i//ncols, i%ncols].plot(batch['targets'][b,i,:].numpy())
                axs[i//ncols, i%ncols].plot(out['pred_mu'][b,i,:].detach().numpy())
                axs[i//ncols, i%ncols].fill_between(list(range(out['pred_mu'][b,i,:].shape[0])),
                                                    out['pred_mu'][b,i,:].detach().numpy() - out['pred_sigma'][b,i,:].detach().numpy(),
                                                    out['pred_mu'][b,i,:].detach().numpy() + out['pred_sigma'][b,i,:].detach().numpy(),
                                                    alpha=0.2)
                axs[i//ncols, i%ncols].title.set_text(batch['metadata'][0]['var_order'][i])
            plt.show()
            
            # plot latent mean
            fig, axs = plt.subplots(nrows, ncols)
            for i in range(nplots):
                axs[i//ncols, i%ncols].plot(batch['inputs'][b,i,:].numpy())
                axs[i//ncols, i%ncols].plot(batch['targets'][b,i,:].numpy())
                axs[i//ncols, i%ncols].plot(out['pred_mu'][b,i,:].detach().numpy())
                axs[i//ncols, i%ncols].title.set_text(batch['metadata'][0]['var_order'][i])
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
            batch['observed_mask'] = (~batch['our_gaps']) * (~batch['existing_gaps']) # where we have observations
            batch['inputs'] = batch['inputs'][:,::2,:] # no longer want binary presence data
            
            # if feed_decoder:
                # batch['dec_inputs'] = torch.cat([batch['aux_inputs'], batch['inputs']], dim=-2)
            # else:
                # batch['dec_inputs'] = batch['aux_inputs']
            
            loss_dict = train_step(batch)
            if loss_dict=='STOP': break
            
            print_values = [('loss', loss_dict['neg_elbo_mean'].item())]
            kbar.update(bidx, values=print_values)
            running_loss.append(loss_dict['neg_elbo_mean'].item())
        losses.append(np.mean(running_loss)) # append epoch average loss
        
        if loss_dict=='STOP': 
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
                batch['observed_mask'] = (~batch['our_gaps']) * (~batch['existing_gaps'])
                batch['inputs'] = batch['inputs'][:,::2,:] # no longer want binary presence data
                
                # if feed_decoder:
                    # batch['dec_inputs'] = torch.cat([batch['aux_inputs'], batch['inputs']], dim=-2)
                # else:
                    # batch['dec_inputs'] = batch['aux_inputs']
                
                loss_dict = val_step(batch)
                if loss_dict=='STOP': break
                
                print_values = [('val_loss', loss_dict['neg_elbo_mean'].item())]
                kbarv.update(bidx, values=print_values)
                running_loss.append(loss_dict['neg_elbo_mean'].item())
            
            if loss_dict=='STOP':
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

