#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 17:17:38 2023

@author: vito.dichio
"""

import numpy as np
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import src.data_miner as miner
import src.pltools as myplt

myseed = 160318; np.random.seed(myseed)
#%% copy-paste from phsp

nameset = 'ce_topo_rf'

M = [2**10]
rf =  [0 + 20*i for i in range(0,71)]
nseeds = 100


# Copy the above from phsp
    
seed = miner.read_seeds(nseeds,nameset)

#%
grid = [(p1, p2, seed[idx]) for p1 in M for p2 in rf for idx in range(nseeds)]


#err_mah, grid_None = miner.build_err_array(nameset,grid,which='mahalanobis',t_range = [0,1,2,3,4,5],grid_none=True)


#%% Compute err - matrices
compute_errs = False

if compute_errs:
    
    err_mah = miner.build_err_array(nameset,grid,which='mahalanobis',t_range = [0,1,2,3,4,5])
    err_mah_fin = miner.build_err_array(nameset,grid,which='mahalanobis',t_range=[-1])
    # err_fit = miner.build_err_array(nameset,grid,which='fitness',t_range = [0,1,2,3,4,5])
    # err_fit_fin = miner.build_err_array(nameset,grid,which='fitness',t_range=[-1])

else:
    setpath = './output/' + nameset + '/'
    err_mah = np.load(setpath + 'err_mah_allt.npy',allow_pickle=True)
    err_mah_fin = np.load(setpath + 'err_mah_fin.npy',allow_pickle=True)
    # err_fit = np.load(setpath + 'err_fit_allt.npy',allow_pickle=True)
    # err_fit_fin = np.load(setpath + 'err_fit_fin.npy',allow_pickle=True)

#%%

# average over seeds
err_mah_av = np.nanmean(err_mah,-1)
err_mah_fin_av = np.nanmean(err_mah_fin,-1)

err_mah_sd = np.nanvar(err_mah,-1)**0.5
err_mah_fin_sd = np.nanvar(err_mah_fin,-1)**0.5


#err_fit_av = np.nanmean(err_fit,-1)
#err_fit_fin_av = np.nanmean(err_fit_fin,-1)


#%%
errs = err_mah_fin_av[0,:]

xticks = rf

myplt.axis_vs_err(nameset,
                  errs,
                  labels = [""],
                  xticks = xticks,
                  which = "rf",
                  save=True
                  )#%%
#%%

#errs = np.vstack((err_mah_av[0,:],err_fit_av[0,:],err_mah_fin_av[0,:],err_fit_fin_av[0,:]))
errs = np.vstack((err_mah_av[0,:],err_mah_fin_av[0,:]))

xticks = rf

myplt.axis_vs_err_multi(nameset,
                        errs,
                        labels = ['mah','fit','mah45','fit45'],
#                        labels = ['mah','mah45'],
                        xticks = xticks,
                        which = "rf",
                        norm=False,
                        save=True
                        )#%%
#%%
errs2d = err_mah_av[:,:]
labels2d = [M,rf]

myplt.rf_rm_err(nameset,
                   errs2d,
                   labels2d,
                   save=True
                   )



#%%%%% Plots for presentation MM2023

import src.pltools as myplt

start = 10

myplt.mah_fin(nameset,
              np.array(rf[start:]),
              err_mah_fin_av[0,start:],
              err_mah_fin_sd[0,start:],
              err_mah_av[0,start:],
              err_mah_sd[0,start:],
              True)

# myplt.mah_allt(nameset,
#               np.array(rf[start:]),
#               err_mah_fin_av[0,start:],
#               err_mah_fin_sd[0,start:],
#               err_mah_av[0,start:],
#               err_mah_sd[0,start:],
#               True)



















#%% copy-paste from phsp

nameset = 'ce_rf_fin'

M = [2**10]
rf =  [901.708872647173]
nseeds = 500


# Copy the above from phsp
    
seed = miner.read_seeds(nseeds,nameset)

#%
grid = [(p1, p2, seed[idx]) for p1 in M for p2 in rf for idx in range(nseeds)]


#err_mah, grid_None = miner.build_err_array(nameset,grid,which='mahalanobis',t_range = [0,1,2,3,4,5],grid_none=True)


#%% Compute err - matrices
compute_errs = False

if compute_errs:
    
    err_mah = miner.build_err_array(nameset,grid,which='mahalanobis',t_range = [0,1,2,3,4,5])
    err_mah_fin = miner.build_err_array(nameset,grid,which='mahalanobis',t_range=[-1])
    # err_fit = miner.build_err_array(nameset,grid,which='fitness',t_range = [0,1,2,3,4,5])
    # err_fit_fin = miner.build_err_array(nameset,grid,which='fitness',t_range=[-1])

else:
    setpath = './output/' + nameset + '/'
    err_mah = np.load(setpath + 'err_mah_allt.npy',allow_pickle=True)
    err_mah_fin = np.load(setpath + 'err_mah_fin.npy',allow_pickle=True)
    # err_fit = np.load(setpath + 'err_fit_allt.npy',allow_pickle=True)
    # err_fit_fin = np.load(setpath + 'err_fit_fin.npy',allow_pickle=True)

#%%

# average over seeds
err_mah_av = np.nanmean(err_mah,-1)
err_mah_fin_av = np.nanmean(err_mah_fin,-1)

err_mah_sd = np.nanvar(err_mah,-1)**0.5
err_mah_fin_sd = np.nanvar(err_mah_fin,-1)**0.5


#err_fit_av = np.nanmean(err_fit,-1)
#err_fit_fin_av = np.nanmean(err_fit_fin,-1)


#%%
errs = err_mah_fin_av[0,:]

xticks = rf

myplt.axis_vs_err(nameset,
                  errs,
                  labels = [""],
                  xticks = xticks,
                  which = "rf",
                  save=True
                  )#%%
#%%

#errs = np.vstack((err_mah_av[0,:],err_fit_av[0,:],err_mah_fin_av[0,:],err_fit_fin_av[0,:]))
errs = np.vstack((err_mah_av[0,:],err_mah_fin_av[0,:]))

xticks = rf

myplt.axis_vs_err_multi(nameset,
                        errs,
                        labels = ['mah','fit','mah45','fit45'],
#                        labels = ['mah','mah45'],
                        xticks = xticks,
                        which = "rf",
                        norm=False,
                        save=True
                        )#%%
#%%
errs2d = err_mah_av[:,:]
labels2d = [M,rf]

myplt.rf_rm_err(nameset,
                   errs2d,
                   labels2d,
                   save=True
                   )



#%%%%% Plots for presentation MM2023

import src.pltools as myplt

start = 10

myplt.mah_fin(nameset,
              np.array(rf[start:]),
              err_mah_fin_av[0,start:],
              err_mah_fin_sd[0,start:],
              err_mah_av[0,start:],
              err_mah_sd[0,start:],
              True)

myplt.mah_allt(nameset,
              np.array(rf[start:]),
              err_mah_fin_av[0,start:],
              err_mah_fin_sd[0,start:],
              err_mah_av[0,start:],
              err_mah_sd[0,start:],
              True)


