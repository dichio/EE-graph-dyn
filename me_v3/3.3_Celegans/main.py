#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 18:16:53 2022

@author: vito.dichio
@version: 3 - UD / Master-Equation

Last update on 29 Mag 2023
"""

#%% Import what is needed

import os, time
os.chdir(os.path.dirname(os.path.realpath(__file__)))
start_time = time.time()

import numpy as np 

import src.building_blocks as sim

myseed = 160318
#%% init

nameset = 'test'

params = np.array([180,             # N    
                   72,              # T
                   1.4263e-3,       # mu*
                   ##############
                   2**10,           # M
                   23,              # nu
                   1.,              # rm
                   901.708872647173,# rf
                   758052841])      # seed

np.random.seed(int(params[-1]))

params = np.concatenate((params,sim.shout_prms(which='combo',random=False))) # add fitness parameters

T, N, M, nu, mu, phi, th, ID = sim.init(params,verbose=True)

outpath = sim.shout_paths(ID,simset=nameset)[0]

cl, ncl, fits = sim.init_pop(M,phi,th)

stats = np.zeros((T+1,sim.statrowlen)); fsts = np.zeros((T+1,2))

#%% evolution

mu0 = mu 

for i in range(0,T+1):
    
    t = i/nu
    
    ## step 0 - savedata
    fsts[i] = sim.save_fitstats(fits,ncl)
    stats[i] = sim.save_stats(t, cl, ncl, th, instasavepath=outpath, save_cl = True)

    # step 1 - expand
    pop = np.repeat(cl, ncl, axis=0) 
    
    # step 2 - mutate  
    #mu = sim.mut_rate_adjust(t,nu,mu0,pop)
    pop = sim.mutations(pop, mu, style = "uphard_mp")  
    
    # step 3 - merge clones
    cl, ncl = np.unique(pop, axis=0, return_counts=True)
    
    # step 4 - selection 
    fits, ncl = sim.selection(t, cl, ncl, M, phi, th)
    #fits = np.zeros(cl.shape[0]) # no selection

    ## step 6 - accessories
    sim.tellme(t,1,ncl,fsts[i],alot=True)
    sim.estimate_simtime(i,T,5)


#%% 
sim.finalise(outpath, params, stats, fsts)

print("--- %f minutes ---" % ((time.time() - start_time)/60.))