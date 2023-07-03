#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 15:10:58 2022

@author: vito.dichio
@version: 3 - UD / Data Analysis (DA) for one simulation 

Last update on 20 Gen 2023
"""

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import src.data_miner as miner
import src.pltools as myplt

simset = 'one-sim'
ID = 'M1024-rf901.71-seed758052841'

#%% import

setpath, datapath, plotpath = miner.shout_paths(ID,simset)

N, nu, ts, phi, th = miner.get_params(datapath,verbose=True)
thl = th[0:5:2]

gexp, sexp, fexp, texp = miner.import_experimental(N,phi,th,scaledist=True)
fs, s = miner.import_alltime(datapath,readfit=True)

nstats = miner.sim.nstats

#%%

sexp_er = miner.cleanexp(sexp,True,thl) 
s_er, s_at = miner.cleansim(s,texp,nu,thl)

#%%
delta_mah = miner.error(sexp_er,s_er,which="mahalanobis",t_range = [-1])
print( 'err = %.2f' %delta_mah)
    
#%%% viz

myplt.fit_vs_t( plotpath,
                nu,
                texp,
                fexp,
                ts,
                fs[:,0],
                fs[:,1]**(1/2),
                save=ID
                )

#%%

i = 2
myplt.stat_vs_t(plotpath,
                texp,
                sexp[:,i],
                ts,
                s_at[:,i],      
                s_at[:,i+nstats]**(1/2),
                title=myplt.statname[i],
                save=ID
                )

#%%

i = 1 ; j = 2
myplt.stats_1vs1(plotpath,
                   texp,
                   sexp[:,i], 
                   sexp[:,j],
                   ts,
                   s_at[:,i], 
                   s_at[:,j],
                   nu,
                   title = myplt.statname[i] + ' vs ' + myplt.statname[j],
                   save=ID
                   )
#%%

# myplt.edgesvsall(plotpath,
#                    ts,
#                    s[:,0:nstats],
#                    texp,
#                    sexp[:,0:nstats],
#                    nu,
#                    title = myplt.statname[0] + " vs all",
#                    save=ID
#                    )

# #%%

# myplt.statspace3d(plotpath,
#                     ts,
#                     s[:,1:nstats],
#                     texp,
#                     sexp[:,1:nstats],
#                     nu,
#                     title = "3D statspace",
#                     save=ID
#                     )

#%% insta

acode = '45h-1' # 0h, 5h, 8h, 16h, 23h, 27h, 45h-1, 45h-2
age, ageidx = miner.ageindex(acode)


ns_t, s_t = miner.import_insta(datapath,acode,clones=False)

for statidx in range(nstats):
    myplt.snaphist(plotpath,
                   ns_t,
                   s_t[:,statidx],
                   sexp[ageidx,statidx],
                   title = '%ih-' %age + myplt.statname[statidx],
                   save = ID)

#%%

ID1 = ID
ID2 = 'M1024-rf0.00-seed758052841' #null model


acode = '27h' # 0h, 5h, 8h, 16h, 23h, 27h, 45h-1, 45h-2
age, ageidx = miner.ageindex(acode)

#statidx = 3 # 0:edges, 1:gwdegree 2:gwesp 3:dist
for statidx in range(3):
    myplt.snaphist_cf2(simset,
                    simset,
                    ID1,
                    ID2,
                    acode,
                    statidx,
                    sexp[ageidx,statidx],
                    title = '%ih-%s' %(age,myplt.statname[statidx]),
                    save = True)



#%% goodness of fit 

# data
compute_gof_stats = True

gof_exp, gof_sim = miner.compute_gof(datapath,gexp,compute_gof_stats = compute_gof_stats)
gof_sim_null = miner.compute_gof('null',gexp,compute_gof_stats = False)[1]


#%%
# Violins - relative errors

re_sim, re_null = miner.compute_rel_errs(gof_exp[:,:7],gof_sim[:,:,:7],gof_sim_null[:,:,:7])

for gof in miner.gofstats:
    myplt.gof_violins(gof, re_sim, re_null, save=plotpath)


#%%
# Distrib

stat = "av_short_path_cc" # ["twopaths","triangles","av_clustering","transitivity","loc_efficiency","glob_efficiency","av_short_path_cc"]
time = '5h' # 5h, 8h, 16h, 23h, 27h, 45h-1, 45h-2

myplt.gof_distrib(stat,time,gof_exp,gof_sim,gof_sim_null,save=plotpath)


#%%

dd_sim = gof_sim[:,:,9:]
dd_null = gof_sim_null[:,:,9:]
dd_exp = gof_exp[:,7:]


#%%
import src.pltools as myplt
time = '5h' # 5h, 8h, 16h, 23h, 27h, 45h-1, 45h-2

maxdeg=30

for time in miner.agecode[1:-1]:
    
    myplt.gof_degrees(time,maxdeg+1,dd_exp,dd_sim,dd_null,save=plotpath)













