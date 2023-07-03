#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 15:11:57 2022

@author: vito.dichio
@version: 3 - UD / Tools for plots for Master-Equation

Last update on 12 Gen 2023
"""

import os

import src.building_blocks as sim

import numpy as np

myseed = 160318; np.random.seed(myseed)
#%% DA_one

def shout_paths(ID,simset=None):

    if simset != None:
        setpath = './output/' + simset + '/'
    else:
        setpath = './output/'
    
    datapath = setpath + ID + '/'

    plotpath = datapath + 'figures'
    if not(os.path.exists(plotpath)):
        os.mkdir(plotpath)
    
    return setpath, datapath, plotpath


def get_params(datapath, verbose=True):
    
    params = np.load(datapath + 'params.npy',allow_pickle=True)
    
    T, N, M, nu, mu, phi, th = sim.init(params,verbose=verbose)[0:-1]
    
    ts = np.arange(0,T+1)/nu

    return N, nu, ts, phi, th
    

def import_experimental(N,phi,th,scaledist=False):
    
    L = int(N*(N-1)/2)
    thl = th[0:5:2]
    
    texp = np.array([0,5,8,16,23,27,45,45])
    
    gexp = np.zeros((8,L))
    sexp = np.zeros((8,sim.nstats))
    fexp = np.zeros((8))
    
    for i in range(8):
        graph = sim.to_vector(np.loadtxt('src/data/'+str(i+1)+'_adj.txt'))
        
        gexp[i] = graph
        
        if scaledist:
            sexp[i] = sim.compute_stats(graph,th,texp[i])
        else:
            sexp[i] = sim.compute_stats(graph,th)
            
        fexp[i] = phi * np.sum( sexp[i,1:sim.nstats] * thl )
    
    return gexp, sexp, fexp, texp

def import_gtexp(N=180):
    
    L = int(N*(N-1)/2)
    texp = np.array([0,5,8,16,23,27,45,45])
    gexp = np.zeros((8,L))
    for i in range(8):
        gexp[i] = sim.to_vector(np.loadtxt('src/data/'+str(i+1)+'_adj.txt'))
        
    return texp, gexp

def compute_sexp(texp,gexp,th,scaledist=False):
    
    sexp = np.zeros((8,sim.nstats))
    for i in range(8):
        if scaledist:
            sexp[i] = sim.compute_stats(gexp[i],th,texp[i])
        else:
            sexp[i] = sim.compute_stats(gexp[i],th)
    
    return sexp    
    
def import_ecov(which="distance"):
    if which == "distance":
        ecovdist = np.loadtxt('src/data/8_ec_dist.txt')
        return ecovdist
    

def import_alltime(datapath,readfit=False):
    gstats = np.load(datapath + '/sts.npy', allow_pickle=True)
    if readfit == False:
        return gstats
    
    fsts = np.load(datapath + '/fsts.npy', allow_pickle=True)
    return fsts, gstats


agecode = ['0h','5h','8h','16h','23h','27h','45h-1','45h-2']
ages = dict(zip(agecode,[0,5,8,16,23,27,45,45]))
ageidx = dict(zip(agecode,[0,1,2,3,4,5,6,7]))


def ageindex(acode):
    
    if not(acode in agecode):
        print("This snapshot does not exist!")
        raise SystemExit
        
    return ages[acode], ageidx[acode]

def import_insta(datapath,agecode,clones=False):
    
    if not(agecode in ages.keys()):
        print("This snapshot does not exist!")
        raise SystemExit
        
    tns = np.load(datapath + 'T'+str(ages[agecode])+'_nc.npy',allow_pickle=True)
    
    tsts = np.load(datapath + 'T'+str(ages[agecode])+'_sts.npy',allow_pickle=True) 
     
    if clones:
        tcl = np.load(datapath + 'T'+str(ages[agecode])+'_cl.npy',allow_pickle=True)
        return tcl, tns, tsts
    else:
        return tns, tsts

#%%
def cleanexp(sexp,scale=False,thl=None):
    # replace last two with their average, discard t=0
    newsexp = np.vstack((sexp[1:-2,1:],np.mean(sexp[-2:,1:],axis=0)))
    if scale:
        newsexp = newsexp*thl # rescale by parameters
    return newsexp

def matrow_idx(which="diag",n=sim.nstats,k=0,bias=0):
    idxmat = np.full((n,n), 0)
    
    if which == "diag":
        np.fill_diagonal(idxmat,1)
        idx = sim.to_vector(idxmat,k)
        return (np.where(idx==1)[0]+bias).tolist()
    
    if which == "noedge":
        idxmat[1:,1:] = 1
        idx = sim.to_vector(idxmat,k)
        return (np.where(idx==1)[0]+bias).tolist()
   
def cleansim(s,texp,nu,thl,at=True):
    # for computation of error
    
    s_er = s[texp[1:-1]*nu] #select times
    keep = [i for i in range(1,sim.nstats)] + matrow_idx('noedge',n=sim.nstats,k=0,bias=sim.nstats)

    s_er = np.take(s_er,keep,axis=1) # remove "edges"
    
    s_er = s_er * np.concatenate([thl,sim.to_vector(np.outer(thl,thl),0)]) # rescale by parameters
    
    if at:
        # for plots
        keep = [i for i in range(0,sim.nstats)] + matrow_idx('diag',n=sim.nstats,k=0,bias=sim.nstats)
        s_at = np.take(s,keep,axis=1)
        return s_er, s_at
    else:
        return s_er


#%% compute error

def mahalanobis(x, av, cov):
    cov = sim.to_adj_mat(cov,sim.nstats-1,0)

    y = x - av
    inv_cov = np.linalg.inv(cov)

    return (np.dot(np.dot(y.T, inv_cov), y))**(1/2)
   
def euclidean(x,av):
    
    y = (x - av)
    
    return (np.sum(y**2))**(1/2)
    

def error(sexp,s,which,t_range=[0,1,2,3,4,5]):
    err_t = np.zeros(6)
    sep = sim.nstats-1
    for i in t_range:
        if which == "mahalanobis":
            err_t[i] = mahalanobis(sexp[i],s[i,:sep],s[i,sep:])
        elif which == "euclidean":
            err_t[i] = euclidean(sexp[i],s[i,:sep]) 
        elif which == "fitness":
            err_t[i] = np.abs(np.sum(sexp[i]) - np.sum(s[i,:sep]))
    return np.sum(err_t)

#%% plotter_phsp

def read_seeds(nseeds,nameset):
    if nseeds == 1:
        return np.array([160318])
    else:
        return np.load('./output/' + nameset + '/seeds.npy')
    
def gen_seeds(seedseed,n):
    np.random.seed(seedseed)
    return sim.seeds(nseeds=n)
    
def err_one(simset,gridv,texp,gexp,which,t_range):
    
    try: 
        
        ID = 'M%i-rf%.2f-seed%i' %gridv
        ID = 'M%i-rf%.2f-seed%i' %gridv
        
        datapath = './output/' + simset + '/' + ID + '/'
        
        # if not(os.path.exists(datapath+'sts.npy')) or os.stat(datapath+'sts.npy').st_size==0:
        #     print(ID + ' -- sts does not exist / empty!')
        #     return None
        
        # if not(os.path.exists(datapath)):
        #     print(ID + ' -- does not exist')
        #     return None
        
        # if not(os.path.exists(datapath+'params.npy')) or os.stat(datapath+'params.npy').st_size==0: 
        #     print(ID + ' -- params does not exist / empty!')
        #     return None
        
        
        N, nu, ts, phi, th = get_params(datapath,verbose=False)
        thl = th[0:5:2]
    
        sexp = compute_sexp(texp,gexp,th,scaledist=True)
    
        sexp_er = cleanexp(sexp,True,thl) 
        
        s = import_alltime(datapath,readfit=False)
        
        s_er = cleansim(s,texp,nu,thl,at=False)
        
        delta = error(sexp_er,s_er,which,t_range)
    
        return delta
    
    except (ValueError,FileNotFoundError,OSError):
        print(ID + ' has some problem')
        return None

def build_err_array(simset,grid,which='mahalanobis',t_range=[i for i in range(6)],grid_none=False):
    
    # compute lenght for each dimensions 
    dims = [] 
    for i in range(np.array(grid).shape[1]):
        dims.append(np.unique(np.array(grid)[:,i],return_counts=True)[0].size) 
    print("[ err ] has dimensions: (", *dims, ')  ----  # sims = %i' %len(grid))
    
    err = np.zeros(dims) 
    
    texp, gexp = import_gtexp()
    
    grid_None = []
    
    idx = 0; e_min=1e10
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                e = err_one(simset,grid[idx],texp,gexp,which,t_range)
                if e == None:
                    grid_None.append(grid[idx])
                err[i,j,k] = e
                if e !=None and e < e_min:
                    e_min = e
                    grid_min = grid[idx]
                idx += 1
                if idx%500 == 0:
                    print("%.1f" %(idx/len(grid)*100) + "% examined" )                        
    print("")                    
    print('err_min = %.4f' %np.nanmin(err))
    print("")
    print('set = ', simset)
    print('ID_min = ', 'M%i-rf%.2f-seed%i' %grid_min )
    
    if len(t_range)==6:
        tlab = 'allt'
    elif t_range[0]==-1:
        tlab = 'fin'
    else:
        tlab = str(t_range)
    # Save errs
    np.save('./output/' + simset + '/err_' + which[:3] + '_' + tlab + '.npy', err )
    
    if grid_none==True:
        return err, grid_None
    else:
        return err

#%% graph stats

import networkx as nx


ngofstats = 7
gofstats = ["twopaths","triangles","av_clustering","transitivity","loc_efficiency","glob_efficiency","s-metric"]
maxdeg = 70


def eval_stats_core(A,to_adj=False,maxdeg=maxdeg):
    if to_adj:
        A = sim.to_adj_mat(A)
    
    # to networkX
    net = nx.from_numpy_matrix(A)
    # remove isolated nodes 
    net.remove_nodes_from(list(nx.isolates(net)))
    if nx.is_connected(net):
        largest_cc = net
    else:
        largest_cc = net.subgraph(max(nx.connected_components(net), key=len)).copy()
        
    all_stats = np.array([
            # ergm-like
            sim.ergm.twopaths(A),
            sim.ergm.triangles(A),
            # clustering
            nx.average_clustering(net),
            nx.transitivity(net),
            # efficiency
            nx.local_efficiency(net),
            nx.global_efficiency(net),
            nx.s_metric(net,False),
            ])
    
    deg_hist = np.zeros((maxdeg))
    temp = np.array(nx.degree_histogram(net))
    deg_hist[:temp.size] = temp
    
    return all_stats, deg_hist
    
def eval_stats(cl, ncl=1, maxdeg=maxdeg, datapath = '', label = ''):
    if len(cl.shape)==2:
        W = cl.shape[0] 
    elif len(cl.shape)==1: 
        W = 1
        cl = cl.reshape(1,len(cl))
        
    stats = np.zeros((W,ngofstats))
    degs = np.zeros((W,maxdeg))
    
    for idx in range(W):
        stats[idx], degs[idx] = eval_stats_core(cl[idx],to_adj=True,maxdeg=maxdeg)
        
        if idx%20 == 0 and idx !=0:
            print("%.1f" %(idx/W*100) + "% examined" ) 
              
    stats = np.repeat(stats, ncl, axis=0) 
    degs = np.repeat(degs, ncl, axis=0)
        
    gof = np.hstack((stats,degs))
    
    return gof

def compute_gof(datapath,gexp,compute_gof_stats=True):
        
    if compute_gof_stats:
        print('Computing gof_exp')
        gof_exp = eval_stats(gexp[0:], ncl=1, maxdeg=maxdeg, datapath = datapath, label='exp')
        np.save(datapath + 'gof_exp.npy', gof_exp)
        
        print('Computing gof_sim -- this will take a while')
        gof_sim = []
        for acode in agecode[1:-1]:
            print('T = %i' %ages[acode])
            cl_t, ns_t = import_insta(datapath,acode,clones=True)[0:2]
            gof_sim.append(
                eval_stats(cl_t, ncl=ns_t, maxdeg=maxdeg, datapath=datapath, label='T'+str(ages[acode]))
                )
        gof_sim = np.array(gof_sim)
        np.save(datapath + '/gof_sim.npy', gof_sim)
    else:
        gof_exp = np.load(datapath + 'gof_exp.npy',allow_pickle=True)
        gof_sim = np.load(datapath + 'gof_sim.npy',allow_pickle=True)
    return gof_exp, gof_sim

def rel_err(true,sample_av):
    return (sample_av-true)/true
    
def compute_rel_errs(exp,sim,null):
    nt, M, ngofstats = np.shape(sim)
    exp = np.vstack((exp[:-2],np.mean(exp[-2:],axis=0)))

    re_sim = np.zeros((nt, M, ngofstats))
    re_null = np.zeros((nt, M, ngofstats))

    for i in range(nt):
        for j in range(M):
            for k in range(ngofstats):
                re_sim[i,j,k]= rel_err(exp[i,k],sim[i,j,k])
                re_null[i,j,k]= rel_err(exp[i,k],null[i,j,k])

    return re_sim, re_null
























    