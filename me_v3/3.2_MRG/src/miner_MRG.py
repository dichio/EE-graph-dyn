#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 14:34:13 2023

@author: vito.dichio
"""

import os

import src.building_blocks_MRG as sim

from scipy.optimize import curve_fit
import numpy as np

myseed = 160318; np.random.seed(myseed)

from scipy.integrate import solve_ivp

#%%
def shout_paths(ID,simset=None):

    if simset != None:
        setpath = './output/' + simset + '/'
    else:
        setpath = './output/'
    
    datapath = setpath + ID + '/'
    
    return setpath, datapath

def get_params(datapath, verbose=True):
    
    params = np.load(datapath + 'params.npy',allow_pickle=True)

    #T, N, M, nu, mu, phi, E_in, E_trgt = sim.init(params,verbose=True)[0:-1]
    T, N, L, M, nu, mu, phi, E_in, E_trgt = sim.init(params,verbose=verbose)[0:-1]
    
    ts = np.arange(0,T+1)/nu
    
    return T, ts, N, L, M, nu, mu, phi, E_in, E_trgt

def import_alltime(datapath):
    gstats = np.load(datapath + '/sts.npy', allow_pickle=True)    
    corrs = np.load(datapath + '/corrs.npy', allow_pickle=True)
    fsts = np.load(datapath + '/fsts.npy', allow_pickle=True)
    allf = np.load(datapath + '/allf.npy', allow_pickle=True)
    return fsts, allf, gstats, corrs

def exact_solution(T, L, nu, mu, phi, E_in, E_trgt):
    mu = mu*nu 
    phi = phi*nu
    
    def f(t,x):
        return -2*mu*x-phi/L**2*(1-x**2)*((L-1)/2*x+L/2-E_trgt) #flip

    y0 = [2*E_in/L-1]
    
    tspan = (0.0, T/nu)

    sol = solve_ivp(f, tspan, y0)
    
    return sol.t, (1+sol.y[0])/2

def exact_solution_agd(T, L, nu, mu, phi, E_in, E_trgt):
    mu = mu*nu 
    phi = phi*nu
    
    def f(t,x):
        return -2*mu*x-phi/L**2*(1-x**2)*((L-1)/2*x+L/2-E_trgt) #flip

    y0 = [2*E_in/L-1]
    
    tspan = (0.0, T/nu)

    sol = solve_ivp(f, tspan, y0)
    
    return (1+sol.y[0,-1])/2


def get_comp_time(simset,gridv):
    ID = 'N%i-T100-M%i-nu1-mu%.1e-rf%.1e-d_in0.50-d_trgt%.2f-seed%i' % gridv
    datapath = shout_paths(ID,simset)[1]
    with open(datapath +'comptime.txt') as f:
        ct = f.readlines()[0]
    return ct

def build_comp_time(simset,grid):
    dims = [] 
    for i in range(np.array(grid).shape[1]):
        dims.append(np.unique(np.array(grid)[:,i],return_counts=True)[0].size) 
    print("[ comp_time ] has dimensions: (", *dims, ')  ----  # sims = %i' %len(grid))
        
    comp_time = np.zeros(dims)

    idx = 0
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                for l in range(dims[3]):
                    for r in range(dims[4]):
                        for q in range(dims[5]):
                            ct = get_comp_time(simset,grid[idx])
                            comp_time[i,j,k,l,r,q] = ct
                            idx +=1

    comp_time_av = np.nanmean(comp_time,-1)
    comp_time_sd = np.nanvar(comp_time,-1)**0.5
    
    return comp_time_av, comp_time_sd

def err_correlations(corrs,L):
    return L*np.mean(abs(np.mean(corrs,axis=1)))

def get_corr_err(simset,gridv):
    
    L = gridv[0]*(gridv[0]-1)/2
    
    ID = 'N%i-T100-M%i-nu1-mu%.1e-rf%.1e-d_in0.50-d_trgt%.2f-seed%i' % gridv
    datapath = shout_paths(ID,simset)[1]
    corrs = np.load(datapath + 'corrs.npy',allow_pickle=True)
    c_err = err_correlations(corrs,L)
    
    return c_err

def err_sol(x,mu,sd):
    return abs(x-mu)/sd
    

def get_sol_err(simset,gridv):
    
    ID = 'N%i-T100-M%i-nu1-mu%.1e-rf%.1e-d_in0.50-d_trgt%.2f-seed%i' % gridv
    datapath = shout_paths(ID,simset)[1]
    
    T, ts, N, L, M, nu, mu, phi, E_in, E_trgt = get_params(datapath,verbose=False)

    s = import_alltime(datapath)[1]/L

    s_ex = exact_solution(T, L, nu, mu, phi, E_in, E_trgt)[1]
    
    return err_sol(s_ex[-1],s[-1,0],s[-1,1])


def build_corr_err(simset,grid):
    dims = [] 
    for i in range(np.array(grid).shape[1]):
        dims.append(np.unique(np.array(grid)[:,i],return_counts=True)[0].size) 
    print("[ comp_time ] has dimensions: (", *dims, ')  ----  # sims = %i' %len(grid))
        
    corr_err = np.zeros(dims)

    idx = 0
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                for l in range(dims[3]):
                    for r in range(dims[4]):
                        for q in range(dims[5]):
                            c_err = get_corr_err(simset,grid[idx])
                            corr_err[i,j,k,l,r,q] = c_err
                            idx +=1

    corr_err_av = np.nanmean(corr_err,-1)
    corr_err_sd = np.nanvar(corr_err,-1)**0.5
    
    return corr_err_av, corr_err_sd

def build_sol_err(simset,grid):
    dims = [] 
    for i in range(np.array(grid).shape[1]):
        dims.append(np.unique(np.array(grid)[:,i],return_counts=True)[0].size) 
    print("[ comp_time ] has dimensions: (", *dims, ')  ----  # sims = %i' %len(grid))
        
    sol_err = np.zeros(dims)

    idx = 0
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                for l in range(dims[3]):
                    for r in range(dims[4]):
                        for q in range(dims[5]):
                            s_err = get_sol_err(simset,grid[idx])
                            sol_err[i,j,k,l,r,q] = s_err
                            idx +=1

    sol_err_av = np.nanmean(sol_err,-1)
    sol_err_sd = np.nanvar(sol_err,-1)**0.5
    
    return sol_err_av, sol_err_sd

def get_runs(simset,x,which='M',IDs=None):
    
    # Reference simulation
    names = ["N","T","M","nu","mu","rho","d_in","d_trgt","seed"]
    ref_sim = [10,50,2**12,100,2.2e-2,5e3,1.0,0.07,160318]
    
    params = []; s = []; t = []; s_ex = []; t_ex = []
    
    print("There are %i %s values"%(len(x),which))
    
    for p in range(len(x)):
        
        if IDs == None:
            sim_par = ref_sim.copy() 
            sim_par[names.index(which)] = x[p]
        
            params.append(sim_par)
        
            ID = 'N%i-T%i-M%i-nu%i-mu%.1e-rf%.1e-d_in%.2f-d_trgt%.2f-seed%i' %tuple(sim_par)
        else:
            ID = IDs[p]
            
        datapath = shout_paths(ID,simset)[1]

        T, ts, N, L, M, nu, mu, phi, E_in, E_trgt = get_params(datapath,verbose=False)

        t.append(ts)
        s.append(import_alltime(datapath)[2]/L)

        t_ex_one, s_ex_one = exact_solution(T, L, nu, mu, phi, E_in, E_trgt)
        
        t_ex.append(t_ex_one)
        s_ex.append(s_ex_one)
        
    return ref_sim, np.array(params), np.array(t), np.array(s), np.array(t_ex), np.array(s_ex)
        










#%% PLOTS

import matplotlib.pyplot as plt
import seaborn as sns

myfontsize = 6
mylabelsize = 6
cm = 1/2.54  # centimeters in inches

def density_one(t_ex,s_ex,t,s,d_trgt,save):
    
    sns.set_theme()
    fig, ax = plt.subplots(figsize=(4.3*cm,4.3*cm))
    
    plt.axhline(0.5,linestyle=':',linewidth=2, label='random',color="#2C5F2D",zorder=1)
    plt.axhline(d_trgt,linestyle=':',linewidth=2, label='target',color="#850000")
    
    
    plt.fill_between(t,s[:,0]-2*s[:,1],s[:,0]+2*s[:,1],alpha=.5,color='#6DA9E4')
    plt.plot(t,s[:,0], linewidth=1, color='#234E70',label=r'$\langle x\rangle_t/L$')

    plt.plot(t_ex,s_ex,linestyle='--',linewidth=1, label='approx',color="k")

    plt.xlabel('t',fontsize=myfontsize)
    ax.xaxis.set_label_coords(1.1, 0.)
    
    print("err_sol = %.2e" %err_sol(s_ex[-1],s[-1,0],s[-1,1])) 
    
    plt.legend(loc=1,fontsize=myfontsize,ncol=1)
    plt.xlim(-1,t_ex[-1])
    plt.ylim(0,1.05)
    plt.xticks(fontsize=mylabelsize)
    plt.yticks(fontsize=mylabelsize)
    
    if save != None:
        plt.savefig('output/'+save+'/one_run.png', dpi=1200, bbox_inches = "tight") 

    return
    
def fit_distribs(ts,allf,myts,save):
    
    idxs = np.where(np.isin(ts, myts))[0]
    
    f = allf[idxs]
    
    avf = np.mean(f,axis=1)
    
    colorz = sns.color_palette("deep",len(f)+1).as_hex()
    colorz_1 = sns.color_palette("dark",len(f)+1).as_hex()

    #colorz = sns.color_palette("colorblind",len(f)+1).as_hex()
    
    fig, ax = plt.subplots(figsize=(12.9*cm,4.3*cm))
    
    for i in range(len(f)):
        
        sns.histplot(f[i],
                 stat='probability',
                 binwidth=.018,
                 #kde=True,
                 color=colorz[i],
                 label = 't=%i'%myts[i])
        plt.vlines(x = avf[i], ymin = 0, ymax = 0.05, lw=2, linestyle='-',color=colorz_1[i])
        #ax.axvline(avf[i],0,color='w')
        
    ax.axvline(x=0, color=colorz[-1], lw=3, linestyle=':')#,label=r'$\Delta t \ F_{max}$')
    
    plt.legend(fontsize=mylabelsize,ncol=4)
    
    plt.xlim(right=1e-2)
    plt.ylim([0,.7])
    
    
    plt.ylabel('')
    plt.xlabel(r'$\Delta t\varphi F$',fontsize=myfontsize)
    ax.xaxis.set_label_coords(1.05, 0.)
    
    plt.xticks(fontsize=mylabelsize)
    #ax.set_xticks([-0.3,-0.15,0.0])
    plt.yticks(fontsize=mylabelsize)
    
    if save != None:
        plt.savefig('output/'+save+'/fit_hists.png', dpi=1200, bbox_inches = "tight") 

    return
    
    
def plot_corrs(t,cs,L,absval=True):
    
    fig, ax = plt.subplots()
    
    #for i in range(cs.shape[1]):    
    #    plt.plot(t,cs[:,i], label='simulations', color="#FF6E31",alpha=0.01)
        
    plt.axhline(0,linestyle='--',linewidth=3,color="k")
    if absval==True:
        plt.plot(t,abs(L*np.mean(cs,axis=1)),color="#AA5656",linewidth=2,label='$|\ \sum_{i<j}\chi_{ij}\ |$')
    else:
        plt.plot(t,L*np.mean(cs,axis=1),color="#AA5656",linewidth=2,label='$|\ \sum_{i<j}\chi_{ij}\ |$')
    plt.axhline(err_correlations(cs,L),linestyle=':',linewidth=3,color="#0A2647", label = '$\delta$ = %.1e' %err_correlations(cs,L))
    
    plt.xlabel('t')
    plt.legend()






def plot_comp_time(grid,av,sd,which='M',save=None):
    
    N = np.unique(np.array(grid)[:,0])
    M = np.unique(np.array(grid)[:,1])
    
    if which=='M':
        
        def func_fit(x,c1):
            return c1 * x 
        
        colors = sns.color_palette("Set2").as_hex()
        for nn in range(len(N)):
            plt.errorbar(M,av[nn,:,0,0,0], yerr=sd[nn,:,0,0,0], fmt="o", label='N=%i'%N[nn])
            #plt.plot(M,av[nn,:,0,0,0], linewidth=0.7,linestyle="-.",color='k')
            
            popt = curve_fit(func_fit, M, av[nn,:,0,0,0])[0]
            plt.plot(M,func_fit(M,popt[0]),linestyle=':',label='$y = %.2e * M$'%(popt[0]),color=colors[nn])
            
        #plt.xscale('log'); plt.yscale('log')
        plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        plt.xlabel('M',fontsize=myfontsize)
        plt.ylabel('[s]',fontsize=myfontsize)
        plt.legend()
        if save != None:
            plt.savefig('output/'+save+'/comp_time_vs_M.png', dpi=1200, bbox_inches = "tight") 
        plt.close()
        
    if which=='N':
        
        def func_fit(x,c1,c2):
            return c1 * x**c2
        
        colors = sns.color_palette().as_hex()

        for mm in range(len(M)):
            plt.errorbar(N,av[:,mm,0,0,0], yerr=sd[:,mm,0,0,0],ms=5, fmt="o",color=colors[mm],label='$M=$%i'%M[mm])
            
            popt = curve_fit(func_fit, N, av[:,mm,0,0,0])[0]
            plt.plot(N,func_fit(N,popt[0],popt[1]),linestyle=':',label='$y \propto N^{%.2f}$'%(popt[1]),color=colors[mm])
            #plt.plot(N,av[:,mm,0,0,0], linewidth=0.7,linestyle="-.",color='k')
            
            
        plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        plt.xlabel('N',fontsize=myfontsize)
        plt.ylabel('[s]',fontsize=myfontsize)
        plt.legend(fontsize=mylabelsize)
        if save != None:
            plt.savefig('output/'+save+'/comp_time_vs_N.png', dpi=1200, bbox_inches = "tight") 
        plt.close()

def plot_corr_err(grid,av,sd,which='M',save=None):
    
    N = np.unique(np.array(grid)[:,0])
    M = np.unique(np.array(grid)[:,1])
    d_trgt = np.unique(np.array(grid)[:,4])
    
    if which=='M':
        for nn in range(len(N)):
            plt.errorbar(M,av[nn,:,0,0,0], yerr=sd[nn,:,0,0,0], fmt="", label='N=%i'%N[nn])
            plt.plot(M,av[nn,:,0,0,0], linewidth=1.0)
        
        plt.xscale('log'); plt.yscale('log')
        #plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        plt.xlabel('M',fontsize=myfontsize)
        plt.ylabel('$\delta$',fontsize=myfontsize)
        plt.legend()
        if save != None:
            plt.savefig('output/'+save+'/corr_err_vs_M.png', dpi=1200, bbox_inches = "tight") 
        plt.close()
        
    if which=='N':
        for mm in range(len(M)):
            plt.errorbar(N,av[:,mm,0,0,0], yerr=sd[:,mm,0,0,0], fmt="o", label='M=%i'%M[mm])
            plt.plot(N,av[:,mm,0,0,0], linewidth=0.7,linestyle="-.",color='k')
        
        #plt.xscale('log'); plt.yscale('log')
        plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        plt.xlabel('N',fontsize=myfontsize)
        plt.ylabel('$\delta$',fontsize=myfontsize)
        plt.legend()
        if save != None:
            plt.savefig('output/'+save+'/corr_err_vs_N.png', dpi=1200, bbox_inches = "tight") 
        plt.close()
        
    if which=='d_trgt':
        for mm in range(len(M)):
            plt.errorbar(d_trgt,av[0,mm,0,0,:], yerr=sd[0,mm,0,0,:], fmt="o", label='M=%i'%M[mm])
            #plt.plot(d_trgt,av[0,mm,0,0,:], linewidth=0.7,linestyle="-.",color='k')
        
        #plt.xscale('log'); plt.yscale('log')
        plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        plt.xlabel('d_trgt',fontsize=myfontsize)
        plt.ylabel('$\delta$',fontsize=myfontsize)
        plt.legend()
        if save != None:
            plt.savefig('output/'+save+'/corr_err_vs_d-trgt.png', dpi=1200, bbox_inches = "tight") 
        plt.close()
        
    if which=='M/2^L':
        colors=['#00235B','#E21818','#FFDD83','#98DFD6']
        for mm in range(len(M)):
            for nn in range(1):
                ratio = M[mm]/2**(N[nn]*(N[nn]-1)/2)
                plt.errorbar(ratio,av[nn,mm,0,0,0],sd[nn,mm,0,0,0],fmt='o',label='%i'%M[mm])#,label='%i,%i'%(mm,nn))
        
        #plt.xscale('log'); plt.yscale('log')
        plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        plt.xlabel('M/2^L',fontsize=myfontsize)
        plt.ylabel('$\delta$',fontsize=myfontsize)
        plt.legend()
        if save != None:
            plt.savefig('output/'+save+'/corr_err_vs_MoverL.png', dpi=1200, bbox_inches = "tight") 
        plt.close()     
    
def plot_sol_err(grid,av,sd,which='M',save=None):
    
    N = np.unique(np.array(grid)[:,0])
    M = np.unique(np.array(grid)[:,1])
    d_trgt = np.unique(np.array(grid)[:,4])
        
    if which=='N':
        for mm in range(len(M)):
            plt.errorbar(N,av[:,mm,0,0,0], yerr=sd[:,mm,0,0,0], fmt="o", label='M=%i'%M[mm])
            #plt.plot(N,av[:,mm,0,0,0], linewidth=0.7,linestyle="-.",color='k')
        
        #plt.xscale('log'); plt.yscale('log')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.xlabel('N',fontsize=myfontsize)
        plt.ylabel('$\delta$',fontsize=myfontsize)
        plt.legend()
        if save != None:
            plt.savefig('output/'+save+'/sol_err_vs_N.png', dpi=1200, bbox_inches = "tight") 
        plt.close()
        
    if which=='d_trgt':
        for mm in range(len(M)):
            plt.errorbar(d_trgt,av[0,mm,0,0,:], yerr=sd[0,mm,0,0,:], fmt="o", label='M=%i'%M[mm])
            #plt.plot(d_trgt,av[0,mm,0,0,:], linewidth=0.7,linestyle="-.",color='k')
        
        #plt.xscale('log'); plt.yscale('log')
        plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        plt.xlabel('d_trgt',fontsize=myfontsize)
        plt.ylabel('$\delta$',fontsize=myfontsize)
        plt.legend()
        if save != None:
            plt.savefig('output/'+save+'/sol_err_vs_d-trgt.png', dpi=1200, bbox_inches = "tight") 
        plt.close()
        
    if which=='M/2^L':
        for mm in range(len(M)):
            for nn in range(1):
                ratio = M[mm]/2**(N[nn]*(N[nn]-1)/2)
                plt.errorbar(ratio,av[nn,mm,0,0,0],sd[nn,mm,0,0,0],fmt='o',label='%i'%M[mm])#,label='%i,%i'%(mm,nn))
        
        #plt.xscale('log'); plt.yscale('log')
        plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        plt.xlabel('M/2^L',fontsize=myfontsize)
        plt.ylabel('$\delta$',fontsize=myfontsize)
        plt.legend()
        if save != None:
            plt.savefig('output/'+save+'/sol_err_vs_MoverL.png', dpi=1200, bbox_inches = "tight") 
        plt.close()     

def plot_runs(ref_sim,simset,params,t,s,t_ex,s_ex,x,which='M',save=None):
    
    if which == 'M':
        L = ref_sim[0]*(ref_sim[0]-1)/2
        d_trgt = int(L*ref_sim[7])/L   
        
        sns.set_theme()
        fig, ax = plt.subplots(figsize=(3, 3))
        
        #runs
        nruns = s.shape[0]
        
        colors = sns.color_palette("tab10").as_hex()
        
        for r in range(nruns):
            if x[r]==ref_sim[2]:
                plt.plot(t[r],s[r,:,0], linewidth=6, color='k')
                plt.plot(t[r],s[r,:,0], linewidth=4, color='w')
                plt.fill_between(t[r,:],s[r,:,0]-s[r,:,1],s[r,:,0]+s[r,:,1],alpha=0.1,color='k',zorder=1)
            else:
                plt.plot(t[r,:],s[r,:,0], linewidth=1, label='$M=2^{%i}$'%(int(np.log(x[r])/np.log(2))), color=colors[r])
                plt.fill_between(t[r,:],s[r,:,0]-s[r,:,1],s[r,:,0]+s[r,:,1],alpha=0.1,color=colors[r],zorder=1)
    
        plt.plot(t_ex[0],s_ex[0],linestyle='--',linewidth=2, label='independent',color="k")
    
        plt.axhline(d_trgt,linestyle=':',linewidth=4, label='target',color="#850000")
        
        #plt.xlabel('$T$',fontsize=myfontsize)    
        #print("err_sol = %.2e" %err_sol(s_ex[-1],s[-1,0],s[-1,1])) 
        
        plt.legend(loc=1,ncol=1,fontsize=myfontsize)
        plt.ylim(0,0.55)
        plt.xlim([-1,50])

        if save != None:
            plt.savefig('output/'+save+'/var_M.png', dpi=1200, bbox_inches = "tight") 
    
        return
        
    if which == 'd_trgt':
        
        sns.set_theme()
        fig, ax = plt.subplots(figsize=(3, 3))
        
        #runs
        nruns = s.shape[0]
        colors = sns.color_palette("deep", 8).as_hex()

        for r in range(nruns):
            
            L = ref_sim[0]*(ref_sim[0]-1)/2
            d_trgt = int(L*params[r,7])/L
            
            if x[r]==ref_sim[7]:
                plt.plot(t[r],s[r,:,0], linewidth=6, color='k')
                plt.plot(t[r],s[r,:,0], linewidth=4, color='w')
                plt.fill_between(t[r,:],s[r,:,0]-s[r,:,1],s[r,:,0]+s[r,:,1],alpha=0.1,color='k')
                #plt.plot(t_ex[r],s_ex[r],linestyle='-.',linewidth=.5,color="k",zorder=1)
                plt.axhline(d_trgt,linestyle=':',linewidth=2,color='k',zorder=1)

            else:
                plt.plot(t[r,:],s[r,:,0], linewidth=1, color=colors[r],label='$d_{\ trgt} = %.1f$'%x[r])
                plt.fill_between(t[r,:],s[r,:,0]-s[r,:,1],s[r,:,0]+s[r,:,1],alpha=0.2,color=colors[r])
                #plt.plot(t_ex[r],s_ex[r],linestyle='-.',linewidth=.5,color=colors[r],zorder=1)
                plt.axhline(d_trgt,linestyle=':',linewidth=2,color=colors[r],zorder=1)
            
        plt.xlim([-1,50])
                
        #plt.legend(loc=9,ncol=3)
        
        if save != None:
            plt.savefig('output/'+save+'/var_d_trgt.png', dpi=1200, bbox_inches = "tight") 
    
        return
    
    if which == 'mu':
        
        sns.set_theme()
        fig, ax = plt.subplots(figsize=(3, 3))
        
        #runs
        nruns = s.shape[0]
        #colors = sns.color_palette("deep", 8).as_hex()
        colors = sns.color_palette("hls",n_colors=nruns).as_hex()
        
        for r in range(nruns):
            
            L = ref_sim[0]*(ref_sim[0]-1)/2
            
            d_trgt = int(L*ref_sim[7])/L
            
            if x[r]==ref_sim[4]:
                plt.plot(t[r],s[r,:,0], linewidth=6, color='k')
                plt.plot(t[r],s[r,:,0], linewidth=4, color='w')
                plt.fill_between(t[r,:],s[r,:,0]-s[r,:,1],s[r,:,0]+s[r,:,1],alpha=0.2,color='k')
               # plt.plot(t_ex[r],s_ex[r],linestyle='-.',linewidth=.5,color="k",zorder=1)

            else:
                plt.plot(t[r,:],s[r,:,0], linewidth=1, color=colors[r],label='$\mu=10^{%i}$'%int(np.log10(x[r])))
                plt.fill_between(t[r,:],s[r,:,0]-s[r,:,1],s[r,:,0]+s[r,:,1],alpha=0.3,color=colors[r])
                #plt.plot(t_ex[r],s_ex[r],linestyle='-.',linewidth=.5,color=colors[r],zorder=1)
        
        plt.axhline(d_trgt,linestyle=':',linewidth=4, label='target',color="#850000")

        plt.xlim([-1,50])
                
        plt.legend(loc=1,fontsize=myfontsize)
        
        if save != None:
            plt.savefig('output/'+save+'/var_mu.png', dpi=1200, bbox_inches = "tight") 
    
        return
    
    if which == 'rho':
        
        sns.set_theme()
        fig, ax = plt.subplots(figsize=(4.3*cm,4.3*cm))
        
        #runs
        nruns = s.shape[0]
        #colors = sns.color_palette("deep", 8).as_hex()
        colors = sns.color_palette("tab10",n_colors=nruns).as_hex()
        colors.pop(0)
        
        for r in range(nruns):
            
            L = ref_sim[0]*(ref_sim[0]-1)/2
            
            d_trgt = int(L*ref_sim[7])/L
            
            if x[r]==ref_sim[5]:
                plt.fill_between(t[r],s[r,:,0]-2*s[r,:,1],s[r,:,0]+2*s[r,:,1],alpha=.5,color='#6DA9E4')
                plt.plot(t[r],s[r,:,0], linewidth=1, color='#234E70', label=r'$\rho= %.0e$' %x[r])
                colors.insert(r,'#234E70')
                #plt.plot(t[r],s[r,:,0], linewidth=6, color='k')
                #plt.plot(t[r],s[r,:,0], linewidth=4, color='w')
                #plt.fill_between(t[r,:],s[r,:,0]-s[r,:,1],s[r,:,0]+s[r,:,1],alpha=0.2,color='k')
               # plt.plot(t_ex[r],s_ex[r],linestyle='-.',linewidth=.5,color="k",zorder=1)

            else:
                plt.plot(t[r,:],s[r,:,0], linewidth=1, color=colors[r],label=r'$\rho= %.0e$' %x[r])
                plt.fill_between(t[r,:],s[r,:,0]-2*s[r,:,1],s[r,:,0]+2*s[r,:,1],alpha=0.3,color=colors[r])
                #plt.plot(t_ex[r],s_ex[r],linestyle='-.',linewidth=.5,color=colors[r],zorder=1)
        
        plt.axhline(d_trgt,linestyle=':',linewidth=2, label='target',color="#850000")

        plt.xlabel('t',fontsize=myfontsize)
        ax.xaxis.set_label_coords(1.1, 0.)
    
        plt.xticks(fontsize=mylabelsize)
        plt.yticks(fontsize=mylabelsize)
        plt.xlim([-1,50])
                
        plt.legend(loc=1,fontsize=myfontsize)
        
        if save != None:
            plt.savefig('output/'+save+'/var_rho.png', dpi=1200, bbox_inches = "tight") 
    
        return
    
    
    
    
    
