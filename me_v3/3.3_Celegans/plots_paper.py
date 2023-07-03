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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import matplotlib.lines as mlines

# names
statname = ["edges","gwdegree","gwesp","dist"]

# plot parameters
mytitlesize = 8
myfontsize = 7
mylabelsize = 7

cm = 1/2.54

#%% import
simset = 'one-sim'
ID = 'M1024-rf901.71-seed758052841'

setpath, datapath, plotpath = miner.shout_paths(ID,simset)

N, nu, ts, phi, th = miner.get_params(datapath,verbose=True)
thl = th[0:5:2]

gexp, sexp, fexp, texp = miner.import_experimental(N,phi,th,scaledist=True)
sexp_er = miner.cleanexp(sexp,True,thl) 

fs, s = miner.import_alltime(datapath,readfit=True)
s_er, s_at = miner.cleansim(s,texp,nu,thl)

#%%

simset_nm = 'one-sim'
ID_nm = 'M1024-rf0.00-seed758052841'

setpath_nm, datapath_nm, plotpath_nm = miner.shout_paths(ID_nm,simset_nm)

phi_nm = miner.get_params(datapath_nm,verbose=True)[3]
thl = th[0:5:2]

fs_nm, s_nm = miner.import_alltime(datapath_nm,readfit=True)
s_er_nm, s_at_nm = miner.cleansim(s_nm,texp,nu,thl)


#%%
    
def fit_vs_t(plotpath,nu,texp,fexp,ts,fs,fs_er,save=False):
    
    mylabelsize = 6
    
    c0 = sns.color_palette("tab10").as_hex()[1]
    c1 = sns.color_palette("dark").as_hex()[3]
    
    plt.style.use('default')

    fig, ax = plt.subplots(figsize=(4.*cm,2.5*cm))
    
    # data
    ax.plot(ts, fs, '--', color = c1,zorder=0)
    #ax.fill_between(ts, fs-fs_er , fs+fs_er, alpha=1)
    ax.scatter(texp[0],fexp[0], s=25, marker = 'o', edgecolor='k', lw=.5,color='w',zorder=2)
    ax.scatter(texp[-1],fexp[-1], s=25, marker = 'o', edgecolor='k', lw=.5,color='w',zorder=2)
    ax.scatter(texp[-2],fexp[-2], s=25, marker = 'o', edgecolor='k', lw=.5,color='w',zorder=2)

    ax.scatter(texp[1:-2], 
               fexp[1:-2], 
               s=25, 
               marker = 'o', 
               edgecolor='k', 
               lw=.5,
               color='w',
               label = r'$\mathbf{F (G^*})$',
               zorder=2)
    
    ax.scatter(texp, 
               fs[texp*nu],
               s=15, 
               marker = 'd',
               edgecolor='black', 
               lw=.5,
               color = c1, 
               label = r'$\mathbf{\langle F  \rangle_{t^*}}$',
               zorder=3 )

    # aesthetics
    plt.legend(loc=2, ncol=1, fontsize = mylabelsize, bbox_to_anchor=(0.1, 1.35))
    
    ax.yaxis.get_offset_text().set_fontsize(mylabelsize)
    ax.xaxis.get_offset_text().set_fontsize(mylabelsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.set_xlabel('$t$ [h]', loc='right',fontsize = myfontsize)
    plt.xticks(fontsize=mylabelsize) ; plt.yticks(fontsize=mylabelsize)
    
    xticks = [0,5,8,16,23,27,45]
    xticks_labels = ["0h","","","16h","","27h","45h"]
    plt.xticks(xticks,xticks_labels)
    #plt.xlabel('$t\ [h]$',fontsize=myfontsize)
    ax.xaxis.set_label_coords(.95, 0.2)

    # save
    if save != False:
        plt.savefig(plotpath+'/fit_allt---'+save+'.png', dpi=1200, bbox_inches = "tight",transparent=True) 
    
    return


fit_vs_t( plotpath,
                nu,
                texp,
                fexp,
                ts,
                fs[:,0],
                fs[:,1]**(1/2),
                save=ID
                )



#%%

def stats_1vs1(plotpath,texp,s1exp,s2exp,t,s1,s2,s1_nm,s2_nm,nu,title="notitle",save=False):
    
    mylabelsize = 7
    myfontsize = 8
    
    #sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(6.6*cm,6.6*cm))
    plt.style.use('default')
    
    # data
    mycm = sns.color_palette("viridis_r", as_cmap=True)
    
    ax.plot(s1, s2, ls='--',lw=1,c='k',zorder=0)

    ax.scatter(s1exp, 
                s2exp, 
                c=texp, 
                cmap=mycm, 
                s=50, 
                marker = 'o', 
                edgecolor='black', 
                linewidth=.5,
                label=r'$\mathbf{x(G^*)}$'
                )
    
    ax.scatter(s1[texp*int(nu)],
                s2[texp*int(nu)],
                s=50, 
                marker = 'd',
                edgecolor='black', 
                linewidth=.5,
                c=texp,
                cmap=mycm,
                label=r'$\mathbf{\langle x\rangle_{t^*}}, \rho=\rho^*$')
    
    ax.plot(s1_nm, s2_nm, ls=':',lw=.5,color='black',zorder=0,alpha=0.5)
    
    ax.scatter(s1_nm[texp*int(nu)], 
                s2_nm[texp*int(nu)], 
                c=texp, 
                cmap=mycm, 
                s=20, 
                edgecolor='black', 
                linewidth=.5,
                marker = 's',
                alpha=.5,
                label=r'$\mathbf{\langle x\rangle_{t^*}}, \rho=0$')
    
    # aesthetics
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    ax.yaxis.get_offset_text().set_fontsize(mylabelsize)
    ax.xaxis.get_offset_text().set_fontsize(mylabelsize)

    ax.yaxis.tick_right()
    ax.tick_params(axis='both', which='both', labelsize=mylabelsize)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)

    exp = mlines.Line2D([], [], color='#D4D4D4', marker='o', markersize=6, markeredgewidth=.5, markeredgecolor='k', ls='', label=r'$\mathbf{x(G^*)}$')
    sim = mlines.Line2D([], [], color='#D4D4D4', marker='d', markersize=6, markeredgewidth=.5, markeredgecolor='k', ls='', label=r'$\mathbf{\langle x\rangle_{t^*}}, \rho=\rho^*$')
    null = mlines.Line2D([], [], color='#D4D4D4', marker='s',markersize=4., markeredgewidth=.5, markeredgecolor='k', ls='', label=r'$\mathbf{\langle x\rangle_{t^*}}, \rho=0$')

    # etc etc
    plt.legend(handles=[exp, sim, null],loc='lower right',fontsize=mylabelsize,ncol=3,bbox_to_anchor=(1.25, -.35)) 

    # Time-bar
    norm = plt.Normalize(vmin=0, vmax=45)
    sm = plt.cm.ScalarMappable(cmap=mycm, norm=norm)
    sm.set_array([])
    
    cax = fig.add_axes([1.025, 0.1, 0.01, 0.8])  # [left, bottom, width, height]
    
    cbar = ax.figure.colorbar(sm,
                              shrink=.75,
                              ticks=texp,
                              cax=cax,
                              orientation='vertical')
    
    cbar.set_ticklabels([str(t)+'h' for t in texp])
    cbar.ax.tick_params(labelsize=mylabelsize)

    
    # save
    if save != False:
        plt.savefig(plotpath+'/'+title+'---'+save+'.png', dpi=1200, bbox_inches = "tight", transparent=True)
                    
    return


i = 1 ; j = 2
stats_1vs1(plotpath,
                   texp,
                   sexp[:,i], 
                   sexp[:,j],
                   ts,
                   s_at[:,i], 
                   s_at[:,j],
                   s_at_nm[:,i], 
                   s_at_nm[:,j],                   
                   nu,
                   title = statname[i] + ' vs ' + statname[j],
                   save=ID
                   )


#%%
gof_exp, gof_sim = miner.compute_gof(datapath,gexp,compute_gof_stats = False)
gof_sim_nm = miner.compute_gof(datapath_nm,gexp,compute_gof_stats = False)[1]

#%%

def plot_stat_gof(plotpath,texp,xexp,xsim,xnm,title,save):
    
    mylabelsize = 7
    myfontsize = 7
    
    c0 = sns.color_palette("Paired")[1]
    c1 = sns.color_palette("Paired")[5]
    c2 = sns.color_palette("Paired")[3]
    
    sns.set_style("darkgrid")   
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True,figsize=(3.7*cm,9.*cm))

    ### top
    
    i = 2
    
    ax1.scatter(texp, 
                xexp[:,i],  
                s=20, 
                color='w',
                marker = 'o', 
                edgecolor='black', 
                linewidth=.5,
                label = '$\mathbf{y(G^*)}$'
                )
    
    xsim_av, xsim_sd = np.mean(xsim[:,:,i],1), np.std(xsim[:,:,i],1)    
        
    ax1.scatter(texp[:-1], 
                np.insert(xsim_av, 0, xexp[0,i]),  
                s=20, 
                color=c0,
                marker = 'd', 
                edgecolor='black', 
                linewidth=.5,
                label = r'$\mathbf{\langle y \rangle_{t^*}}$'
                )
    
    xnm_av, xnm_sd = np.mean(xnm[:,:,i],1), np.std(xnm[:,:,i],1)    
        
    ax1.scatter(texp[:-1], 
                np.insert(xnm_av, 0, xexp[0,i]),  
                s=10, 
                color=c0,
                marker = 's', 
                edgecolor='black', 
                linewidth=.5,
                alpha = .5
                )
 
    ax1.set_ylim([.1,.32])
    ax1.set_ylabel('transitivity',fontsize=mylabelsize)
    
    ax1.tick_params(axis='y', labelleft=False, labelright=True, right=False)
    ax1.tick_params(axis='both', which='both', labelsize=mylabelsize)
       
    #ax1.legend(fontsize=mylabelsize,ncol=2,loc=2,bbox_to_anchor=(-.15,1.5))
    #leg = ax1.get_legend()
    #leg.legendHandles[0].set_color('k')
    #leg.legendHandles[1].set_color('k')
    ### center
    
    i = 4
    
    ax2.scatter(texp, 
                xexp[:,i],  
                s=20, 
                color='w',
                marker = 'o', 
                edgecolor='black', 
                linewidth=.5,
                )
    
    xsim_av, xsim_sd = np.mean(xsim[:,:,i],1), np.std(xsim[:,:,i],1)    
        
    ax2.scatter(texp[:-1], 
                np.insert(xsim_av, 0, xexp[0,i]),  
                s=20, 
                color=c1,
                marker = 'd', 
                edgecolor='black', 
                linewidth=.5,
                )

    xnm_av, xnm_sd = np.mean(xnm[:,:,i],1), np.std(xnm[:,:,i],1)    
        
    ax2.scatter(texp[:-1], 
                np.insert(xnm_av, 0, xexp[0,i]),  
                s=10, 
                color=c1,
                marker = 's', 
                edgecolor='black', 
                linewidth=.5,
                alpha = .5
                )
    
    ax2.set_ylim([.15,.6])
    ax2.set_ylabel('local efficiency',fontsize=mylabelsize)
    
    ax2.tick_params(axis='y', labelleft=False, labelright=True, right=False)
    ax2.tick_params(axis='both', which='both', labelsize=mylabelsize)
    
    yticks = [.2,.4,.6]
    ax2.set_yticks(yticks)
    
    ### bottom  
    
    i = 6
    
    ax3.scatter(texp, 
                xexp[:,i],  
                s=20, 
                color='w',
                marker = 'o', 
                edgecolor='black', 
                linewidth=.5,
                )
    
    xsim_av, xsim_sd = np.mean(xsim[:,:,i],1), np.std(xsim[:,:,i],1)    
        
    ax3.scatter(texp[:-1], 
                np.insert(xsim_av, 0, xexp[0,i]),  
                s=20, 
                color=c2,
                marker = 'd', 
                edgecolor='black', 
                linewidth=.5,
                ) 
    
    xnm_av, xnm_sd = np.mean(xnm[:,:,i],1), np.std(xnm[:,:,i],1)    
        
    ax3.scatter(texp[:-1], 
                np.insert(xnm_av, 0, xexp[0,i]),  
                s=10, 
                color=c2,
                marker = 's', 
                edgecolor='black', 
                linewidth=.5,
                alpha = .5
                )
    
    # bottom - aestetics 
    
    ax3.set_ylim([1.,9.99e5])
    ax3.set_ylabel('$S$-metric',fontsize=mylabelsize)
    ax3.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax3.yaxis.get_offset_text().set_fontsize(mylabelsize)
    ax3.get_yaxis().get_offset_text().set_position((.9,0))    

    ax3.tick_params(axis='y', labelleft=False, labelright=True, right=False)
    ax3.tick_params(axis='both', which='both', labelsize=mylabelsize)
    
    xticks = [0,5,8,16,23,27,45]
    xticklabels = ["0h","","","16h","","27h","45h"]
    ax3.set_xticks(xticks)
    ax3.set_xticklabels(xticklabels)
    
    #yticks = [2.0e5,5.0e5,8.0e5]
    #ax3.set_yticks(yticks)


    if save != False:
        plt.savefig(plotpath+'/'+title+'---'+save+'.png', dpi=1200, bbox_inches = "tight")
          


    return


plot_stat_gof(plotpath,
           texp,
           gof_exp, 
           gof_sim, 
           gof_sim_nm,  
           title = 'gof',             
           save=ID
           )


#%%

def plot_stat_gof_sm(plotpath,texp,xexp,xsim,xnm,title,save):
    
    mylabelsize = 7
    myfontsize = 7
    
    c0 = sns.color_palette("Paired")[11]
    c1 = sns.color_palette("Paired")[10]
    
    sns.set_style("darkgrid")   
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,figsize=(4*cm,8.*cm))

    ### top
    
    i = 3
    
    ax1.scatter(texp, 
                xexp[:,i],  
                s=20, 
                color='w',
                marker = 'o', 
                edgecolor='black', 
                linewidth=.5,
                label = '$\mathbf{y(G^*)}$'
                )
    
    xsim_av, xsim_sd = np.mean(xsim[:,:,i],1), np.std(xsim[:,:,i],1)    
        
    ax1.scatter(texp[:-1], 
                np.insert(xsim_av, 0, xexp[0,i]),  
                s=20, 
                color=c0,
                marker = 'd', 
                edgecolor='black', 
                linewidth=.5,
                label = r'$\mathbf{\langle y \rangle_{t^*}}$'
                )
    
    xnm_av, xnm_sd = np.mean(xnm[:,:,i],1), np.std(xnm[:,:,i],1)    
        
    ax1.scatter(texp[:-1], 
                np.insert(xnm_av, 0, xexp[0,i]),  
                s=10, 
                color=c0,
                marker = 's', 
                edgecolor='black', 
                linewidth=.5,
                alpha = .5
                )
 
    ax1.set_ylim([.1,.32])
    ax1.set_ylabel('transitivity',fontsize=mylabelsize)
    
    ax1.tick_params(axis='y', labelleft=False, labelright=True, right=False)
    ax1.tick_params(axis='both', which='both', labelsize=mylabelsize)
       
    yticks = [.1,.2,.3]
    ax1.set_yticks(yticks)

    
    i = 5
    
    ax2.scatter(texp, 
                xexp[:,i],  
                s=20, 
                color='w',
                marker = 'o', 
                edgecolor='black', 
                linewidth=.5,
                )
    
    xsim_av, xsim_sd = np.mean(xsim[:,:,i],1), np.std(xsim[:,:,i],1)    
        
    ax2.scatter(texp[:-1], 
                np.insert(xsim_av, 0, xexp[0,i]),  
                s=20, 
                color=c1,
                marker = 'd', 
                edgecolor='black', 
                linewidth=.5,
                )

    xnm_av, xnm_sd = np.mean(xnm[:,:,i],1), np.std(xnm[:,:,i],1)    
        
    ax2.scatter(texp[:-1], 
                np.insert(xnm_av, 0, xexp[0,i]),  
                s=10, 
                color=c1,
                marker = 's', 
                edgecolor='black', 
                linewidth=.5,
                alpha = .5
                )
    
    ax2.set_ylim([.36,.55])
    ax2.set_ylabel('global efficiency',fontsize=mylabelsize)
    ax2.tick_params(axis='y', labelleft=False, labelright=True, right=False)
    ax2.tick_params(axis='both', which='both', labelsize=mylabelsize)
    

    #ax2.set_yticks(yticks) 

    
    xticks = [0,5,8,16,23,27,45]
    xticklabels = ["0h","","","16h","","27h","45h"]
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticklabels)
    
    #yticks = [2.0e5,5.0e5,8.0e5]
    #ax3.set_yticks(yticks)


    if save != False:
        plt.savefig(plotpath+'/'+title+'---'+save+'.pdf', dpi=1200, bbox_inches = "tight")
          


    return


plot_stat_gof_sm(plotpath,
           texp,
           gof_exp, 
           gof_sim, 
           gof_sim_nm,  
           title = 'gof_sm',             
           save=ID
           )



#%%
degs = gof_sim[:,:,7:]
degs_nm = gof_sim_nm[:,:,7:]
degs_exp = gof_exp[1:,7:]


#%%
from scipy.stats import ks_2samp
from matplotlib.ticker import ScalarFormatter

def cumul_degree(plotpath,degs,degs_nm,degs_exp,save=False):
   
    # Data 
    t_idx = 5
    d = np.mean(degs[t_idx,:,:50],axis=0)
    d_nm = np.mean(degs_nm[t_idx,:,:50],axis=0)
    dexp = degs_exp[t_idx:,:50]
   
    cdf = 1-np.cumsum(d)/np.sum(d)
    cdf_nm = 1-np.cumsum(d_nm)/np.sum(d_nm)
    cdf_exp_0 = 1-np.cumsum(dexp[0])/np.sum(dexp[0])
    cdf_exp_1 = 1-np.cumsum(dexp[1])/np.sum(dexp[1])    
    
    
    D, p = ks_2samp(cdf, cdf_exp_0)
    print("Sim // 1")
    print("KS = %.2f , p = %.2e" %(D, p))
    print("")
    D, p = ks_2samp(cdf, cdf_exp_1)
    print("Sim // 2")
    print("KS = %.2f , p = %.2e" %(D, p))
    print("")
    D, p = ks_2samp(cdf_nm, cdf_exp_0)
    print("nm // 1")
    print("KS = %.2f , p = %.2e" %(D, p))
    print("")
    D, p = ks_2samp(cdf_nm, cdf_exp_1)
    print("nm // 2")
    print("KS = %.2f , p = %.2e" %(D, p))


    # Parameters
    
    plt.style.use('default')
    mylabelsize = 7
    myfontsize = 7
    myc = sns.color_palette("viridis_r", n_colors=45)[-1]
    
    fig, ax = plt.subplots(figsize=(8.*cm,8*cm))


    # Main
    
    ax.plot(cdf_exp_0,
             ls = '-',
             lw = 1.,
             c = myc,
             alpha = 1,
             ) 
    
    ax.plot(cdf_exp_1,
             ls = '-',
             lw = 1.,
             c = myc,
             alpha = 1,
             label = "$\mathbf{G^{*}}_{T}}$", 
             ) 
    
    ax.plot(cdf_nm,
             ls = '--',
             lw = 1.,
             c = 'k',
             alpha = .5,
             label = r"$\langle n.m.\rangle$", 
             )  
    
    ax.plot(cdf,
             ls = '--',
             lw = 1.5,
             c = sns.color_palette("tab10")[1],
             alpha = 1,
             label = r"$\langle sim\rangle$", 
             ) 
    
    # Layout
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='both', labelsize=mylabelsize)
    
    ax.set_xlabel('k',fontsize=myfontsize)
    ax.set_ylabel(r'$P_{deg}^{(c)}(k)$',fontsize=myfontsize)


    plt.legend(loc=3, fontsize=myfontsize)
    
    # Inset
    
    ax1 = plt.axes([0.6, 0.5, 0.4, 0.4])  # [left, bottom, width, height]
    
    ax1.loglog(cdf_exp_0,
             ls = '-',
             lw = 1.,
             c = myc,
             alpha = 1,
             ) 
    
    ax1.loglog(cdf_exp_1,
             ls = '-',
             lw = 1.,
             c = myc,
             alpha = 1,
             label = "$\mathbf{G^{*}}_{T}}$", 
             ) 
    
    ax1.loglog(cdf_nm,
             ls = '--',
             lw = 1.,
             c = 'k',
             alpha = .5,
             label = r"$\langle n.m.\rangle$", 
             )  
    
    ax1.loglog(cdf,
             ls = '--',
             lw = 1.5,
             c = sns.color_palette("tab10")[1],
             alpha = 1,
             label = r"$\langle sim\rangle$", 
             ) 
    
    k_in = 20
    k_fin = 40
    
    # Perform linear regression in log space
    x = np.arange(k_in,k_fin+1,1)
    log_x = np.log(x)
    log_y = np.log(cdf[k_in:k_fin+1])
    coefficients = np.polyfit(log_x, log_y, 1)
    log_y_fit = np.polyval(coefficients, log_x)
    y_fit = np.exp(log_y_fit)
    
    # Plot the fitted line
    ax1.loglog(x,
               y_fit,
               ls = '-',
               lw = .8,
               c = sns.color_palette("tab10")[2],
               label=r'$\gamma=%.2f$'%-coefficients[0])


    ax1.set_xlim([k_in, k_fin])
    ax1.set_ylim([1e-3, None])
    
    ax1.tick_params(axis='both', which='both', labelsize=mylabelsize)

    ax1.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    xticks = [20,30,40]
    ax1.set_xticks(xticks)
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    x_annotate = 35
    y_annotate = .1
    ax1.annotate(r'$P_{deg}^{(c)}\sim k^{-%.2f}$'%-coefficients[0], xy=(x_annotate, y_annotate), xytext=(32, .5),
             arrowprops=dict( arrowstyle='->', color = sns.color_palette("tab10")[2]),fontsize=myfontsize, color=sns.color_palette("tab10")[2])
    
    
    if save != False:
        plt.savefig(plotpath+'/'+save+'.pdf', dpi=1200, bbox_inches = "tight")
          
    return


t = 5

cumul_degree(plotpath,
             degs,
             degs_nm,
             degs_exp,
             save='cumul_degree')


#%%

from scipy import stats

def stat_distrib(simset,ID,sexp,save=False):

    datapath = miner.shout_paths(ID,simset)[1]
    mycm = sns.color_palette("viridis_r", n_colors=45)
    myfontsize = 7
    mylabelsize = 6
    
    fig, axs = plt.subplots(2, 2, figsize=(8.6*cm, 8.6*cm))    

    t=['5','16','27','45'] 

    for i in range(len(t)):
        ns = np.load(datapath + 'T'+t[i]+'_nc.npy',allow_pickle=True); sts = np.load(datapath + 'T'+t[i]+'_sts.npy',allow_pickle=True)[:,1:] 
        s = np.repeat(sts,ns,axis=0)
        
        ax = axs.flatten()[i]
        
        sns.scatterplot(x=s[:, 0], y=s[:, 1], color = mycm[int(t[i])-1] , marker='d', edgecolor='white', linewidth = .6, label = 't=%s'%t[i], ax=ax)
        
        sns.regplot(x=s[:,0], y=s[:,1], scatter=False,
                    line_kws={'color': 'k'},
                    ax=ax 
                    )
        ax.tick_params(axis='both', which='both', labelsize=mylabelsize)
        
        # Perform linear regression
        slope = stats.linregress(s[:,0], s[:,1])[0]
        
        ax.legend(loc=1,fontsize=myfontsize)
        ax.tick_params(axis='both', which='both', labelsize=mylabelsize)

        num_ticks = 4  # Number of ticks you want
        ax.set_yticks(np.rint(np.linspace(min(s[:,1]), max(s[:,1]), num_ticks)))
        num_ticks = 2
        ax.set_xticks(np.rint(np.linspace(min(s[:,0]), max(s[:,0]), num_ticks)))
        
        
    
    axs[0,0].set_ylabel('gwesp',fontsize=mylabelsize)
    axs[1,0].set_ylabel('gwesp',fontsize=mylabelsize)
    axs[1,0].set_xlabel('gwd',fontsize=mylabelsize)
    axs[1,1].set_xlabel('gwd',fontsize=mylabelsize)


    plt.tight_layout()  # Adjust the layout
    if save != False:
        plt.savefig(plotpath+'/'+save+'.pdf', dpi=1200, bbox_inches = "tight")
          
    return

stat_distrib(simset,ID,sexp,save='scatters')








#%% not revised

import pandas as pd

gof_palette = sns.color_palette(palette='bright', n_colors=6)

def gof_violins(which_stat, re_sim, re_null, save=None):
    
    nt, M, ngofstats = np.shape(re_sim)
    k = miner.gofstats.index(which_stat)
    
    colnames = ["5h","8h","16h","23h","27h","45h"]
    df_sim = pd.DataFrame(re_sim[:,:,k].T, columns=colnames)    
    df_null = pd.DataFrame(re_null[:,:,k].T, columns=colnames)    

    fig = plt.figure(figsize=(3,6))  
    
    sns.set(style="darkgrid")
    
    sns.violinplot(data=df_null,inner="quartile",linewidth=0.5,color='gray',alpha=0.5)

    sns.violinplot(data=df_sim, inner="quartile",linewidth=0.5,palette=gof_palette)
    
    
    plt.ylabel("$\delta y $")
    plt.title(which_stat, fontsize = mytitlesize)


    if save != None:
        plt.savefig(save + '/gof_' + which_stat + '.png', dpi=600, bbox_inches = "tight")   

    return

re_sim, re_nm = miner.compute_rel_errs(gof_exp[:,:7],gof_sim[:,:,:7],gof_sim_nm[:,:,:7])

for gof in miner.gofstats:
    gof_violins(gof, re_sim, re_nm, save=plotpath)











