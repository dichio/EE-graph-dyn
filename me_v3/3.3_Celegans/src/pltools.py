#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 14:10:16 2023

@author: vito.dichio
"""

import os

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import pandas as pd
from sklearn.metrics import r2_score

import src.data_miner as miner

import numpy as np
myseed = 160318; np.random.seed(myseed)
cm = 1/2.54  # centimeters in inches

#%%

# names
statname = ["edges","gwdegree","gwesp","dist"]

# plot parameters
mytitlesize = 12
myfontsize = 10
mylabelsize = 8



def fit_vs_t(plotpath,nu,texp,fexp,ts,fs,fs_er,save=False):

    sns.set_theme()
    fig, ax = plt.subplots()
    
    # data
    ax.plot(ts, fs, '-', color = '#234E70', label = 'sim')
    ax.fill_between(ts, fs-fs_er , fs+fs_er, alpha=0.2)
    ax.plot(texp, fexp, 'o', color='tab:brown', label = 'exp')
    ax.plot(texp, fs[texp*nu], 's', color = "#234E70")
    
    # aesthetics
    plt.title('fitness', fontsize = mytitlesize)
    plt.legend(loc=2, fontsize = myfontsize)
    ax.set_xlabel('time [hours]', fontsize = myfontsize)
    plt.xticks(fontsize=mylabelsize) ; plt.yticks(fontsize=mylabelsize)
    
    # save
    if save != False:
        plt.savefig(plotpath+'/fit_allt---'+save+'.png', dpi=600) 
    
    return


#%%
def stat_vs_t(plotpath,texp,sexp,t,s,errs,title="notitle",save=False):
    
    
    sns.set_theme()
    fig, ax = plt.subplots()
    
    # data
    mylcolor = "#150050"
    
    ax.plot(t, s, '-', label = 'sim', color=mylcolor)
    ax.fill_between(t, s - errs, s + errs, alpha=0.2, color=mylcolor)
    ax.plot(texp, sexp, 'o', color='tab:brown', label = 'exp')
    
    # aesthetics
    plt.title(title, fontsize = mytitlesize)
    plt.legend(loc=2, fontsize = myfontsize)
    ax.set_xlabel('time [hours]', fontsize = myfontsize)
    plt.xticks(fontsize=mylabelsize) ; plt.yticks(fontsize=mylabelsize)
    # save
    if save != False:
        plt.savefig(plotpath+'/'+title+'_allt---'+save+'.png', dpi=600) 
    
    return  
    
#%%

def stats_1vs1(plotpath,texp,s1exp,s2exp,t,s1,s2,nu,title="notitle",save=False):
    
    sns.set_style("darkgrid")
    fig, ax = plt.subplots()
    
    # data
    mycm = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
        
    sns.scatterplot(x=s1, y=s2, hue=t, palette=mycm, s=10, marker = 'd', legend = False)
    sns.scatterplot(x=s1[texp*int(nu)], y=s2[texp*int(nu)], hue=texp, palette=mycm, s=100, marker = 'd', legend = False)
    sns.scatterplot(x=s1exp, y=s2exp, hue=texp, palette=mycm, s=150, marker = 'p', legend = False)

    # aesthetics
    plt.title(title, fontsize = mytitlesize)
    plt.xticks(fontsize=mylabelsize) ; plt.yticks(fontsize=mylabelsize)
    
    # Time-bar
    norm = plt.Normalize(vmin=0, vmax=45)
    sm = plt.cm.ScalarMappable(cmap=mycm, norm=norm)
    sm.set_array([])
    
    cbar = ax.figure.colorbar(sm,shrink=.75,ticks=texp)
    cbar.set_label('[h]', fontsize = myfontsize, labelpad=10,rotation=0)
    cbar.ax.tick_params(labelsize=mylabelsize)
    
    # save
    if save != False:
        plt.savefig(plotpath+'/'+title+'---'+save+'.png', dpi=600)
                    
    return
       
def orth_proj(plotpath,t,s,texp,sexp,nu,title="",save=False):
    
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8,6), sharex='col', sharey = 'row')
    plt.subplots_adjust(wspace=0.25, hspace=0.3)
    ax[1,1].axis('off')
    
    mycm = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
    
    sns.scatterplot(x=s[:,0], y=s[:,1], hue=t, palette=mycm, s=10, marker = 'd', legend = False, ax=ax[1,0])
    sns.scatterplot(x=s[texp*int(nu),0], y=s[texp*int(nu),1], hue=texp, palette=mycm, s=100, marker = 'd', legend = False,ax=ax[1,0])
    sns.scatterplot(x=sexp[:,0], y=sexp[:,1], hue=texp, palette=mycm, s=150, marker = 'p', legend = False, ax=ax[1,0])
    
    sns.scatterplot(x=s[:,0], y=s[:,2], hue=t, palette=mycm, s=10, marker = 'd', legend = False, ax=ax[0,0])
    sns.scatterplot(x=s[texp*int(nu),0], y=s[texp*int(nu),2], hue=texp, palette=mycm, s=100, marker = 'd', legend = False,ax=ax[0,0])
    sns.scatterplot(x=sexp[:,0], y=sexp[:,2], hue=texp, palette=mycm, s=150, marker = 'p', legend = False, ax=ax[0,0])

    sns.scatterplot(x=s[:,1], y=s[:,2], hue=t, palette=mycm, s=10, marker = 'd', legend = False, ax=ax[0,1])
    sns.scatterplot(x=s[texp*int(nu),1], y=s[texp*int(nu),2], hue=texp, palette=mycm, s=100, marker = 'd', legend = False,ax=ax[0,1])
    sns.scatterplot(x=sexp[:,1], y=sexp[:,2], hue=texp, palette=mycm, s=150, marker = 'p', legend = False, ax=ax[0,1])
    ax[0,1].invert_xaxis()
        
    # aesthetics
    fig.suptitle(title, fontsize=mytitlesize)
    
    ax[1,0].set_xlabel("gwdegree", fontsize = myfontsize)
    ax[1,0].set_ylabel("gwesp", fontsize = myfontsize)
    ax[0,0].set_ylabel("dist", fontsize = myfontsize)
    
    plt.tick_params(axis='both', which='major', labelsize=mylabelsize)

    # Time-bar
    norm = plt.Normalize(vmin=0, vmax=45)
    sm = plt.cm.ScalarMappable(cmap=mycm, norm=norm)
    sm.set_array([])
    
    axbar = fig.add_axes([.6,.4,.25,.02])
    cbar = ax[1,1].figure.colorbar(sm,shrink=.75,ticks=texp,cax=axbar,orientation='horizontal')
    cbar.set_label('[h]', fontsize = myfontsize)
    cbar.ax.tick_params(labelsize=mylabelsize)

    # save
    if save != False:
        plt.savefig(plotpath+'/'+title+'---'+save+'.png', dpi=600, bbox_inches = "tight")
    
    return

def edgesvsall(plotpath,t,s,texp,sexp,nu,title="",save=False):
    
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8,6), sharex='col')
    plt.subplots_adjust(wspace=0.25, hspace=0.3)
    ax[1,1].axis('off')
    
    mycm = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
    
    sns.scatterplot(x=s[:,0], y=s[:,1], hue=t, palette=mycm, s=10, marker = 'd', legend = False, ax=ax[1,0])
    sns.scatterplot(x=s[texp*int(nu),0], y=s[texp*int(nu),1], hue=texp, palette=mycm, s=100, marker = 'd', legend = False,ax=ax[1,0])
    sns.scatterplot(x=sexp[:,0], y=sexp[:,1], hue=texp, palette=mycm, s=150, marker = 'p', legend = False, ax=ax[1,0])
    
    sns.scatterplot(x=s[:,0], y=s[:,2], hue=t, palette=mycm, s=10, marker = 'd', legend = False, ax=ax[0,0])
    sns.scatterplot(x=s[texp*int(nu),0], y=s[texp*int(nu),2], hue=texp, palette=mycm, s=100, marker = 'd', legend = False,ax=ax[0,0])
    sns.scatterplot(x=sexp[:,0], y=sexp[:,2], hue=texp, palette=mycm, s=150, marker = 'p', legend = False, ax=ax[0,0])

    sns.scatterplot(x=s[:,0], y=s[:,3], hue=t, palette=mycm, s=10, marker = 'd', legend = False, ax=ax[0,1])
    sns.scatterplot(x=s[texp*int(nu),0], y=s[texp*int(nu),3], hue=texp, palette=mycm, s=100, marker = 'd', legend = False,ax=ax[0,1])
    sns.scatterplot(x=sexp[:,0], y=sexp[:,3], hue=texp, palette=mycm, s=150, marker = 'p', legend = False, ax=ax[0,1])      
    # aesthetics
    fig.suptitle(title, fontsize=mytitlesize)
    
    ax[1,0].set_ylabel("gwdegree", fontsize = myfontsize)
    ax[0,0].set_ylabel("gwesp", fontsize = myfontsize)
    ax[0,1].set_ylabel("dist", fontsize = myfontsize)
    
    plt.tick_params(axis='both', which='major', labelsize=mylabelsize)

    # Time-bar
    norm = plt.Normalize(vmin=0, vmax=45)
    sm = plt.cm.ScalarMappable(cmap=mycm, norm=norm)
    sm.set_array([])
    
    axbar = fig.add_axes([.6,.25,.25,.02])
    cbar = ax[1,1].figure.colorbar(sm,shrink=.75,ticks=texp,cax=axbar,orientation='horizontal')
    cbar.set_label('[h]', fontsize = myfontsize)
    cbar.ax.tick_params(labelsize=mylabelsize)

    # save
    if save != False:
        plt.savefig(plotpath+'/'+title+'---'+save+'.png', dpi=600, bbox_inches = "tight")
    
    return



def statspace3d(plotpath,t,s,texp,sexp,nu,title="",save=False):
    
    sns.set_style("whitegrid")
    
    fig = plt.figure(figsize=(6,6))
    ax = Axes3D(fig, auto_add_to_figure=False,azim=30)
    fig.add_axes(ax)
    
    mycm = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
    
    ax.scatter(xs=s[:,0],ys=s[:,1],zs=np.min(s[:,2]),marker='d',s=5, cmap=mycm,c=t)
    ax.scatter(xs=s[texp*int(nu),0], ys=s[texp*int(nu),1],zs=np.min(s[:,2]), cmap=mycm,c=texp, s=50, marker = 'd',alpha=1)
    ax.scatter(xs=sexp[:,0], ys=sexp[:,1],zs=np.min(s[:,2]), cmap=mycm, c=texp, s=100, marker = 'p',alpha=1)

    
    ax.scatter(s[:,0],ys=np.min(s[:,1]),zs=s[:,2],marker='d',s=5, alpha=1,cmap=mycm,c=t)
    ax.scatter(xs=s[texp*int(nu),0], ys=np.min(s[:,1]),zs=s[texp*int(nu),2], cmap=mycm,c=texp, s=50, marker = 'd',alpha=1)
    ax.scatter(xs=sexp[:,0], ys=np.min(s[:,1]),zs=sexp[:,2], cmap=mycm, c=texp, s=100, marker = 'p',alpha=1)


    ax.scatter(xs=np.min(s[:,0]),ys=s[:,1],zs=s[:,2],marker='d',s=5, alpha=1,cmap=mycm,c=t)
    ax.scatter(xs=np.min(s[:,0]),ys=s[texp*int(nu),1],zs=s[texp*int(nu),2], cmap=mycm, c=texp, s=50, marker = 'd', alpha=1)
    ax.scatter(xs=np.min(s[:,0]), ys=sexp[:,1],zs=sexp[:,2], cmap=mycm, c=texp, s=100, marker = 'p',alpha=1)

    # aesthetics
    fig.suptitle(title, fontsize=mytitlesize)
    
    ax.set_xlabel('gwdegree',fontsize=myfontsize)
    ax.set_ylabel('gwesp',fontsize=myfontsize)
    ax.set_zlabel('dist',fontsize=myfontsize)

    plt.tick_params(axis='both', which='major', labelsize=mylabelsize)
    
    # Time-bar
    norm = plt.Normalize(vmin=0, vmax=45)
    sm = plt.cm.ScalarMappable(cmap=mycm, norm=norm)
    sm.set_array([])
    
    axbar = fig.add_axes([1,.2,.02,.5])
    cbar = ax.figure.colorbar(sm,shrink=.75,ticks=texp,cax=axbar)
    cbar.set_label('[h]', fontsize = myfontsize,labelpad=10,rotation=0)
    cbar.ax.tick_params(labelsize=mylabelsize)
    
    # save
    if save != False:
        plt.savefig(plotpath+'/'+title+'---'+save+'.png', dpi=600, bbox_inches = "tight")   
    
    return

#%%    
def snaphist(plotpath,counts,s,sexp,title="notitle",save=False):
    
    sns.set_style("darkgrid")
    s = np.repeat(s,counts)
    
    fig, ax = plt.subplots()
    
    plt.axvline(sexp, 0, 1, color='#234E70', label = 'exp')
    sns.histplot(s,kde=True,stat = 'density', color='#2C5F2D', label = 'sim')
    
    
    # aesthetics
    plt.title(title, fontsize = mytitlesize)
    ax.set_ylabel('density',fontsize = myfontsize)
    ax.legend(fontsize=myfontsize)
    plt.tick_params(axis='both', which='major', labelsize=mylabelsize)
    
    # save
    if save != False:
        plt.savefig(plotpath+'/'+title+'---'+save+'.png', dpi=600, bbox_inches = "tight")   
        
    return


#%%
def snaphist_cf2(simset1,simset2,ID1,ID2,age,statidx,sexp,title="notitle",save=False):
    
    ns1,gs1 = miner.import_insta(miner.shout_paths(ID1,simset1)[1],age)#[1:3]
    ns2,gs2 = miner.import_insta(miner.shout_paths(ID2,simset2)[1],age)#[1:3]
    
    s1 = np.repeat(gs1[:,statidx],ns1)
    s2 = np.repeat(gs2[:,statidx],ns2)
    
    fig, ax = plt.subplots()
    
    plt.axvline(sexp, 0, 1, color='#234E70', label = 'exp')
    sns.histplot(s1,kde=True,stat = 'density', color = '#AC4425', alpha = .4, label = ID1)
    sns.histplot(s2,kde=True,stat = 'density', color = '#224B0C', alpha = .4, label = ID2)


    # aesthetics
    plt.title(title, fontsize = mytitlesize)
    ax.set_ylabel('density',fontsize=myfontsize)
    ax.legend(fontsize=myfontsize,loc='upper center', bbox_to_anchor=(0.5, -0.1),)
    plt.tick_params(axis='both', which='major', labelsize=mylabelsize)
    
    
    plotpath = miner.shout_paths(ID1,simset1)[0] + 'figures-multi'
    if not(os.path.exists(plotpath)):
        os.mkdir(plotpath)
    # save
    if save != False:
        plt.savefig(plotpath + '/' + title + '_' + ID1 + '_' + ID2 + '.png', dpi=600, bbox_inches = "tight")   
        
    return

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
    
def gof_distrib(which_stat,which_time,exp,sim,null,save=None):
    
    idx_stat = miner.gofstats.index(which_stat)
    
    idx_time = miner.agecode.index(which_time)-1
    
    if idx_time == 6:
        idx_time = 5

    simnull = np.vstack((sim[idx_time,:,idx_stat],null[idx_time,:,idx_stat])).T
    df = pd.DataFrame(simnull,columns=['sim','null'])   
    
    plt.figure(figsize=(6,6))
    sns.histplot(data=df['null'],stat="probability",bins=5,color="gray",alpha=0.5)
    sns.histplot(data=df['sim'],stat="probability",bins=5,color=gof_palette[idx_time])
    
    if idx_time !=5:
        plt.axvline(exp[idx_time,idx_stat],0,0.2, linewidth=3,color="#850000")
        plt.legend(labels=[ 'exp','sim', 'null'],loc=1)

    else:
        plt.axvline(exp[5,idx_stat],0,0.2, linewidth=3,color="#850000")
        plt.axvline(exp[6,idx_stat],0,0.2, linewidth=3,color="#850000")
        plt.legend(labels=['exp',' exp', 'sim', 'null'],loc=1)

    plt.title(which_stat+'\n t = ' + str(miner.ageindex(which_time)[0]) + 'h',fontsize=mytitlesize )
    
    if save != None:
        plt.savefig(save + '/gof_' + which_stat + '_t_' + str(miner.ageindex(which_time)[0]) +  '.png', dpi=600, bbox_inches = "tight")   
    
    return



def gof_degrees(which_time,maxdeg,exp,sim,null,save=None):
    
    idx_time = miner.agecode.index(which_time)-1
    
    fig, ax = plt.subplots(figsize=(6,6))

    df = pd.DataFrame(np.column_stack((np.arange(0,maxdeg,dtype=int),
                                       exp[idx_time,:maxdeg],
                                       np.average(sim[idx_time,:,:maxdeg],axis=0),
                                       np.average(null[idx_time,:,:maxdeg],axis=0),
                                       np.std(sim[idx_time,:,:maxdeg],axis=0),
                                       np.std(null[idx_time,:,:maxdeg],axis=0))),
                                      columns=['deg','exp','sim_av','null_av','sim_std','null_std'])
    
    df["deg"] = pd.to_numeric(df["deg"], downcast='integer')
    
    ax.errorbar(x=df['deg'], y=df['null_av'], yerr=df["null_std"], fmt="o",color='gray' ,alpha=1,label='null')
    ax.errorbar(x=df['deg'], y=df['sim_av'], yerr=df["sim_std"], fmt="o", color=gof_palette[idx_time],alpha=1,label='sim')

    sns.lineplot(data=df,x='deg',y='exp',color="#850000",alpha=1,drawstyle='steps-mid',zorder=10,label='exp')
    plt.stackplot(df["deg"], df["exp"], color="#850000", step='mid', alpha=0.2)
    
    
    plt.xlim([-0.5,maxdeg])
    plt.ylim(bottom=0)
    #plt.xticks(rotation=0)
    plt.locator_params(axis='x', nbins=10)
    
    plt.title('degree distribution \n t = ' + str(miner.ageindex(which_time)[0]) + 'h',fontsize=mytitlesize )

    
    if save != None:
        plt.savefig(save + '/gof_degdis' + '_t_' + str(miner.ageindex(which_time)[0]) +  '.png', dpi=600, bbox_inches = "tight")   
    
    return

    















#%% DA_multi

def axis_vs_err(setname,errs,labels,xticks,which,save=True):
    
    sns.set_style("darkgrid")
    fig, ax = plt.subplots()

    ax.plot(xticks, errs, '-', label = labels[0])
    ax.scatter(xticks[np.argmin(errs)],np.min(errs))
    
    # aesthetics
    #plt.title("", fontsize = mytitlesize)
    plt.legend(loc=1, fontsize = myfontsize)
    
    ax.set_ylabel(r'$\delta$',fontsize = myfontsize)
    ax.set_xlabel(which, fontsize = myfontsize)
    plt.xticks(fontsize=mylabelsize) ; plt.yticks(fontsize=mylabelsize)
    
    plotpath = './output/'+ setname + '/set_figs'
    if not(os.path.exists(plotpath)):
        os.mkdir(plotpath)
    # save
    if save == True:
        plt.savefig(plotpath+'/'+which+'_vs_err.png', dpi=600, bbox_inches = "tight") 
    return

def axis_vs_err_multi(setname, errs, labels, xticks, which, norm=False,save=True):
    
    n = errs.shape[0]
    
    sns.set_style("darkgrid")
    fig, ax = plt.subplots()
    
    if norm:
        for i in range(n):
            ax.plot(xticks, errs[i]/np.max(errs[i]), '-', label = labels[i])
            ax.scatter(xticks[np.argmin(errs[i])],np.min(errs[i])/np.max(errs[i]))
    else:
        for i in range(n):
            ax.plot(xticks, errs[i], '-', label = labels[i])
            ax.scatter(xticks[np.argmin(errs[i])],np.min(errs[i]))
    
    # aesthetics
    #plt.title("", fontsize = mytitlesize)
    plt.legend(loc=3, fontsize = myfontsize)
    ax.set_xlabel(which, fontsize = myfontsize)
    ax.set_ylabel(r'$\delta$',fontsize = myfontsize)
    plt.xticks(fontsize=mylabelsize) ; plt.yticks(fontsize=mylabelsize)
    
    plotpath = './output/'+ setname + '/set_figs'
    if not(os.path.exists(plotpath)):
        os.mkdir(plotpath)
    # save
    if save == True:
        plt.savefig(plotpath+'/'+which+'_vs_err_multi.png', dpi=600, bbox_inches = "tight") 
    return


def value_to_color(val,cl_min,cl_max,palette,n_colors):  
    val_position = float((val - cl_min)) / (cl_max - cl_min) 
    ind = int(val_position * (n_colors - 1))
    return palette[ind]


def rf_rm_err(setname, errs, labels, save=True):
    
    fig, ax = plt.subplots()

    sns.heatmap(errs,
                cbar = True,
                xticklabels = ["%.1f" %x for x in labels[1]],
                yticklabels = ["%.2f" %x for x in labels[0]]
        )
    
    
    plotpath = './output/'+ setname + '/set_figs'
    if not(os.path.exists(plotpath)):
        os.mkdir(plotpath)
    # save
    if save == True:
        plt.savefig(plotpath+'/err_vs_nu-r.png', dpi=600, bbox_inches = "tight") 
    return

from scipy.optimize import curve_fit

def mah_fin(nameset,x,y,err,y_allt,err_allt,save):
     
    # plot parameters
    mytitlesize = 12
    myfontsize = 7
    mylabelsize = 7
    
    def parabola(x,a,b,c):
        return c+b*x+a*x**2
    
    fig, ax = plt.subplots(figsize=(4.3*cm,4.*cm))
    
    sns.set_style("darkgrid")
    
    c0 = sns.color_palette("tab10").as_hex()[1]
    c1 = sns.color_palette("dark").as_hex()[1]
    
    popt_fin, pcov_fin = curve_fit(parabola, x, y, sigma=err)
    
    # Compute R-squared
    y_pred = parabola(x, *popt_fin)
    print("R2_fin = %.3f" %r2_score(y, y_pred))
    
    print(popt_fin)
    
    ax.errorbar(x,y,yerr=err,lw=.5,alpha=.75,ecolor=c0,errorevery=1,color=c0,zorder=2)
    ax.plot(x,y,lw=1.,color=c0,alpha=.5,zorder=0)#,label=r'$\delta^{mah}_T$')
    ax.plot(x,parabola(x,popt_fin[0],popt_fin[1],popt_fin[2]),lw=.75,color='k',zorder=1,ls = '--')
    
    
    xmin_fin = -popt_fin[1]/(2*popt_fin[0])
    ax.scatter(xmin_fin,parabola(xmin_fin,popt_fin[0],popt_fin[1],popt_fin[2]),s=30,marker='v',linewidths=.6, edgecolors='k',color=c1,zorder=4,label=r'$\tilde{\delta}^{mah}_{T}(\rho^*)$')
    print(-popt_fin[1]/(2*popt_fin[0]))
    
    c2 = sns.color_palette("dark").as_hex()[0]
    
    popt_allt, pcov_allt = curve_fit(parabola, x, y_allt, sigma=err_allt)
    
    # Compute R-squared
    y_pred_1 = parabola(x, *popt_allt)
    print("R2_allt = %.3f" %r2_score(y_allt, y_pred_1))
    
    xmin_allt = -popt_allt[1]/(2*popt_allt[0])
    ax.scatter(xmin_allt,parabola(xmin_allt,popt_fin[0],popt_fin[1],popt_fin[2]),s=30,marker='v', linewidths=.6, edgecolors='k',color=c2,zorder=4,label=r'$\tilde{\delta}^{mah}_{T}(\rho^{**})$')


    
    # Layout 
    
    plt.xlabel(r'$\rho$',fontsize=myfontsize)
    ax.xaxis.set_label_coords(.95, -0.05)

    #plt.ylabel(r'$\delta^{mah}_T$',fontsize=myfontsize)
    
    plt.xticks(fontsize=mylabelsize)
    plt.yticks(fontsize=mylabelsize)
    
    ax.tick_params(axis='both', which='major', pad=0)
    
    plt.legend(loc=1,ncol=1,fontsize=mylabelsize)
    
    # Save
    plotpath = './output/'+ nameset + '/set_figs'
    if save == True:
        plt.savefig(plotpath+'/err_mah_T.png', dpi=600, bbox_inches = "tight")
    return

def mah_allt(nameset,x,y,err,y_allt,err_allt,save):
     
    # plot parameters
    myfontsize = 7
    mylabelsize = 7
    
    def parabola(x,a,b,c):
        return c+b*x+a*x**2
    
    fig, ax = plt.subplots(figsize=(8.6*cm,8.6*cm))
    
    sns.set_style("darkgrid")
    
    c0 = sns.color_palette("tab10").as_hex()[0]
    c1 = sns.color_palette("dark").as_hex()[0]
    
    popt_allt, pcov_allt = curve_fit(parabola, x, y_allt, sigma=err_allt)

    ax.errorbar(x,y_allt,yerr=err_allt,lw=.75,alpha=.75,ecolor=c0,errorevery=1,color=c0,zorder=2,label=r'$\delta^{mah}_{\mathbf{t}^*}\ [av,sd]$')
    ax.plot(x,y_allt,lw=1.,color=c0,alpha=.5,zorder=0)#,label=r'$\delta^{mah}_T$')
    ax.plot(x,parabola(x,popt_allt[0],popt_allt[1],popt_allt[2]),lw=.75,color='k',zorder=1,ls = '--',label=r'$\tilde{\delta}^{mah}_{\mathbf{t}^*}(\rho)$')

    xmin_allt = -popt_allt[1]/(2*popt_allt[0])
    ax.scatter(xmin_allt,parabola(xmin_allt,popt_allt[0],popt_allt[1],popt_allt[2]),s=50,marker='v', linewidths=.5, edgecolors='k',color=c1,zorder=4,label=r'$\tilde{\delta}^{mah}_{\mathbf{t}^*}(\rho^{**})$')
    print(-popt_allt[1]/(2*popt_allt[0]))
    
    c2 = sns.color_palette("dark").as_hex()[1]
    
    popt_fin, pcov_fin = curve_fit(parabola, x, y, sigma=err)
    xmin_fin = -popt_fin[1]/(2*popt_fin[0])
    ax.scatter(xmin_fin,parabola(xmin_fin,popt_allt[0],popt_allt[1],popt_allt[2]),s=50, marker='v',linewidths=.5, edgecolors='k',color=c2,zorder=4,label=r'$\tilde{\delta}^{mah}_{\mathbf{t}^*}(\rho^{*})$')
    

    
    # Layout 
    
    plt.xlabel(r'$\rho$',fontsize=myfontsize)
    ax.xaxis.set_label_coords(.95, -0.05)

    #plt.ylabel(r'$\delta^{mah}_T$',fontsize=myfontsize)
    
    plt.xticks(fontsize=mylabelsize)
    plt.yticks(fontsize=mylabelsize)
    
    ax.tick_params(axis='both', which='major', pad=0)
    
    
    handles, labels = plt.gca().get_legend_handles_labels()
    new_order = [3, 0, 2, 1]  # New order of the legend items
    handles = [handles[i] for i in new_order]
    labels = [labels[i] for i in new_order]
    
    plt.legend(handles, labels, loc="upper center",ncol=2,fontsize=mylabelsize)
    
    # Save
    plotpath = './output/'+ nameset + '/set_figs'
    if save == True:
        plt.savefig(plotpath+'/err_mah_allt.pdf', dpi=600, bbox_inches = "tight")
    return









