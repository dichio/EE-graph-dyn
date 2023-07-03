
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 20 Apr 2023

@author: vito.dichio
@version: 3.2 - MRG 

Last update on 26 Apr 2023
"""

#%% Import what is needed

import os, time
os.chdir(os.path.dirname(os.path.realpath(__file__)))
start_time = time.time()

import numpy as np 

import src.miner_MRG as miner
import src.building_blocks_MRG as sim

myseed = 160318

#%%
nameset = 'test'

N = 10; L = N*(N-1)/2

params = np.array([N,              # N    
                   50,             # T
                   ##############
                   2**11,          # M
                   1,              # nu
                   None,           # mu
                   None,           # rf
                   1,              # d_in
                   0.0666,         # d_trgt
                   160318])        # seed

#%%
mus = [0.25*i/L for i in range(1,9)]

phis = [5e3/L * 0.5*i for i in range(1,5) ]

agd = np.zeros((len(mus),len(phis)))
agd_approx = np.zeros((len(mus),len(phis)))

print('# sims = %i x %i' % (len(mus),len(phis)))

idx = 0
for k in range(len(mus)):
    for j in range(len(phis)): 
        
        
        params[4] = mus[k]; params[5] = phis[j]/mus[k]; 
    
        np.random.seed(int(params[-1]))
        
        T, N, L, M, nu, mu, phi, E_in, E_trgt, ID = sim.init(params,verbose=False)
        cl, ncl, fits = sim.init_pop(L,M,phi,E_in,E_trgt,how="alldiff")
        stats = np.zeros((T+1,2))
        
        for i in range(0,T+1):
            
            t = i/nu
        
            stats[i] = sim.save_stats(t, cl, ncl )
    
            pop = np.repeat(cl, ncl, axis=0) 
    
            pop = sim.mutations(pop, 1-np.exp(-mu), style = "flip")  
    
            cl, ncl = np.unique(pop, axis=0, return_counts=True)
    
            fits, ncl = sim.selection(cl, ncl, M, phi, E_trgt)
    
    
        idx += 1
        if idx%5==0:
            print('%i over %i'%(idx,len(mus)*len(phis)))
            
        agd[k,j] = np.mean(stats[-5:,0])/L
        agd_approx[k,j] = miner.exact_solution_agd(T, L, nu, mu, phi, E_in, E_trgt)
    
print("--- %f minutes ---" % ((time.time() - start_time)/60.))

agd = np.flip(agd,axis=0)-round(params[7])
agd_approx = np.flip(agd_approx,axis=0)-round(params[7])


#%% Viz

import matplotlib.pyplot as plt
import seaborn as sns

myfontsize = 8
mylabelsize = 7

#%%
sns.set_theme(style="white")

cm = 1/2.54  # centimeters in inches

fig, ax = plt.subplots(figsize=(4.3*cm,6.6*cm))
    
im = ax.imshow(agd,
                cmap=sns.color_palette("YlOrBr", as_cmap=True))

cax = fig.add_axes([.2,1.0,0.6,.02])

cax.tick_params(labelsize=mylabelsize)
cax.set_title(r'$d_{\infty}-E^*/L$', fontsize = myfontsize)

fig.colorbar(im, 
             cax=cax,
             orientation='horizontal',
             pad=.05)

xticks = [r'$\frac{1}{2}\bar{\varphi}$',r'$\bar{\varphi}$',r'$\frac{3}{2}\bar{\varphi}$',r'$2\bar{\varphi}$']
ax.set_xticks(np.arange(len(xticks)))
ax.set_xticklabels(xticks,fontsize=mylabelsize)

yticks = [r'$2\bar{\mu}$','',r'$\frac{3}{2}\bar{\mu}$','',r'$\bar{\mu}$','',r'$\frac{1}{2}\bar{\mu}$','']
ax.set_yticks(np.arange(len(yticks)))
ax.set_yticklabels(yticks,fontsize=mylabelsize)
#ax.set_yticks(yticks)


plt.savefig('output/agd.png', dpi=1200, bbox_inches = "tight") 

#%%
sns.set_theme(style="white")

cm = 1/2.54  # centimeters in inches

fig, ax = plt.subplots(figsize=(4.3*cm,6.6*cm))
    
im = ax.imshow(agd_approx,
                cmap=sns.color_palette("Blues_r", as_cmap=True))

cax = fig.add_axes([.2,1.0,0.6,.02])
fig.colorbar(im, 
             cax=cax,
             orientation='horizontal')

cax.set_title(r'$d_{\infty}-E^*/L$', fontsize = myfontsize)
cax.tick_params(labelsize=mylabelsize)



xticks = [r'$\frac{1}{2}\bar{\varphi}$',r'$\bar{\varphi}$',r'$\frac{3}{2}\bar{\varphi}$',r'$2\bar{\varphi}$']
ax.set_xticks(np.arange(len(xticks)))
ax.set_xticklabels(xticks,fontsize=mylabelsize)

yticks = [r'$2\bar{\mu}$','',r'$\frac{3}{2}\bar{\mu}$','',r'$\bar{\mu}$','',r'$\frac{1}{2}\bar{\mu}$','']
ax.set_yticks(np.arange(len(yticks)))
ax.set_yticklabels(yticks,fontsize=mylabelsize)
#ax.set_yticks(yticks)


plt.savefig('output/dec_approx/agd_approx.png', dpi=1200, bbox_inches = "tight")

#%%
sns.set_theme(style="white")

cm = 1/2.54  # centimeters in inches

fig, ax = plt.subplots(figsize=(4.3*cm,6.6*cm))
    
im = ax.imshow(agd-agd_approx,
                cmap=sns.color_palette("mako_r", as_cmap=True),
                )

cax = fig.add_axes([.2,1.0,0.6,.02])
fig.colorbar(im, 
             cax=cax,
             orientation='horizontal')

cax.set_title(r'$d_{\infty}-d_{\infty}^{dec}$', fontsize = myfontsize)
cax.tick_params(labelsize=mylabelsize)



xticks = [r'$\frac{1}{2}\bar{\varphi}$',r'$\bar{\varphi}$',r'$\frac{3}{2}\bar{\varphi}$',r'$2\bar{\varphi}$']
ax.set_xticks(np.arange(len(xticks)))
ax.set_xticklabels(xticks,fontsize=mylabelsize)

yticks = [r'$2\bar{\mu}$','',r'$\frac{3}{2}\bar{\mu}$','',r'$\bar{\mu}$','',r'$\frac{1}{2}\bar{\mu}$','']
ax.set_yticks(np.arange(len(yticks)))
ax.set_yticklabels(yticks,fontsize=mylabelsize)
#ax.set_yticks(yticks)


plt.savefig('output/dec_approx/agd_vs_approx.png', dpi=1200, bbox_inches = "tight")
 
#%%
from scipy import optimize

def to_spin(x):
    return 2*x-1

def to_edge(x):
    return (1+x)/2

mu = 1/L
rho = 5e2
E = round(L*params[7])
d0 = .9 ; m0=to_spin(d0)


def dmdt(m,L,mu,rho,E):
    phi = mu*rho
    return -2*mu*m - phi/L**2*(1-m**2)*((L-1)/2*m+L/2-E)

dmdtv = np.vectorize(dmdt)

f = lambda m: -2*mu*m - rho*mu/L**2*(1-m**2)*((L-1)/2*m+L/2-E)

m1, m2, m3 = optimize.fsolve(f,[-10,0,10])

#%% Vector field ode

fig, ax = plt.subplots(figsize=(5*cm,5*cm))

mylabelsize=7
myfontsize=7

xmin = -1.35
xmax = 1.15
epsx = .06
epsy = 2.5e-2

cz = sns.color_palette("dark")
cz1 = sns.color_palette("bright")

plt.axvspan(xmin, -1., facecolor='0.2', alpha=0.1)
plt.axvspan(1, xmax, facecolor='.2', alpha=0.1)

m = np.arange(-1,1,.01)
plt.plot(m,dmdtv(m,L,mu,rho,E),zorder=1,label=r'$\dot{m}$',c='k',lw=1)   

m = np.arange(xmin,-1,.01)
plt.plot(m,dmdtv(m,L,mu,rho,E),zorder=1,c='k',ls='--',lw=.5)  
m = np.arange(1,xmax,.01)
plt.plot(m,dmdtv(m,L,mu,rho,E),zorder=1,c='k',ls='--',lw=.5)  

plt.scatter(m0,0,s=50,color='w',marker="d",linewidth=1,edgecolors='k',zorder=3)
ax.annotate('$m_0$', (m0,0), xytext=(m0-2*epsx,epsy),fontsize=myfontsize)
plt.scatter(m1,0,s=50,color=cz[3],linewidth=1,edgecolors='k',zorder=3)
ax.annotate('$m_1$', (m1,0), xytext=(m1-2*epsx, epsy),fontsize=myfontsize)
plt.scatter(m2,0,s=50,c=cz[2],linewidth=1,edgecolors='k',zorder=3)
ax.annotate('$m_2$', (m2,0), xytext=(m2, epsy),fontsize=myfontsize)
plt.scatter(m3,0,s=50,color=cz[3],linewidth=1,edgecolors='k',zorder=3)
ax.annotate('$m_3$', (m1,0), xytext=(m3+epsx, epsy),fontsize=myfontsize)
#X = np.arange(-1,1,0.2)
#Y = np.repeat(0,len(X))
#plt.quiver([X,Y],[np.sign(dmdtv(X,L,mu,rho),Y)])

x_pos = np.arange(-1,1.1,0.25)
y_pos = [0 for i in range(len(x_pos))]

x_direct = dmdtv(x_pos,L,mu,rho,E)

ax.quiver(x_pos, y_pos, x_direct, y_pos, alpha=1, scale=2, width=5e-3,headwidth=7,headlength=4,color=cz[2],zorder=3)       
       
plt.legend(loc=3)
#plt.xlabel('$m$',fontsize=myfontsize)
#ax.xaxis.set_label_coords(1., .55)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))

ax.set_xticks([-1.0,1.0])

plt.xticks(fontsize=mylabelsize)
plt.yticks(fontsize=mylabelsize)
ax.set(yticklabels=[])  # remove the tick labels
ax.tick_params(left=False)  # remove the ticks

plt.savefig('output/dec_approx/dmdt.png', dpi=1200, bbox_inches = "tight") 



#%% solution
from scipy.integrate import solve_ivp

ts = []
ms = []

for mu in [1/L,.5/L, 2/L]:
    
    def f(t,x):
        return -2*mu*x-rho*mu/L**2*(1-x**2)*((L-1)/2*x+L/2-E) #flip
    
    t = (0.,100.)
    sol =  solve_ivp(f, t_span=t, y0=[m0])
    ts.append(sol.t)
    ms.append(sol.y[0])
    

mylabelsize=7
myfontsize=9

fig, ax = plt.subplots(figsize=(8.6*cm,8.6*cm))
    
epsx = 11
epsy = 2e-2

d0 = .9
m0 = to_spin(d0)

cz = sns.color_palette("tab10")


plt.plot(ts[0],to_edge(ms[0]),c=cz[0],label=r'$\bar{\mu}$')
plt.plot(ts[1],to_edge(ms[1]),c=cz[1],label=r'$0.5\bar{\mu}$')
plt.plot(ts[2],to_edge(ms[2]),c=cz[5],label=r'$2\bar{\mu}$')


cz = sns.color_palette("dark")
plt.scatter(0,to_edge(m0),s=150,color='w',marker="d",linewidth=1,edgecolors='k',zorder=3)
ax.annotate('$d_0$', (0,to_edge(m0)), xytext=(-epsx, to_edge(m0)),fontsize=myfontsize)
plt.scatter(0,to_edge(m2),s=150,c=cz[2],linewidth=1,edgecolors='k',zorder=3)
ax.annotate('$d_2$', (0,to_edge(m2)), xytext=(-epsx, to_edge(m2)),fontsize=myfontsize)
plt.scatter(0,E/L,s=150,c=cz[4],linewidth=1,edgecolors='k',zorder=3)
ax.annotate(r'$\bar{E}/L$', (0,E/L), xytext=(-1.2*epsx, E/L-epsy),fontsize=myfontsize)

plt.legend(loc=4,ncol=3,fontsize=mylabelsize)
plt.ylim([0,1])

ax.set_yticks([.5,1.0])

plt.xlabel('t',fontsize=myfontsize)
ax.xaxis.set_label_coords(1., -0.1)
plt.ylabel('$d_t$',fontsize=myfontsize,rotation=0)
ax.yaxis.set_label_coords(0.05, 1.05)  

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', 0))
#ax.spines['bottom'].set_position(('data', 0))

plt.xticks(fontsize=mylabelsize)
plt.yticks(fontsize=mylabelsize)

plt.savefig('output/dec_approx/mt.png', dpi=1200, bbox_inches = "tight")






























