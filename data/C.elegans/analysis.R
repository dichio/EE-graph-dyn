# Created 29/11/2022 -- Last modified 16/01/2023
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

source("src_code/data_analysis.R") 

library(ergm)

# Import networks
nets <- import_nets()

ID = 1
dists <- as.matrix(read.csv(file = paste('processed_data/',ID,'_ec_dist.csv',sep=""),row.names = 1))

#compute_stats(nets)

# ERGM

m0 <- 'gwdegree(fixed=F) + gwesp(fixed=F) + edgecov(dists),
          constraints = ~ edges,
          verbose=T, 
          eval.loglik = TRUE,
          control=snctrl( MCMLE.maxit = 60,
                          MCMC.interval = 1e5,
                          init = c(1.28,1.96,.81,1.35,-.1),
                          bridge.nsteps = 32)'

m1 <- 'gwdegree(fixed=F) + gwesp(fixed=F),
          constraints = ~ edges,
          verbose=T, 
          eval.loglik = TRUE,
          control=snctrl( MCMLE.maxit = 60,
                          init = c(1.,1.,1.,1.),
                          bridge.nsteps = 16)'

set.seed(160318)

bfit <- eval(parse(text = paste('ergm(nets[[',ID,']] ~', m1, ')',sep="")))

summary(bfit)



bfit.gof <- gof(bfit,control=control.gof.formula(nsim = 1000)) #output simulate
plot(bfit.gof)
#mcmc.diagnostics(bfit)
