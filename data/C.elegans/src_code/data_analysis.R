import_nets <- function(){
  y <- vector('list',10); names(y) <- c('0h','5h','8h','16h','23h','27h','45h-1','45h-2','45h-3','45h-4')
  for (ID in 1:10){
    #### import ####
    nodes <- read.csv(paste('processed_data/',ID,'_node_attrs.csv',sep=""),row.names = 1)
    adj <- as.matrix(read.csv(file = paste('processed_data/',ID,'_adj.csv',sep=""),row.names = 1))
    #### network ####
    y[[ID]] <- network(adj, directed = FALSE, loops = FALSE, multiple = FALSE, vertex.names = rownames(nodes), vertex.attr = nodes, vertex.attrnames = names(nodes))
  }
  y
}

compute_stats <- function(x){
  my_stats = 'edges + twopath + triangle + gwdegree(decay=1.96842,fixed=TRUE) + gwesp(decay=1.53711,fixed=TRUE) + nodematch("ntype") + edgecov(dists)'; nstats = 7
  stats <- array(NaN,c(10,nstats)); rownames(stats) <- names(x); colnames(stats) <- names(eval(parse(text = paste("summary(x[[1]] ~", my_stats, ")"))))
  # compute stats
  for (ID in 1:10){
    dists <- as.matrix(read.csv(file = paste('processed_data/',ID,'_ec_dist.csv',sep=""),row.names = 1))
    net <- x[[ID]]
    stats[ID,] <- eval(parse(text = paste("summary(net ~", my_stats, ")")))
  }
  # export 
  write.csv(stats, file = 'output/statistics.csv',row.names=TRUE)
  
  # plot
  pdf(file ="./output/stats-no7.pdf",width=10,height=8)
  t = c(0,5,8,16,23,27,45,45,45,45)
  par(mfrow=c(4,2))
  for (which_stat in 1:7){
    plot(t[-7],stats[-7,which_stat], type="o", col=c("#EB811B"), lwd=2, pch=15, main = colnames(stats)[which_stat], xlab="", ylab="")
  }
  dev.off()
}