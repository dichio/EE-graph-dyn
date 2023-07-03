# Created 28/11/2022 -- Last modified 06/12/2022
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

source("src_code/data_mining.R") 

library(igraph)

ID = 1

for (ID in 1:8){
  
#### edgelist ####
# import
el <- read.csv(paste("src_data/witvliet2021_nemanode.org/witvliet_2020_", ID, ".csv", sep=""), sep="\t", header=TRUE)[1:3]
# filter
el <- filter_edgelist(el,which_etype="chemical",which_ntype=c('sensory','inter','motor','modulatory')) 

#### node attrs ####
nodes <- node_attrs(edgelist_to_nodelist(el[1:2]),ID,compute=T,scale=T)

#### edge attrs ####
el <- cbind(el,sapply(el[1:2], function(x) nodes[x,][[1]])); names(el)[3:4] <- c('pre_type','post_type')
el <- cbind(el,apply(el[3:4],1,conn_type)); names(el)[5] <- 'conn_type'
el <- cbind(el,apply(el,1,edge_attr_dist)); names(el)[6] <- c('dist')

#### edge covariates ####
ec <- ecov_dist(nodes,compute=T)

#### make graph and plot ####
g <- graph_from_data_frame(el, directed = TRUE)
vertex_attr(g) <- cbind(name=rownames(nodes), nodes)


cm = 1/2.54

# Open the PDF graphics device
pdf(paste0('ID',ID,'.pdf'), width = 8*cm, height = 8*cm, pointsize = 12)

my_nplot(g)

dev.off()

write.csv(get.adjacency(g,attr='dist',sparse=FALSE), file = paste('processed_data/',ID,'_adj.csv',sep=""),row.names=TRUE)

}








# ADJs for Python

####  Take ID = 8 as mold ####
ID = 8

el8 <- read.csv(paste("src_data/witvliet2021_nemanode.org/witvliet_2020_", ID, ".csv", sep=""), sep="\t", header=TRUE)[1:3]
el8 <- filter_edgelist(el8,which_etype="chemical",which_ntype=c('sensory','inter','motor','modulatory')) 

nnames <- edgelist_to_nodelist(el8); N <- length(nnames)

adj8 <- create_adj(el8) 



for (ID in 1:8){
  el <- read.csv(paste("src_data/witvliet2021_nemanode.org/witvliet_2020_", ID, ".csv", sep=""), sep="\t", header=TRUE)[1:3]
  el <- filter_edgelist(el,which_etype="chemical",which_ntype=c('sensory','inter','motor','modulatory')) 
  adj <- create_adj(el) 
  write.table(adj,file = paste('output/for_py/',ID,'_adj.txt',sep=""),append = FALSE,sep = " ",dec = ".", row.names = FALSE, col.names = FALSE)
}

ec_dist8 <- read.csv(paste("processed_data/", ID, "_ec_dist.csv", sep=""),row.names = 1)
write.table(ec_dist8,file = paste('output/for_py/',ID,'_ec_dist.txt',sep=""),append = FALSE,sep = " ",dec = ".", row.names = FALSE, col.names = FALSE)

#ec_ctype8 <- read.csv(paste("processed_data/", ID, "_ec_conn_type.csv", sep=""),row.names = 1)
#write.table(ec_ctype8,file = paste('output/data_UD-Celegans/',ID,'_ec_ctype.txt',sep=""),append = FALSE,sep = " ",dec = ".", row.names = FALSE, col.names = FALSE)



##### check undirectedness

ID = 8

el <- read.csv(paste("src_data/witvliet2021_nemanode.org/witvliet_2020_", ID, ".csv", sep=""), sep="\t", header=TRUE)[1:3]
el <- filter_edgelist(el,which_etype="chemical",which_ntype=c('sensory','inter','motor','modulatory')) 
adj_d <- create_adj_full(el,directed = TRUE)
undirectedness(adj_d)







