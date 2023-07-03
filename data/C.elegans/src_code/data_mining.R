# Created 28/11/2022 -- Last modified 29/11/2022

library("reticulate")
use_condaenv("myconda3")
source_python("./src_data/witvliet2021_github/src/data/neuron_info.py")

#### Internal variables ####

ncoor <- read.csv("src_data/skuhersky2022/LowResAtlasWithHighResHeadsAndTails.csv", sep=",", row.names = 1, header=FALSE, col.names = c("n","x", "y",'z')) 

#### functions #####

filter_edgelist <- function(x, which_etype = "chemical", which_ntype = c('sensory','inter','motor','modulatory')){
  # filter by edge-type
  y <- x[x$type==which_etype,][1:2]
  # filter by node type
  which_store <- rep(T,dim(y)[1])
  for (i in 1:dim(y)[1]){
    if (!(all(sapply(y[i,1:2],ntype) %in% which_ntype))){
      which_store[i] <- FALSE 
    }
  }
  y <- y[which_store,]
  y
}

edgelist_to_nodelist <- function(x){
  # Input: 2-column matrix
  y <- unique(c(as.matrix(x)))
  y
}

L_i = 250.
L_f = 1150.
times = c(0,5,8,16,23,27,45,45,45,45)

node_attrs <- function(x,ID,compute=T,scale=T){
  if (compute == T){
    if (scale == T){
      scale = 1/1e3*1/max(ncoor$x)*(L_i+(L_f-L_i)*times[ID]/45.) # linear scaling [1e-6m]
    } else {
      scale = 1/1e3*1150./max(ncoor$x) # [1e-6m]
    }
    y <- ncoor[rownames(ncoor) %in% x,]*scale
    y <- cbind(sapply(rownames(y), ntype),y); names(y)[1] <- "ntype"
    y <- y[match(x, rownames(y)),]
    write.csv(y, file = paste('processed_data/',ID,'_node_attrs.csv',sep=""),row.names = TRUE)
    y
  } else {
    y <- read.csv(paste('processed_data/',ID,'_node_attrs.csv',sep=""),row.names = 1)
  }
}

dist3d <- function(x1,x2){
  sqrt(sum((x1-x2)**2))
}

ecov_dist <- function(x,compute=T){
  if (compute==T){
    N = dim(x)[1]
    #print(paste('N = ', N, sep=""))
    y <- matrix(0, nrow = N, ncol = N)
    for (i in 2:N){
      for (j in 1:(i-1)){
        y[i,j] <- dist3d(x[i,c("x","y","z")],x[j,c("x","y","z")])
      }
    }
    y <- y + t(y) ; colnames(y) <- rownames(x); rownames(y) <- rownames(x); 
    write.csv(y, file = paste('processed_data/',ID,'_ec_dist.csv',sep=""),row.names = TRUE)
    y
  } else {
    y <- read.csv(paste('processed_data/',ID,'_ec_dist.csv',sep=""),row.names=1)
  }
}

edge_attr_dist <- function(x){
  d <- dist3d(nodes[x[[1]],c("x","y","z")],nodes[x[[2]],c("x","y","z")])
  d
}

create_info_flow_matrix <- function(){
  # 1 : feedforward, -1 : feedback, 0 : recurrent
  y <- matrix(0, nrow = 4, ncol = 4); 
  y[upper.tri(y)] <- 1; y[lower.tri(y)] <- -1; 
  rownames(y) = colnames(y) <- c('sensory','modulatory','inter','motor')
  y
}
info_flow_matrix <- create_info_flow_matrix()

conn_type <- function(x){
  y <- info_flow_matrix[x[[1]],x[[2]]]
  y
}

ecov_conn_type <- function(x,compute=T){
  if (compute==T){
    N = dim(x)[1]
    #print(paste('N = ', N, sep=""))
    y <- matrix(0, nrow = N, ncol = N)
    for (i in 1:N){
      for (j in 1:N){
        y[i,j] <- conn_type(c(x[i,1],x[j,1]))
      }
    }
    colnames(y) = rownames(y) <- rownames(x); 
    write.csv(y, file = paste('processed_data/',ID,'_ec_conn_type.csv',sep=""),row.names=TRUE)
    y
  } else {
    y <- read.csv(paste('processed_data/',ID,'_ec_conn_type.csv',sep=""),row.names=1)
  }
}

# "solid","dashed","dotted","dotdash","longdash","twodash" 
line_styles <- list('solid','solid','solid') # FB, R, FF
line_width <- list(1,1,1) # FB, R, FF
plot_titles <- c('0h','5h','8h','16h','23h','27h','45h-1','45h-2')


#dark_colors$sensory <- '#8C0800'
#dark_colors$inter <- '#001C7F'
#dark_colors$motor <- '#12711C'
#dark_colors$modulatory <- '#B1400D' 
  
dark_colors$sensory <- '#e8000b'
dark_colors$inter <- '#023eff'
dark_colors$motor <- '#1ac938'
dark_colors$modulatory <- '#ffc400' 
        
my_nplot <- function(g) {
  par(bg=NA, mar = c(0, 0, 0, 0))
  plot(g, 
       # === vertex
       vertex.color = sapply(V(g)$ntype, function(x) dark_colors[x][[1]]),
       vertex.shape = 'circle',
       lwd = 0.01,
       # Node color
       vertex.frame.color = 'white', #"#FCF6F5FF",             # Node border color
       vertex.shape="circle",                        # One of “none”, “circle”, “square”, “csquare”, “rectangle” “crectangle”, “vrectangle”, “pie”, “raster”, or “sphere”
       vertex.size=2*log(degree(g)),                                # Size of the node (default is 15)
       # === vertex label
       vertex.label= NA,#V(g)$name,                      # Character vector used to label the nodes
       vertex.label.color="white",
       vertex.label.family="Times",                  # Font family of the label (e.g.“Times”, “Helvetica”)
       vertex.label.font=1,                          # Font: 1 plain, 2 bold, 3, italic, 4 bold italic, 5 symbol
       vertex.label.cex=.4,                          # Font size (multiplication factor, device-dependent)
       vertex.label.dist=0,                          # Distance between the label and the vertex
       vertex.label.degree=0 ,                       # The position of the label in relation to the vertex (use pi)
       # === Edge
       edge.color="black",                           # Edge color
       edge.width=.05* sapply(E(g)$conn_type, function(x) line_width[[x+2]]),                                 # Edge width, defaults to 1
       edge.arrow.size=.05,                           # Arrow size, defaults to 1
       edge.arrow.width=.05,                          # Arrow width, defaults to 1
       edge.lty=sapply(E(g)$conn_type, function(x) line_styles[[x+2]]),                   # Line type, could be 0 or “blank”, 1 or “solid”, 2 or “dashed”, 3 or “dotted”, 4 or “dotdash”, 5 or “longdash”, 6 or “twodash”
       edge.curved=0.1    ,                          # Edge curvature, range 0-1 (FALSE sets it to 0, TRUE to 0.5)
       # === Layout
       #margin = 0.0,
       asp = 1,
       layout = layout_with_fr #layout_nicely 
  )
#  title(plot_titles[ID],cex.main=1.5,col.main="#FCE77D")
}

my_nplot_onecolor <- function(g) {
  par(bg=NA)
  plot(g, 
       # === vertex
       vertex.color = '#ff7f0e',
       vertex.shape = 'circle',
       # Node color
       vertex.frame.color = '#ff7f0e', #"#FCF6F5FF",             # Node border color
       vertex.shape="circle",                        # One of “none”, “circle”, “square”, “csquare”, “rectangle” “crectangle”, “vrectangle”, “pie”, “raster”, or “sphere”
       vertex.size=2*log(degree(g)),                                # Size of the node (default is 15)
       # === vertex label
       vertex.label= NA,#V(g)$name,                      # Character vector used to label the nodes
       vertex.label.color="white",
       vertex.label.family="Times",                  # Font family of the label (e.g.“Times”, “Helvetica”)
       vertex.label.font=1,                          # Font: 1 plain, 2 bold, 3, italic, 4 bold italic, 5 symbol
       vertex.label.cex=.4,                          # Font size (multiplication factor, device-dependent)
       vertex.label.dist=0,                          # Distance between the label and the vertex
       vertex.label.degree=0 ,                       # The position of the label in relation to the vertex (use pi)
       # === Edge
       edge.color="#ff7f0e",                           # Edge color
       edge.width=.5* sapply(E(g)$conn_type, function(x) line_width[[x+2]]),                                 # Edge width, defaults to 1
       edge.arrow.size=.1,                           # Arrow size, defaults to 1
       edge.arrow.width=.1,                          # Arrow width, defaults to 1
       edge.lty='solid',                            # Line type, could be 0 or “blank”, 1 or “solid”, 2 or “dashed”, 3 or “dotted”, 4 or “dotdash”, 5 or “longdash”, 6 or “twodash”
       edge.curved=0.1    ,                          # Edge curvature, range 0-1 (FALSE sets it to 0, TRUE to 0.5)
       # === Layout
       margin = 0.0,
       asp = 1,
       layout = layout_with_fr #layout_nicely 
  )
  #  title(plot_titles[ID],cex.main=1.5,col.main="#FCE77D")
}


# Two fuctions for "adjsforpy.R"

build_adj <- function(g){
  # g is an igraph -- build lower triangular undirected adjacency matrices
  adj <- as_adj(g, type = "lower", names = TRUE, sparse = FALSE)
  adj[which(adj!=0,arr.ind = T)] <- 1
  adj
}

create_adj <- function(x){
  # x is an edgelist -- build lower triangular undirected adjacency matrices
  adj <- matrix(0, nrow = N, ncol = N); rownames(adj) = colnames(adj) <- nnames
  for (i in 1:dim(x)[1]){
    adj[x[i,1],x[i,2]] <- 1.0
  }
  adj <- adj + t(adj)
  adj[which(adj!=0,arr.ind = T)] <- 1.0
  adj[upper.tri(adj,diag=TRUE)] <- 0
  adj
}

create_adj_full <- function(x,directed=TRUE){
  adj <- matrix(0, nrow = N, ncol = N); rownames(adj) = colnames(adj) <- nnames
  for (i in 1:dim(x)[1]){
    adj[x[i,1],x[i,2]] <- 1.0
  }
  if (directed==FALSE){
    adj <- adj + t(adj)   
  }
  adj[which(adj!=0,arr.ind = T)] <- 1.0
  adj
}

undirectedness <- function(adj){
  adj <- adj + t(adj)
  counts <- table(adj)
  d <- counts[[2]]/2
  u <- counts[[3]]/2
  print(paste("d-conn: ", d))
  print(paste("u-conn: ", u))
  print(paste("ratio: ", round(u/(u+d),2)))
}
