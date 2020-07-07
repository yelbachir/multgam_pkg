################################
#### Simulation functions ######
################################

sim.gev <- function(mu, sigma, xi){
  
  n <- length(mu)
  if( (n != length(sigma)) | (n != length(xi)) ){
    stop("mu, sigma and xi should have the same length")
    }
  
  are.0 <- abs(xi) <= .Machine$double.eps^.3
  out <- vector("numeric", n)
  out[are.0] <- mu[are.0] - sigma[are.0] * log(rexp(sum(are.0))) # if u is uniform, -log(u) is exponential
  are.0 <- !are.0
  out[are.0] <- mu[are.0] + sigma[are.0] * expm1(-xi[are.0]*log(rexp(sum(are.0))))/xi[are.0]
  
  return(out)
}

sim.gpd <- function(sigma, xi){
  
  n <- length(sigma)
  if( n != length(xi) ){
    stop("sigma and xi should have the same length")
  }
  
  are.0 <- abs(xi) <= .Machine$double.eps^.3
  out <- vector("numeric", n)
  out[are.0] <- sigma[are.0] * rexp(sum(are.0)) # if u is uniform, -log(u) is exponential
  are.0 <- !are.0
  out[are.0] <- sigma[are.0] * (expm1(rexp(sum(are.0))*xi[are.0]))/xi[are.0]
  
  return(out)
}

sim.rgev <- function(mu, sigma, xi, r){
  
  n <- length(mu)
  if( (n != length(sigma)) | (n != length(xi)) ){
    stop("mu, sigma and xi should have the same length")
  }
  
  if(r == 1) {
    outsorted <- sim.gev(mu, sigma, xi)
  } else {
    
    out <- matrix(NA, nrow=n, ncol=r)
    
    for(i in 1:n){
      u <- cumprod(runif(r))
      v <- -log(-log(u))
      if( abs(xi[i]) <= .Machine$double.eps^.3){
        out[i, ] <- mu[i] + sigma[i] * v
      } else {
        out[i, ] <- mu[i] + sigma[i] * expm1(xi[i]*v) /xi[i]
      }
    }
    
    # sort the cols in ascending order
    outsorted <- matrix(NA, nrow=n, ncol=r)
    for(j in 1:r){
      outsorted[,j] <- out[,r-j+1]
    }
  }
  
  return(outsorted)
}

## simulation of event times of a Poisson process with rate lambda_i until time T
#pp.generate <- function(tInterv, rate){
  
#  t <- tInterv[1]
#  tT <- tInterv[2]
#  k <- 0
  
#  out <- vector("numeric")
#  r <- runif(1)
#  t <- t-log(r)/rate
#  while(t <= tT){
#    k <- k+1
#    out[k] <- t
#    r <- runif(1)
#    t <- t-log(r)/rate
#  }
  
#  return(out) ##(t_1, ..., t_ni)
#}

#sim.pp <- function(mu, sigma, xi, u=NULL, tInterval=c(0,1)){
  ## returns a matrix = ( ni | u | yi), where ni is the scalar size of the cluster of the i-th block
  ## if u is not provided, then it will be computed
  
#  n <- length(mu)
#  if( (n != length(sigma)) | (n != length(xi))){
#    stop("mu, sigma and xi should have the same length")
#  }
  
#  are.0 <- abs(xi) <= .Machine$double.eps^.3
#  not.0 <- !are.0
  
  ## define rate of exceedances of the poisson process
#  if(is.null(u)){ ## define the threshold u
#    u <- vector("numeric", n)
#    are.neg <- (xi<0) & not.0
#    u[are.neg] <- mu[are.neg] - 0.5*sigma[are.neg]/xi[are.neg]
    
#    xiPos0 <- are.0 | ((xi>0) & not.0)
#    mu.pos <- (mu>0) & xiPos0
#    u[mu.pos] <- 1.05*mu[mu.pos]
    
#    mu.neg <- (mu<0) & xiPos0
#    u[mu.neg] <- 0.95*mu[mu.neg]
#  }
#  rate <- vector("numeric", n) 
#  rate[are.0] <- exp((mu[are.0]-u[are.0])/sigma[are.0])
#  rate[not.0] <- exp(-log1p(xi[not.0]*(u[not.0]-mu[not.0])/sigma[not.0])/xi[not.0]) 
  
  ## generate a Poisson point process for each block
#  ti <- vector("list", n)
#  ni <- vector("numeric", n)
#  for(i in 1:n){
#    ti[[i]] <- pp.generate(tInterval, rate[i])
#    ni[i] <- length(ti[[i]])
#    }
  
#  out <- matrix(0, nrow=n, ncol=max(ni))
#  for(i in 1:n){
    
#    j <- ni[i]
#    if(j != 0){
  
#      v <- -log(-log(ti[[i]]))
#      if(are.0[i]){
#        out[i, 1:j] <- mu[i] + sigma[i] * v # if u is uniform, -log(u) is exponential
#        } else {
#          out[i, 1:j] <- mu[i] + sigma[i] * expm1(xi[i]*v) /xi[i]
#        }
#    }
#  }
#  out <- cbind(ni, u, out)
  
#  return(out)
#}

simExtrem <- function(mu=NULL, sigma=NULL, xi=NULL, r=NULL, family="gev"){
##simExtrem <- function(mu=NULL, sigma=NULL, xi=NULL, r=NULL, tInterval=c(0,1), family="gev"){
  
  if(family=="gev"){
    
    out <- sim.gev(mu, sigma, xi)
    return(out)
    
  } else if(family=="gpd"){
    
    out <- sim.gpd(sigma, xi)
    return(out)
    
  } else if(family=="rgev"){
    
    out <- sim.rgev(mu, sigma, xi, r)
    return(out)
    
  } #else if(family=="pp"){ ## r is  vector of u_i
    
    #out <- sim.pp(mu, sigma, xi, r, tInterval)
    #return(out)
  #}
}

#########################
#### Return levels ######
#########################

rl.gev <- function(prob, mu, sigma, xi){
  
  n <- length(mu)
  if( (n != length(sigma)) | (n != length(xi)) ){
    stop("mu, sigma and xi should have the same length")
  }
  
  are.0 <- abs(xi) <= .Machine$double.eps^.3
  out <- vector("numeric", n)
  out[are.0] <- mu[are.0] - sigma[are.0] * log(-log(prob))
  are.0 <- !are.0
  out[are.0] <- mu[are.0] + sigma[are.0] * expm1(-xi[are.0]*log(-log(prob)))/xi[are.0] ## (rexp(sum(not.0))^(-xi[not.0])-1)/xi[not.0]
  
  return(out)
}

rl.gpd <- function(prob, sigma, xi){
  
  n <- length(sigma)
  if( n != length(xi) ){
    stop("sigma and xi should have the same length")
  }
  
  are.0 <- abs(xi) <= .Machine$double.eps^.3
  out <- vector("numeric", n)
  out[are.0] <- -sigma[are.0] * log(1-p)
  are.0 <- !are.0
  out[are.0] <- sigma[are.0] * (expm1(-xi[are.0]*log(1-p)))/xi[are.0]
  
  return(out)
  
}

returnLevel <- function(prob=NULL, mu=NULL, sigma=NULL, xi=NULL, family="gev"){
##returnLevel <- function(prob=NULL, mu=NULL, sigma=NULL, xi=NULL, r=NULL, family="gev"){
  
  if(family=="gev"){
    
    out <- rl.gev(prob, mu, sigma, xi)
    return(out)
    
  } else if(family=="gpd"){
    
    out <- rl.gpd(prob, sigma, xi)
    return(out)
    
  } 
}