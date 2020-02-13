###########################################################################################
########## 1. Generate data from gauss, poisson, gamma, exponential, bernoulli, gev #######
########## 2. Save the data in ./data                                               #######
###########################################################################################

setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) ## set working directory to the location of the running file 


n <- 25e+03 ## sample size
siz <- "25e3"

## smooth functions
f1 <- function(x){ return(0.2 * x^11 * (10 * (1 - x))^6 + 10 * (10 * x)^3 * (1 - x)^10) }
f2 <- function(x){ return(2 * sin(pi * x)) }
f3 <- function(x){ return(exp(2*x)) }

f4 <- function(x){ return(0.1*x^2) }
f5 <- function(x){ return(.5*sin(2*pi*x)) }
f6 <- function(x){ return(-.2-0.5*x^3) }

R <- 100 ## nb replicates

########################
########## Gauss #######
########################
Ldat <- vector("list", R)
Lmu <- matrix(NA, nrow=n, ncol=R)
Lsigma <- matrix(NA, nrow=n, ncol=R)
  
for(r in 1:R){  ## iterate over the replications
  
  # generate functional parameters
  Ldat[[r]] <- data.frame(x1=runif(n), x2=runif(n), x3=runif(n), x4=runif(n), x5=runif(n), x6=runif(n))
  Lmu[,r] <- f1(Ldat[[r]]$x1) + f2(Ldat[[r]]$x2) + f3(Ldat[[r]]$x3)
  Lsigma[,r] <- exp(.5*(f4(Ldat[[r]]$x4) + f5(Ldat[[r]]$x5) + f6(Ldat[[r]]$x6)))
    
  ## generate data
  Ldat[[r]]$y <- rnorm(n, mean=Lmu[,r], sd=Lsigma[,r])
  #save(Ldat, Lmu, Lsigma, file=paste("./data/data", eval(siz), "Gauss.Rdata", sep="")) # in case of crash 
}
#rm(Ldat, Lmu, Lsigma) ## to save memory

##########################
########## Poisson #######
##########################
Ldat <- vector("list", R)
Lmu <- matrix(NA, nrow=n, ncol=R)

for(r in 1:R){  ## iterate over the replications
  
  # generate functional parameters
  Ldat[[r]] <- data.frame(x1=runif(n), x2=runif(n), x3=runif(n))
  Lmu[,r] <- exp( (f1(Ldat[[r]]$x1) + f2(Ldat[[r]]$x2) + f3(Ldat[[r]]$x3))/6)
    
  ## generate data
  Ldat[[r]]$y <- rpois(n, Lmu[,r])
  #save(Ldat, Lmu, file=paste("./data/data", eval(siz), "Poiss.Rdata", sep="")) # in case session crashes
}
#rm(Ldat, Lmu)

##############################
########## Exponential #######
##############################
Ldat <- vector("list", R)
Lmu <- matrix(NA, nrow=n, ncol=R)
  
for(r in 1:R){  ## iterate over the replications
    
  # generate functional parameters
  Ldat[[r]] <- data.frame(x1=runif(n), x2=runif(n), x3=runif(n))
  Lmu[,r] <- exp( (f1(Ldat[[r]]$x1) + f2(Ldat[[r]]$x2) + f3(Ldat[[r]]$x3))/6)
    
  ## generate data
  Ldat[[r]]$y <- rexp(n, Lmu[,r])
  #save(Ldat, Lmu, file=paste("./data/data", eval(siz), "Expon.Rdata", sep="")) # in case session crashes
}
#rm(Ldat, Lmu)

########################
########## Gamma #######
########################
Ldat <- vector("list", R)
Lmu <- matrix(NA, nrow=n, ncol=R)
Lsigma <- matrix(NA, nrow=n, ncol=R)
  
for(r in 1:R){  ## iterate over the replications
  
  # generate functional parameters
  Ldat[[r]] <- data.frame(x1=runif(n), x2=runif(n), x3=runif(n), x4=runif(n), x5=runif(n), x6=runif(n))
  Lmu[,r] <- exp((f1(Ldat[[r]]$x1) + f2(Ldat[[r]]$x2) + f3(Ldat[[r]]$x3))/6)
  Lsigma[,r] <- exp(f4(Ldat[[r]]$x4) + f5(Ldat[[r]]$x5) + f6(Ldat[[r]]$x6))
    
  ## generate data
  Ldat[[r]]$y <- rgamma(n, shape=Lmu[,r], scale=1/Lsigma[,r])
  #save(Ldat, Lmu, Lsigma, file=paste("./data/data", eval(siz), "Gamma.Rdata", sep="")) # in case session crashes
}
#rm(Ldat, Lmu, Lsigma)

######################
########## Gev #######
######################

f7 <- function(x){ return(-0.5*x^2 + sin(pi*x)) }

sim.gev <- function(mu, sigma, xi){
  
  are.0 <- abs(xi) <= .Machine$double.eps^.3
  out <- vector("numeric", length(mu))
  out[are.0] <- mu[are.0] - sigma[are.0] * log(rexp(sum(are.0))) # if u is uniform, -log(u) is exponential
  out[!are.0] <- mu[!are.0] + sigma[!are.0] * (rexp(sum(!are.0))^(-xi[!are.0])-1)/xi[!are.0]
  
  return(out)
}

Ldat <- vector("list", R)
Lmu <- matrix(NA, nrow=n, ncol=R)
Lsigma <- matrix(NA, nrow=n, ncol=R)
Lxi <- matrix(NA, nrow=n, ncol=R)
  
for(r in 1:R){  ## iterate over the replications
  
  # generate functional parameters
  Ldat[[r]] <- data.frame(x1=runif(n), x2=runif(n), x3=runif(n), x4=runif(n), x5=runif(n), x6=runif(n), x7=runif(n))
  Lmu[,r] <- f1(Ldat[[r]]$x1) + f2(Ldat[[r]]$x2) + f3(Ldat[[r]]$x3)
  Lsigma[,r] <- exp(f4(Ldat[[r]]$x4) + f5(Ldat[[r]]$x5) + f6(Ldat[[r]]$x6))
  Lxi[,r] <- f7(Ldat[[r]]$x7)
    
  ## generate data
  Ldat[[r]]$y <- sim.gev(Lmu[,r], Lsigma[,r], Lxi[,r])
  #save(Ldat, Lmu, Lsigma, Lxi, file=paste("./data/data", eval(siz), "Gev.Rdata", sep="")) # in case session crashes
}
#rm(Ldat, Lmu, Lsigma, Lxi)

###########################
########## Binomial #######
###########################
Ldat <- vector("list", R)
Lmu <- matrix(NA, nrow=n, ncol=R)
  
for(r in 1:R){  ## iterate over the replications
  
  # generate functional parameters
  Ldat[[r]] <- data.frame(x1=runif(n), x2=runif(n), x3=runif(n))
  Lmu[,r] <- binomial()$linkinv( (f1(Ldat[[r]]$x1) + f2(Ldat[[r]]$x2) + f3(Ldat[[r]]$x3) -5)/6)
  
  ## generate data
  Ldat[[r]]$y <- rbinom(n, 1, Lmu[,r])
  #save(Ldat, Lmu, file=paste("./data/data", eval(siz), "Binom.Rdata", sep="")) # in case session crashes
}
#rm(Ldat, Lmu)
