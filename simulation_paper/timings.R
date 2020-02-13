###################################################################################################################################
##########    Reproduce the timing results of Section 3 in the paper for gauss, poisson, gamma, exponential, bernoulli, gev  ######
########## 1. Generate the data and save them in ./data                                                                      ######
########## 2. Save output results in ./output                                                                                ######
###################################################################################################################################

setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) ## set working directory to the location of the running file 

library(multgam)
library(mgcv)
library(INLA)
library(brms)

#################################################
######################## Gauss ##################
#################################################

N <- c(1e+05, 5e+05)
nName <- c("1e5", "5e5")

NN <- length(N)
time <- matrix(NA, nrow=6, ncol=NN)
rownames(time) <- c("Multgam", "Mgcv Bam", "Mgcv ML", "Inla", "Stan MCMC", "Stan VI")
colnames(time) <- as.character(N)


## generate the data and fit the model by the different libraries
f1 <- function(x){ return(0.2 * x^11 * (10 * (1 - x))^6 + 10 * (10 * x)^3 * (1 - x)^10) }
f2 <- function(x){ return(2 * sin(pi * x)) }
f3 <- function(x){ return(exp(2*x)) }

f4 <- function(x){ return(0.1*x^2) }
f5 <- function(x){ return(.5*sin(2*pi*x)) }
f6 <- function(x){ return(-.2-0.5*x^3) }

k <- 10
L.formula <- list(y ~ s(x1, bs="cr", k=k) + s(x2, bs="cr", k=k) + s(x3, bs="cr", k=k), 
                  ~ s(x4, bs="cr", k=k) + s(x5, bs="cr", k=k) + s(x6, bs="cr", k=k))

for(i in 1:NN){
  
  n <- N[i]
  siz <- nName[i]
  
  dat <- data.frame(x1=runif(n), x2=runif(n), x3=runif(n), x4=runif(n), x5=runif(n), x6=runif(n))
  mu <- f1(dat$x1) + f2(dat$x2) + f3(dat$x3)
  sigma <- exp(.5*(f4(dat$x4) + f5(dat$x5) + f6(dat$x6)))
  dat$y <- rnorm(n, mean=mu, sd=sigma)
  rm(mu, sigma)
  
  #save(dat,  file=paste("./data/dat", siz, "TimeGauss.Rdata", sep=""))
  
  ##==================== Multgam
  fit <- try(as.numeric(system.time(fMultgam <- mtgam(dat, 
                                                      L.formula, 
                                                      fmName="gauss"
  )
  )[3]), TRUE
  )
  if(class(fit) != "try-error"){
    time[1,i] <- fit
    rm(fMultgam)
  }
  #save(time, file=paste("./output/out", siz, "TimeGauss.Rdata", sep="")) ## save results for every replicates in case of crash
  
  
  ##========= mgcv gam
  fit <- try(as.numeric(system.time(fMgcvML <- gam(data=dat, 
                                                     formula=L.formula, 
                                                     family = gaulss(b=0)
    )
    )[3]), TRUE
    )
    if(class(fit) != "try-error"){
      time[3,i] <- fit
      rm(fMgcvML)
    }
  
  #save(time, file=paste("./output/out", siz, "TimeGauss.Rdata", sep="")) ## save results for every replicates in case of crash
}

rm(list=ls())
gc()

#################################################
######################## Gamma ##################
#################################################
library(multgam)

N <- c(1e+05, 5e+05)
nName <- c("1e5", "5e5")

NN <- length(N)
time <- matrix(NA, nrow=6, ncol=NN)
rownames(time) <- c("Multgam", "Mgcv Bam", "Mgcv ML", "Inla", "Stan MCMC", "Stan VI")
colnames(time) <- as.character(N)

## generate the data and fit the model by the different libraries
f1 <- function(x){ return(0.2 * x^11 * (10 * (1 - x))^6 + 10 * (10 * x)^3 * (1 - x)^10) }
f2 <- function(x){ return(2 * sin(pi * x)) }
f3 <- function(x){ return(exp(2*x)) }

f4 <- function(x){ return(0.1*x^2) }
f5 <- function(x){ return(.5*sin(2*pi*x)) }
f6 <- function(x){ return(-.2-0.5*x^3) }

k <- 10
L.formula <- list(y ~ s(x1, bs="cr", k=k) + s(x2, bs="cr", k=k) + s(x3, bs="cr", k=k), 
                  ~ s(x4, bs="cr", k=k) + s(x5, bs="cr", k=k) + s(x6, bs="cr", k=k))

for(i in 1:NN){
  
  n <- N[i]
  siz <- nName[i]
  
  dat <- data.frame(x1=runif(n), x2=runif(n), x3=runif(n), x4=runif(n), x5=runif(n), x6=runif(n))
  mu <- exp((f1(dat$x1) + f2(dat$x2) + f3(dat$x3))/6)
  sigma <- exp(f4(dat$x4) + f5(dat$x5) + f6(dat$x6))
  dat$y <- rgamma(n, shape=mu, scale=1/sigma)
  rm(mu, sigma)
  
  #save(dat,  file=paste("./data/dat", siz, "TimeGamma.Rdata", sep=""))
  
  ##==================== Multgam
  fit <- try(as.numeric(system.time(fMultgam <- mtgam(dat, 
                                                      L.formula, 
                                                      fmName="gamma"
  )
  )[3]), TRUE
  )
  if(class(fit) != "try-error"){
    time[1,i] <- fit
    rm(fMultgam)
  }
  
  #save(time, file=paste("./output/out", siz, "TimeGamma.Rdata", sep="")) ## save results for every replicates in case of crash
  
}

rm(list=ls())
gc()
  
#######################################################
######################## Exponential ##################
#######################################################
library(multgam)
library(mgcv)
library(INLA)
library(brms)

N <- c(1e+05, 5e+05)
nName <- c("1e5", "5e5")

NN <- length(N)
time <- matrix(NA, nrow=6, ncol=NN)
rownames(time) <- c("Multgam", "Mgcv Bam", "Mgcv ML", "Inla", "Stan MCMC", "Stan VI")
colnames(time) <- as.character(N)

## generate the data and fit the model by the different libraries
f1 <- function(x){ return(0.2 * x^11 * (10 * (1 - x))^6 + 10 * (10 * x)^3 * (1 - x)^10) }
f2 <- function(x){ return(2 * sin(pi * x)) }
f3 <- function(x){ return(exp(2*x)) }

k <- 10
L.formula <- list(y ~ s(x1, bs="cr", k=k) + s(x2, bs="cr", k=k) + s(x3, bs="cr", k=k))

for(i in 1:NN){
  
  n <- N[i]
  siz <- nName[i]
  
  dat <- data.frame(x1=runif(n), x2=runif(n), x3=runif(n))
  mu <- exp( (f1(dat$x1) + f2(dat$x2) + f3(dat$x3))/6)
  dat$y <- rexp(n, mu)
  rm(mu)
  
  #save(dat,  file=paste("./data/dat", siz, "TimeExpon.Rdata", sep=""))
  
  ##==================== Multgam
  fit <- try(as.numeric(system.time(fMultgam <- mtgam(dat, 
                                                      L.formula, 
                                                      fmName="expon"
  )
  )[3]), TRUE
  )
  if(class(fit) != "try-error"){
    time[1,i] <- fit
    rm(fMultgam)
  }

  #save(time, file=paste("./output/out", siz, "TimeExpon.Rdata", sep="")) ## save results for every replicates in case of crash
  
  ##========= mgcv Bam
  fit <- try(as.numeric(system.time(fBam <- bam(data=dat,
                                                formula=L.formula[[1]],
                                                family=Gamma("log")
  )
  )[3]), TRUE
  )
  if(class(fit) != "try-error"){
    time[2,i] <- fit
    rm(fBam)
  }

  #save(time, file=paste("./output/out", siz, "TimeExpon.Rdata", sep="")) ## save results for every replicates in case of crash
  
  ##========= mgcv gam
  fit <- try(as.numeric(system.time(fMgcvML <- gam(data=dat, 
                                                   formula=L.formula[[1]], 
                                                   family=Gamma("log")
                                                   )
                                    )[3]), TRUE
             )
  if(class(fit) != "try-error"){
    time[3,i] <- fit
    rm(fMgcvML)
  }
  
  #save(time, file=paste("./output/out", siz, "TimeExpon.Rdata", sep="")) ## save results for every replicates in case of crash
  
  ##=================== Inla
  fit <- try(as.numeric(system.time(fInla <- inla(y~f(x1, model="rw2")+f(x2, model="rw2")+f(x3, model="rw2"), 
                                                  data=dat, 
                                                  verbose=FALSE, 
                                                  family="exponential",
                                                  num.threads=8, 
                                                  control.predictor=list(compute=TRUE)
  )
  )[3]), TRUE
  )
  if(class(fit) == "try-error"){
    fit <- try(as.numeric(system.time(fInla <- inla(y~f(inla.group(x1), model="rw2")+f(inla.group(x2), model="rw2")+f(inla.group(x3), model="rw2"), 
                                                    data=dat, 
                                                    verbose=FALSE, 
                                                    family="exponential",
                                                    num.threads=8, 
                                                    control.predictor=list(compute=TRUE)
    )
    )[3]), TRUE
    )
  }
  if(class(fit) != "try-error"){
    time[4,i] <- fit
    rm(fInla)
  }
  
  #save(time, file=paste("./output/out", siz, "TimeExpon.Rdata", sep="")) ## save results for every replicates in case of crash
}

rm(list=ls())
gc()

###################################################
######################## Poisson ##################
###################################################
library(multgam)
library(mgcv)
library(INLA)
library(brms)

N <- c(1e+05, 5e+05)
nName <- c("1e5", "5e5")

NN <- length(N)
time <- matrix(NA, nrow=6, ncol=NN)
rownames(time) <- c("Multgam", "Mgcv Bam", "Mgcv ML", "Inla", "Stan MCMC", "Stan VI")
colnames(time) <- as.character(N)

## generate the data and fit the model by the different libraries
f1 <- function(x){ return(0.2 * x^11 * (10 * (1 - x))^6 + 10 * (10 * x)^3 * (1 - x)^10) }
f2 <- function(x){ return(2 * sin(pi * x)) }
f3 <- function(x){ return(exp(2*x)) }

k <- 10
L.formula <- list(y ~ s(x1, bs="cr", k=k) + s(x2, bs="cr", k=k) + s(x3, bs="cr", k=k))

for(i in 1:NN){
  
  n <- N[i]
  siz <- nName[i]
  
  dat <- data.frame(x1=runif(n), x2=runif(n), x3=runif(n))
  mu <- exp( (f1(dat$x1) + f2(dat$x2) + f3(dat$x3))/6)
  dat$y <- rpois(n, mu)
  rm(mu)

  #save(dat, file=paste("./data/dat", siz, "TimePoiss.Rdata", sep=""))
  
  ##==================== Multgam
  fit <- try(as.numeric(system.time(fMultgam <- mtgam(dat, 
                                                      L.formula, 
                                                      fmName="poisson"
  )
  )[3]), TRUE
  )
  if(class(fit) != "try-error"){
    time[1,i] <- fit
    rm(fMultgam)
  }
  
  #save(time, file=paste("./output/out", siz, "TimePoiss.Rdata", sep="")) ## save results for every replicates in case of crash
  
  
  ##========= mgcv Bam
  fit <- try(as.numeric(system.time(fBam <- bam(data=dat,
                                                formula=L.formula[[1]], 
                                                family=poisson()
  )
  )[3]), TRUE
  )
  if(class(fit) != "try-error"){
    time[2,i] <- fit
    rm(fBam)
  }
 
  #save(time, file=paste("./output/out", siz, "TimePoiss.Rdata", sep="")) ## save results for every replicates in case of crash
  
  ##========= mgcv gam
  fit <- try(as.numeric(system.time(fMgcvML <- gam(data=dat,
                                                   formula=L.formula[[1]], 
                                                   family=poisson()
  )
  )[3]), TRUE
  )
  if(class(fit) != "try-error"){
    time[3,i] <- fit
    rm(fMgcvML)
  }

  #save(time, file=paste("./output/out", siz, "TimePoiss.Rdata", sep="")) ## save results for every replicates in case of crash
  
  ##=================== Inla
  fit <- try(as.numeric(system.time(fInla <- inla(y~f(x1, model="rw2")+f(x2, model="rw2")+f(x3, model="rw2"), 
                                                  data=dat, 
                                                  verbose=FALSE, 
                                                  family="poisson",
                                                  num.threads=8, 
                                                  control.predictor=list(compute=TRUE)
  )
  )[3]), TRUE
  )
  if(class(fit) == "try-error"){
    fit <- try(as.numeric(system.time(fInla <- inla(y~f(inla.group(x1), model="rw2")+f(inla.group(x2), model="rw2")+f(inla.group(x3), model="rw2"), 
                                                    data=dat, 
                                                    verbose=FALSE, 
                                                    family="poisson",
                                                    num.threads=8, 
                                                    control.predictor=list(compute=TRUE)
    )
    )[3]), TRUE
    )
  }
  if(class(fit) != "try-error"){
    time[4,i] <- fit
    rm(fInla)
  }

  #save(time, file=paste("./output/out", siz, "TimePoiss.Rdata", sep="")) ## save results for every replicates in case of crash
}

rm(list=ls())
gc()

####################################################
######################## Binomial ##################
####################################################

library(multgam)
library(mgcv)
library(INLA)
library(brms)

N <- c(1e+05, 5e+05)
nName <- c("1e5", "5e5")

NN <- length(N)
time <- matrix(NA, nrow=6, ncol=NN)
rownames(time) <- c("Multgam", "Mgcv Bam", "Mgcv ML", "Inla", "Stan MCMC", "Stan VI")
colnames(time) <- as.character(N)

## generate the data and fit the model by the different libraries
f1 <- function(x){ return(0.2 * x^11 * (10 * (1 - x))^6 + 10 * (10 * x)^3 * (1 - x)^10) }
f2 <- function(x){ return(2 * sin(pi * x)) }
f3 <- function(x){ return(exp(2*x)) }

k <- 10
L.formula <- list(y ~ s(x1, bs="cr", k=k) + s(x2, bs="cr", k=k) + s(x3, bs="cr", k=k))

for(i in 1:NN){
  
  n <- N[i]
  siz <- nName[i]
  
  dat <- data.frame(x1=runif(n), x2=runif(n), x3=runif(n))
  mu <- binomial()$linkinv( (f1(dat$x1) + f2(dat$x2) + f3(dat$x3) -5)/6)
  dat$y <- rbinom(n, 1, mu)
  rm(mu)
  
  #save(dat,  file=paste("./data/dat", siz, "TimeBinom.Rdata", sep=""))
  
  ##==================== Multgam
  fit <- try(as.numeric(system.time(fMultgam <- mtgam(dat, 
                                                      L.formula, 
                                                      fmName="binom"
  )
  )[3]), TRUE
  )
  if(class(fit) != "try-error"){
    time[1,i] <- fit
    rm(fMultgam)
  }

  #save(time, file=paste("./output/out", siz, "TimeBinom.Rdata", sep="")) ## save results for every replicates in case of crash
  
  
  ##========= mgcv Bam
  fit <- try(as.numeric(system.time(fBam <- bam(data=dat,
                                                formula=L.formula[[1]], 
                                                family = binomial()
  ) 
  )[3]), TRUE
  )
  if(class(fit) != "try-error"){
    time[2,i] <- fit
    rm(fBam)
  }

  #save(time, file=paste("./output/out", siz, "TimeBinom.Rdata", sep="")) ## save results for every replicates in case of crash
  
  ##========= mgcv gam
  fit <- try(as.numeric(system.time(fMgcvML <- gam(data=dat, 
                                                   formula=L.formula[[1]], 
                                                   family = binomial()
  ) 
  )[3]), TRUE
  )
  if(class(fit) != "try-error"){
    time[3,i] <- fit
    rm(fMgcvML)
  }

  #save(time, file=paste("./output/out", siz, "TimeBinom.Rdata", sep="")) ## save results for every replicates in case of crash
  
  ##=================== Inla
  fit <- try(as.numeric(system.time(fInla <- inla(y~f(x1, model="rw2")+f(x2, model="rw2")+f(x3, model="rw2"), 
                                                  data=dat, 
                                                  verbose=FALSE, 
                                                  family="binomial", Ntrials=nrow(dat),
                                                  num.threads=8, 
                                                  control.predictor=list(compute=TRUE)
  )
  )[3]), TRUE
  )
  if(class(fit) == "try-error"){
    fit <- try(as.numeric(system.time(fInla <- inla(y~f(inla.group(x1), model="rw2")+f(inla.group(x2), model="rw2")+f(inla.group(x3), model="rw2"), 
                                                    data=dat, 
                                                    verbose=FALSE, 
                                                    family="binomial", Ntrials=nrow(dat),
                                                    num.threads=8, 
                                                    control.predictor=list(compute=TRUE)
    )
    )[3]), TRUE
    )
  }
  if(class(fit) != "try-error"){
    time[4,i] <- fit
    rm(fInla)
  }
  
  #save(time, file=paste("./output/out", siz, "TimeBinom.Rdata", sep="")) ## save results for every replicates in case of crash

}

rm(list=ls())
gc()


###############################################
######################## GEV ##################
###############################################
library(multgam)
library(mgcv)
library(INLA)
library(brms)

N <- c(1e+05, 5e+05)
nName <- c("1e5", "5e5")

NN <- length(N)
time <- matrix(NA, nrow=6, ncol=NN)
rownames(time) <- c("Multgam", "Mgcv Bam", "Mgcv ML", "Inla", "Stan MCMC", "Stan VI")
colnames(time) <- as.character(N)

## generate the data and fit the model by the different libraries
f1 <- function(x){ return(0.2 * x^11 * (10 * (1 - x))^6 + 10 * (10 * x)^3 * (1 - x)^10) }
f2 <- function(x){ return(2 * sin(pi * x)) }
f3 <- function(x){ return(exp(2*x)) }

f4 <- function(x){ return(0.1*x^2) }
f5 <- function(x){ return(.5*sin(2*pi*x)) }
f6 <- function(x){ return(-.2-0.5*x^3) }

f7 <- function(x){ return(-0.5*x^2 + sin(pi*x)) }

sim.gev <- function(mu, sigma, xi){
  
  are.0 <- abs(xi) <= .Machine$double.eps^.3
  out <- vector("numeric", length(mu))
  out[are.0] <- mu[are.0] - sigma[are.0] * log(rexp(sum(are.0))) # if u is uniform, -log(u) is exponential
  out[!are.0] <- mu[!are.0] + sigma[!are.0] * (rexp(sum(!are.0))^(-xi[!are.0])-1)/xi[!are.0]
  
  return(out)
}

k <- 10
L.formula <- list(y ~ s(x1, bs="cr", k=k) + s(x2, bs="cr", k=k) + s(x3, bs="cr", k=k), 
                  ~ s(x4, bs="cr", k=k) + s(x5, bs="cr", k=k) + s(x6, bs="cr", k=k), 
                  ~ s(x7, bs="cr")
                  )

for(i in 1:NN){
  
  n <- N[i]
  siz <- nName[i]
  
  dat <- data.frame(x1=runif(n), x2=runif(n), x3=runif(n), x4=runif(n), x5=runif(n), x6=runif(n), x7=runif(n))
  mu <- f1(dat$x1) + f2(dat$x2) + f3(dat$x3)
  sigma <- exp(f4(dat$x4) + f5(dat$x5) + f6(dat$x6))
  xi <- f7(dat$x7)
  dat$y <- sim.gev(mu, sigma, xi)
  rm(mu, sigma, xi)

  #save(dat,  file=paste("./data/dat", siz, "TimeGev.Rdata", sep=""))
  
  ##==================== Multgam
  #ListConvInfo=list("iterMax"=200, "progressPen"=FALSE, "PenTol"=1e-07, "progressML"=FALSE, "MLTol"=1e-06)
  fit <- try(as.numeric(system.time(fMultgam <- mtgam(dat, 
                                                      L.formula, 
                                                      fmName="gev", ListConvInfo = ListConvInfo
  )
  )[3]), TRUE
  )
  if(class(fit) != "try-error"){
    time[1,i] <- fit
    rm(fMultgam)
  }
 
  #save(time, file=paste("./output/out", siz, "TimeGev.Rdata", sep="")) ## save results for every replicates in case of crash
  
  
  ##========= mgcv gam
  fit <- try(as.numeric(system.time(fMgcvML <- gam(data=dat,
                                                   formula=L.formula, 
                                                   family=gevlss(link=list("identity", "identity", "identity"))
  )
  )[3]), TRUE
  )
  if(class(fit) != "try-error"){
    time[3,i] <- fit
    rm(fMgcvML)
  }
  
  #save(time, file=paste("./output/out", siz, "TimeGev.Rdata", sep="")) ## save results for every replicates in case of crash
  
}

rm(list=ls())
gc()