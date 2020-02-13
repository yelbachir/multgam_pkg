###################################################################################
########## 1. Run simulations on the gauss model by brms (stan MCMC) library ######
########## 2. Save output results in ../output                               ######
###################################################################################
##if(!require(brms)) install.packages("brms")
##if(!require(rstan)) install.packages("rstan")
library(brms)
library(rstan)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) ## set working directory to the location of the running file 

siz <- "25e3"

R <- 100 ## nb replicates

## outputs
x <- rep(NA, R)
outStanMCMC <- list(time=x, mu=x, sigma=x)
rm(x)

k <- 10

for(r in 1:R){ ## iterate over the replications
    
    load(file=paste("../data/data", eval(siz), "Gauss.Rdata", sep=""))
    dat <- Ldat[[r]]
    mu <- Lmu[,r]
    sigma <- Lsigma[,r]
    rm(Ldat, Lmu, Lsigma) ## read one replicate at a time to save memory
    
    fit <- try(fStan <- brm(bf(y ~ s(x1, bs="cr", k=k) + s(x2, bs="cr", k=k) + s(x3, bs="cr", k=k), 
                               sigma ~ s(x4, bs="cr", k=k) + s(x5, bs="cr", k=k) + s(x6, bs="cr", k=k)), 
                            cores=4,
                            data = dat, 
                            family = gaussian(),
                            silent=TRUE,
                            refresh=0
                            ), TRUE
               )
    if(class(fit) != "try-error"){
      outStanMCMC$time[r] <- sum(get_elapsed_time(fStan$fit))
      outStanMCMC$mu[r] <- mean( (mu - fitted(fStan, dpar="mu", scale="linear")[,1])^2 )
      outStanMCMC$sigma[r] <- mean( (sigma - exp(fitted(fStan, dpar="sigma", scale="linear")[,1]))^2 )
      rm(fStan)
      }
    #save(outStanMCMC, file=paste("../output/out", eval(siz), "GaussStanMCMC", Rstart, ".Rdata", sep="")) ## save results for every replicates in case of crash
}