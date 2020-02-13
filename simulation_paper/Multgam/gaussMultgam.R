##########################################################################
########## 1. Run simulations on the gauss model by multgam library ######
########## 2. Save output results in ../output          #           ######
##########################################################################
library(multgam)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) ## set working directory to the location of the running file 

siz <- "25e3"
R <- 100 ## nb replicates

## outputs
x <- rep(NA, R)
outMultgam <- list(time=x, mu=x, sigma=x)
rm(x)

k <- 10
L.formula <- list(y ~ s(x1, bs="cr", k=k) + s(x2, bs="cr", k=k) + s(x3, bs="cr", k=k), 
                  ~ s(x4, bs="cr", k=k) + s(x5, bs="cr", k=k) + s(x6, bs="cr", k=k)
                  )
fmName <- "gauss"

for(r in 1:R){ ## iterate over the replications
    
    load(file=paste("../data/data", eval(siz), "Gauss.Rdata", sep=""))
    dat <- Ldat[[r]]
    mu <- Lmu[,r]
    sigma <- Lsigma[,r]
    rm(Ldat, Lmu, Lsigma) ## read one replicate at a time to save memory
    
    fit <- try(as.numeric(system.time(fMultgam <- mtgam(dat, 
                                                        L.formula, 
                                                        fmName=fmName
                                                        )
                                      )[3]), TRUE
               )
    if(class(fit) != "try-error"){
      outMultgam$time[r] <- fit
      outMultgam$mu[r] <- mean( (mu-fMultgam$fitted.values[,1])^2 )
      outMultgam$sigma[r] <- mean( (sigma-exp(.5*fMultgam$fitted.values[,2]))^2 )
      rm(fMultgam)
    }
    #save(outMultgam, file=paste("../output/out", eval(siz), "GaussMultgam.Rdata", sep="")) ## save results for every replicates in case of crash
}
  
