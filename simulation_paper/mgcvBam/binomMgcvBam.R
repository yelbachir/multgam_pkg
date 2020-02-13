################################################################################
########## 1. Run simulations on the binomial model by mgcv (bam) library ######
########## 2. Save output results in ../output                            ######
################################################################################
library(mgcv)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) ## set working directory to the location of the running file 

siz <- "25e3"

R <- 100 ## nb replicates

## outputs
x <- rep(NA, R)
outMgcvBam <- list(time=x, mu=x)
rm(x)

k <- 10
L.formula <- list(y ~ s(x1, bs="cr", k=k) + s(x2, bs="cr",  k=k) + s(x3, bs="cr",  k=k))

for(r in 1:R){ ## iterate over the replications
    
    load(file=paste("../data/data", eval(siz), "Binom.Rdata", sep=""))
    dat <- Ldat[[r]]
    mu <- Lmu[,r]
    rm(Ldat, Lmu) ## read one replicate at a time to save memory
    
    fit <- try(as.numeric(system.time(fBam <- bam(data=dat,
                                                  formula=L.formula[[1]], 
                                                  family = binomial()
                                                  ) 
                                      )[3]), TRUE
               )
    if(class(fit) != "try-error"){
      outMgcvBam$time[r] <- fit
      outMgcvBam$mu[r] <- mean( (mu-fBam$fitted.values/length(fBam$fitted.values))^2 ) # as E[Xi] = np is fitted.values
      rm(fBam)
    }
    #save(outMgcvBam, file=paste("../output/out", eval(siz), "BinomMgcvBam.Rdata", sep="")) ## save results for every replicates in case of crash
}
