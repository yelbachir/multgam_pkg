#############################################################################
########## 1. Run simulations on the binomial model by multgam library ######
########## 2. Save output results in ../output                         ######
#############################################################################
##if(!require(multgam)) install.packages("multgam")
library(multgam)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) ## set working directory to the location of the running file 

siz <- "25e3"

R <- 100 ## nb replicates

## outputs
x <- rep(NA, R)
outMultgam <- list(time=x, mu=x)
rm(x)

k <- 10
L.formula <- list(y ~ s(x1, bs="cr", k=k) + s(x2, bs="cr",  k=k) + s(x3, bs="cr",  k=k))
fmName <- "binom"

for(r in 1:R){ ## iterate over the replications
    
    load(file=paste("../data/data", eval(siz), "Binom.Rdata", sep=""))
    dat <- Ldat[[r]]
    mu <- Lmu[,r]
    rm(Ldat, Lmu) ## read one replicate at a time to save memory

    fit <- try(as.numeric(system.time(fMultgam <- mtgam(dat, 
                                                        L.formula, 
                                                        fmName=fmName
                                                        )
                                      )[3]), TRUE
               )
    if(class(fit) != "try-error"){
      outMultgam$time[r] <- fit
      outMultgam$mu[r] <- mean( (mu-binomial()$linkinv(fMultgam$fitted.values))^2 ) # as tau = Logit(p) is GAM is fitted.values
      rm(fMultgam)
    }
    #save(outMultgam, file=paste("../output/out", eval(siz), "BinomMultgam.Rdata", sep="")) ## save results for every replicates in case of crash
}
