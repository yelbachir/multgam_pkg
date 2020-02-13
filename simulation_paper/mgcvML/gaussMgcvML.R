#############################################################################
########## 1. Run simulations on the gauss model by mgcv (gam) library ######
########## 2. Save output results in ../output                         ######
#############################################################################
library(mgcv)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) ## set working directory to the location of the running file 

siz <- "25e3" #as.character(args[7])

R <- 100 ## nb replicates

## outputs
x <- rep(NA, R)
outMgcvML <- list(time=x, mu=x, sigma=x)
rm(x)

k <- 10
L.formula <- list(y ~ s(x1, bs="cr", k=k) + s(x2, bs="cr", k=k) + s(x3, bs="cr", k=k), 
                  ~ s(x4, bs="cr", k=k) + s(x5, bs="cr", k=k) + s(x6, bs="cr", k=k))

for(r in 1:R){ ## iterate over the replications
    
    load(file=paste("../data/data", eval(siz), "Gauss.Rdata", sep=""))
    dat <- Ldat[[r]]
    mu <- Lmu[,r]
    sigma <- Lsigma[,r]
    rm(Ldat, Lmu, Lsigma) ## read one replicate at a time to save memory
    
    fit <- try(as.numeric(system.time(fMgcvML <- gam(data=dat, 
                                                     formula=L.formula, 
                                                     family = gaulss(b=0)
                                                     )
                                      )[3]), TRUE
               )
    if(class(fit) != "try-error"){
      outMgcvML$time[r] <- fit
      outMgcvML$mu[r] <- mean( (mu-fMgcvML$fitted.values[,1])^2 )
      outMgcvML$sigma[r] <- mean( (sigma-1/fMgcvML$fitted.values[,2])^2 )
      rm(fMgcvML)
    }
    #save(outMgcvML, file=paste("../output/out", eval(siz), "GaussMgcvML.Rdata", sep="")) ## save results for every replicates in case of crash
}
