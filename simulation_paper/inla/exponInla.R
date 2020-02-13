#############################################################################
########## 1. Run simulations on the exponential model by inla library ######
########## 2. Save output results in ../output                         ######
#############################################################################
##if(!require(INLA)) install.packages("INLA", repos=c(getOption("repos"), INLA="https://inla.r-inla-download.org/R/stable"), dep=TRUE)
library(INLA)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) ## set working directory to the location of the running file 

siz <- "25e3"

R <- 100 ## nb replicates

## outputs
x <- rep(NA, R)
outInla <- list(time=x, mu=x)
rm(x)

for(r in 1:R){ ## iterate over the replications
    
    load(file=paste("../data/data", eval(siz), "Expon.Rdata", sep=""))
    dat <- Ldat[[r]]
    mu <- Lmu[,r]
    rm(Ldat, Lmu) ## read one replicate at a time to save memory
    
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
      outInla$time[r] <- fit
      outInla$mu[r] <- mean( (mu-fInla$summary.fitted.values[,1])^2 )
      rm(fInla)
    }
    #save(outInla, file=paste("../output/out", eval(siz), "ExponInla.Rdata", sep="")) ## save results for every replicates in case of crash
}
