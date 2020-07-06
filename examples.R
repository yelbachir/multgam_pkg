######################################################
## Examples for section 2.2.3. of the documentation
## in the README.md file
######################################################

library(multgam)

n <- 20e+03 ## sample size

## smooth functions
f1 <- function(x){ return(0.2 * x^11 * (10 * (1 - x))^6 + 10 * (10 * x)^3 * (1 - x)^10) }
f2 <- function(x){ return(2 * sin(pi * x)) }
f3 <- function(x){ return(exp(2*x)) }
f4 <- function(x){ return(0.1*x^2) }
f5 <- function(x){ return(.5*sin(2*pi*x)) }
f6 <- function(x){ return(-.2-0.5*x^3) }
f7 <- function(x){ return(-0.45*x^2 + .55*sin(pi*x)) }

#######################################
########## 1. Usage of groupReg #######
#######################################

## generate functional parameters
datGauss <- data.frame(x1=runif(n), x2=runif(n), x3=runif(n), x4=runif(n), x5=runif(n), x6=runif(n))
muGauss <- datGauss$x4 + datGauss$x5 + f1(datGauss$x1) + f2(datGauss$x2) + f3(datGauss$x3)
sigmaGauss <- exp(.5*( datGauss$x1 + datGauss$x2 + f4(datGauss$x4) + f5(datGauss$x5) + f6(datGauss$x6)))

## generate data
datGauss$y <- rnorm(n, mean=muGauss, sd=sigmaGauss)

## fit model
L.formula <- list(y ~ x4 + x5 + s(x1, bs="cr") + s(x2, bs="cr") + s(x3, bs="cr"), ## additive structure for mu
                  ~ x1 + x2 + s(x4, bs="cr") + s(x5, bs="cr") + s(x6, bs="cr")) ## additive structure for tau     
groupReg1 <- list(c(1,1), 2) ## mu = beta_0 + beta_4 x4 + beta_5 x5 + f_1(x1) + f_2(x2) + f_3(x3), 
## where beta_4 is constrained by lambda_4 and beta_5 is constrained by lambda_5, 
## and all the beta of f_j are constrained by their corresponding lambda_j,
## whereas the beta_j of x1 and x2 for tau are constrained by the same lambda
fit1 <- mtgam(dat=datGauss, L.formula=L.formula, fmName="gauss", groupReg=groupReg1)
fit1$sp ## learned hyper-parameters: the first correspond to x4, the second to x5, the third to f_1(x1),
## the fourth to f_2(x2), the fifth to f_3(x_3), the sixth to x1 and x2, the seventh to f_4(x4),
## the eighth to f_5(x5) and the nineth to f_6(x6)

groupReg2 <- list(2, 2) ## the beta_j of x4 and x5 for mu are constrained by the same lambda
## and the beta_j of x1 and x2 for tau are constrained by the same lambda  
fit2 <- mtgam(dat=datGauss, L.formula=L.formula, fmName="gauss", groupReg=groupReg2)
fit2$sp ## learned hyper-parameters: the first correspond to x4 and x5, the second to f_1(x1),
## the third to f_2(x2), the fourth to f_3(x_3), the fifth to x1 and x2, the sixth to f_4(x4),
## the seventh to f_5(x5) and the eighth to f_6(x6)

## example with offset on tau only
muGauss <- datGauss$x4 + datGauss$x5 + f1(datGauss$x1) + f2(datGauss$x2) + f3(datGauss$x3)
sigmaGauss <- exp(.5*(rep(0, n)))
datGauss$y <- rnorm(n, mean=muGauss, sd=sigmaGauss)
L.formula <- list(y ~ x4 + x5 + s(x1, bs="cr") + s(x2, bs="cr") + s(x3, bs="cr"), ## additive structure for mu
                  ~ 1)     
groupReg3 <- list(c(1,1), 0) 
fit3 <- mtgam(dat=datGauss, L.formula=L.formula, fmName="gauss", groupReg=groupReg3)
fit3$sp ## learned hyper-parameters: the first correspond to x4, the second to x5, the third to f_1(x1),
## the fourth to f_2(x2), the fifth to f_3(x_3)

##############################
########## 2. Examples #######
##############################

####################
## Gaussian model ##
####################

## generate functional parameters
datGauss <- data.frame(x1=runif(n), x2=runif(n), x3=runif(n), x4=runif(n), x5=runif(n), x6=runif(n))
muGauss <- f1(datGauss$x1) + f2(datGauss$x2) + f3(datGauss$x3)
sigmaGauss <- exp(.5*(f4(datGauss$x4) + f5(datGauss$x5) + f6(datGauss$x6)))

## generate data
datGauss$y <- rnorm(n, mean=muGauss, sd=sigmaGauss)

## fit model
L.formula <- list(y ~ s(x1, bs="cr") + s(x2, bs="cr") + s(x3, bs="cr"), 
                  ~ s(x4, bs="cr") + s(x5, bs="cr") + s(x6, bs="cr"))                
fit <- mtgam(dat=datGauss, L.formula=L.formula, fmName="gauss")
fit$fitted.values[,1] ## fitted mu
fit$fitted.values[,2] ## fitted tau

###################
## Poisson model ##
###################

## generate functional parameters
datPoiss <- data.frame(x1=runif(n), x2=runif(n), x3=runif(n))
muPoiss <- exp( (f1(datPoiss$x1) + f2(datPoiss$x2) + f3(datPoiss$x3))/6)

## generate data
datPoiss$y <- rpois(n, muPoiss)

## fit model
L.formula <- list(y ~ s(x1, bs="cr") + s(x2, bs="cr") + s(x3, bs="cr"))
fit <- mtgam(dat=datPoiss, L.formula=L.formula, fmName="poisson")
fit$fitted.values[,1] ## fitted mu

#######################
## Exponential model ##
#######################

## generate functional parameters
datExp <- data.frame(x1=runif(n), x2=runif(n), x3=runif(n))
muExp <- exp( (f1(datExp$x1) + f2(datExp$x2) + f3(datExp$x3))/6)

## generate data
datExp$y <- rexp(n, muExp)

## fit model
L.formula <- list(y ~ s(x1, bs="cr") + s(x2, bs="cr") + s(x3, bs="cr"))
fit <- mtgam(dat=datExp, L.formula=L.formula, fmName="expon")
fit$fitted.values[,1] ## fitted mu

#################
## Gamma model ##
#################

## generate functional parameters
datGamma <- data.frame(x1=runif(n), x2=runif(n), x3=runif(n), x4=runif(n), x5=runif(n), x6=runif(n))
muGamma <- exp((f1(datGamma$x1) + f2(datGamma$x2) + f3(datGamma$x3))/6)
sigmaGamma <- exp(f4(datGamma$x4) + f5(datGamma$x5) + f6(datGamma$x6))

## generate data
datGamma$y <- rgamma(n, shape=muGamma, scale=1/sigmaGamma)

## fit model
L.formula <- list(y ~ s(x1, bs="cr") + s(x2, bs="cr") + s(x3, bs="cr"), 
                  ~ s(x4, bs="cr") + s(x5, bs="cr") + s(x6, bs="cr"))
fit <- mtgam(dat=datGamma, L.formula=L.formula, fmName="gamma")
fit$fitted.values[,1] ## fitted mu
fit$fitted.values[,2] ## fitted tau

####################
## Binomial model ##
####################

## generate functional parameters
datBinom <- data.frame(x1=runif(n), x2=runif(n), x3=runif(n))
muBinom <- binomial()$linkinv( (f1(datBinom$x1) + f2(datBinom$x2) + f3(datBinom$x3) -5)/6)

## generate data
datBinom$y <- rbinom(n, 1, muBinom)

# fit model
L.formula <- list(y ~ s(x1, bs="cr") + s(x2, bs="cr") + s(x3, bs="cr"))
fit <- mtgam(dat=datBinom, L.formula=L.formula, fmName="binom")
fit$fitted.values[,1] ## fitted mu

###############
## GEV model ##
###############

## generate functional parameters
datGEV <- data.frame(x1=runif(n), x2=runif(n), x3=runif(n), x4=runif(n), x5=runif(n), x6=runif(n), x7=runif(n))
muGEV <- f1(datGEV$x1) + f2(datGEV$x2) + f3(datGEV$x3)
sigmaGEV <- exp(f4(datGEV$x4) + f5(datGEV$x5) + f6(datGEV$x6))
xiGEV <- f7(datGEV$x7)

## generate data
datGEV$y <- simExtrem(mu=muGEV, sigma=sigmaGEV, xi=xiGEV, family="gev")

## fit model
L.formula <- list(y ~ s(x1, bs="cr") + s(x2, bs="cr") + s(x3, bs="cr"), 
                  ~ s(x4, bs="cr") + s(x5, bs="cr") + s(x6, bs="cr"),  
                  ~ s(x7, bs="cr"))                
fit <- mtgam(dat=datGEV, L.formula=L.formula, fmName="gev")
fit$fitted.values[,1] ## fitted mu
fit$fitted.values[,2] ## fitted tau
fit$fitted.values[,3] ## fitted xi

##############
# GPD model ##
##############

## generate functional parameters
datGPD <- data.frame(x4=runif(n), x5=runif(n), x6=runif(n), x7=runif(n))
sigmaGPD <- exp(f4(datGPD$x4) + f5(datGPD$x5) + f6(datGPD$x6))
xiGPD <- f7(datGPD$x7)

## generate data
datGPD$y <- simExtrem(sigma=sigmaGPD, xi=xiGPD, family="gpd")

## fit model
L.formula <- list(y ~ s(x4, bs="cr") + s(x5, bs="cr") + s(x6, bs="cr"),  
                  ~ s(x7, bs="cr"))
fit <- mtgam(dat=datGPD, L.formula=L.formula, fmName="gpd")
fit$fitted.values[,1] ## fitted tau
fit$fitted.values[,2] ## fitted xi

################
## rGEV model ##
################

## generate functional parameters
datrGEV <- data.frame(x1=runif(n), x2=runif(n), x3=runif(n), x4=runif(n), x5=runif(n), x6=runif(n), x7=runif(n))
murGEV <- f1(datrGEV$x1) + f2(datrGEV$x2) + f3(datrGEV$x3)
sigmarGEV <- exp(f4(datrGEV$x4) + f5(datrGEV$x5) + f6(datrGEV$x6))
xirGEV <- f7(datrGEV$x7)

## generate data
rl <- 10
datrGEV$y <- simExtrem(mu=murGEV, sigma=sigmarGEV, xi=xirGEV, r=rl, family="rgev")

## fit model
L.formula <- list(y ~ s(x1, bs="cr") + s(x2, bs="cr") + s(x3, bs="cr"), 
                  ~ s(x4, bs="cr") + s(x5, bs="cr") + s(x6, bs="cr"),  
                  ~ s(x7, bs="cr"))
fit <- mtgam(dat=datrGEV, L.formula=L.formula, fmName="rgev")
fit$fitted.values[,1] ## fitted mu
fit$fitted.values[,2] ## fitted tau
fit$fitted.values[,3] ## fitted xi

##########################################################
########## 3. Definition of `dat` for the PP model #######
##########################################################

## for the PP model, assume that the dataset is decomposed in n blocks, 
## each of which contains n_i exceedances above the threshold u_i, such that:
## - u: vector (of length n) of u_i
## - Ni: vector (of length n) of n_i
## - y_i: vector (of length n_i) of the threshold exceedances for the i-th block,
## then (in pseudo code):

N <- max(Ni)
Yi <- matrix(NA, nrow=n, ncol=N)
for(i in 1:n){
  Yi[1:Ni[i]] <- y_i ## the first n_i elements of the i-th row of Yi contain 
  ## the vector of threshold exceedances y_i
}

datPP$y <- cbind(Ni, u, Yi)
```