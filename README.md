# multgam
Rcpp package implementing automatic smoothing for multiple generalized additive models

# multgam : Automatic smoothing for multiple generalized additive models 
Rcpp package implementing the empirical Bayes method for automatic smoothing as described in the paper "Fast Automatic Smoothing for Generalized Additive Models"

## Installation
multgam must be installed from source as follows: 
1. Download the repository multgam.
2. Run install.R


## Usage and examples

## Convergence criteria
Convergence criteria for the multgam package are so strict that sometimes, the code throws an error about convergence failure even if this is not *really* true. This is because multgam favours accuracy over efficiency. This issue will be solved for the final release of the package. In the meantime and in convergence failure cases, rerun the computations with 

mtgam(..., ListConvInfo=list("iterMax"=500, "progressPen"=FALSE, "PenTol"=.Machine$double.eps^.5, "progressML"=FALSE, "MLTol"=1e-06), ...) # increases convergence tolerance for the EM optimization for the log-marginal likelihood

If the error remains, try again with  

mtgam(..., ListConvInfo=list("iterMax"=500, "progressPen"=FALSE, "PenTol"=1e-07, "progressML"=FALSE, "MLTol"=1e-06), ...) # increases convergence tolerance for the EM optimization for the log-marginal likelihood and the Newton--Raphson algorithm for the penalized log-likelihood

If the error remains, then convergence is indeed troublesome. Please report your bug with a reproducible example to yousra.elbachir@gmail.com

## Addition of new families of distributions
