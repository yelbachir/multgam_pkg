# multgam: automatic smoothing for multiple GAMs
The Rcpp package `multgam` implements the empirical Bayes optimization algorithm described in El-Bachir and Davison (2019), which trains multiple generalized additive models (GAMs) and automatically tunes their L2 regularization hyper-parameters. In particular, `multgam` also provides automatic L2 regularization (ridge penalty) for multiple parametric non-linear regression models, i.e., the linear or non-linear functions of inputs are not necessarily smooth but their regression weights are constrained by the L2 penalty with possibly different hyper-parameters.

The package `multgam` uses R as an interface for the optimization code implemented in C++, and uses the R package `mgcv` to set up the matrix of inputs and to visualize the learned functions and perform predictions.

## Table of content
############ TODO

## 1. Installation
The package `multgam` must be installed from source as follows.
- Download the repository `multgam`.
- Run the file `./install.R`.

## 2. Usage

The output variable can be a vector or a matrix from a univariate or a multivariate probability distribution, but the log-likelihood for the full dataset must be expressed as the sum of the log-likelihoods for an individual observation. A particular case is independent random observations. 

In practice, `multgam` interprets a GAM as a multiple linear regression model whose weights are subject to the L2 penalty. When the functions of inputs are smooth, the regularization matrices are dense and represent the smoothing matrices, which are computed by the package. When the functions of inputs are weighted sums, the regularization matrices are the identity matrices, to which the user can assign different regularization hyper-parameters; see the argument `groupReg` in the function `mtgam` in Section 2.1. 

### 2.1. Main function

Train a multiple generalized additive model using the function `mtgam` as follows
```R
fit <- mtgam(dat, L.formula, fmName="gauss", lambInit=NULL, betaInit=NULL, groupReg=NULL, 
             ListConvInfo=list("iterMax"=200, "progressPen"=FALSE, "PenTol"=.Machine$double.eps^.5, 
                               "progressML"=FALSE, "MLTol"=1e-07), ...)
``` 
with **arguments**:
- `dat`: a list or a data frame whose columns contain the input and the output variables used in `L.formula`; family specific considerations can be found in Section 2.2.,
- `L.formula`: a list of as many formulae as there are output variables having additive structures linking the input variables,
- `fmName`: a character variable for the name of the probability distribution of the output variables: `"gauss"` for the Gaussian distribution, `"poisson"` for the Poisson distribution, `"binom"` for the binomial distribution, `"expon"` for the exponential distribution, `"gamma"` for the gamma distribution, `"gev"` for the generalized extreme value distribution, `"gpd"` for the generalized Pareto distribution, `"pp"` for the point process approach in extreme value analysis, `"rgev"` for the r-largest extreme value distribution. Details on their parametrization and specific considerations can be found in Section 2.2.,
- `lambInit`: a vector of starting values for the L2 regularization hyper-parameters. This should contain as many values as non-zero elements supplied to `groupReg`, in addition to the number of smooth functions. Default values are provided,
- `betaInit`: a vector of starting values for the regression weights. Default values are provided,
- `groupReg`: a list of length `L.formula` which indicates how to regularize the regression weights of the input variables in the multiple parametric regression models described in each formula of `L.formula`. Each element of `groupReg` is a vector associated to a formula, and contains the numbers of successive input variables in that formula whose regression weights share the same hyper-parameter. The value `0` in place of a vector indicates that the regression weight corresponding to that input variable is an offset, and so should not be penalized. If `NULL`: the regression weights of a smooth function of inputs share the same hyper-parameter, but different smooth functions have different hyper-parameters, and all the remaining non-smooth functions share the same hyper-parameter. 
For example, if we have `L.formula <- list(y ~ x1 + x2 + x3 + s(x1) + s(x2), ~ 1)`, the argument `groupReg=NULL` would correspond to one hyper-parameter associated with the regression weights of the triple `(x1, x2, x3)`, one hyper-parameter for `s(x1)`, one hyper-parameter for `s(x2)` and no hyper-parameters on the offset. However, if the regression weight for the input variable `x1` is constrained by an L2 penalty, and `x2` and `x3` share the same hyper-parameter, then the corresponding argument should be `groupReg <- list(c(1, 2), 0)`, where `1` corresponds to having one hyper-parameter on the regression weight of `x1`, `2` to having one hyper-parameter on the pair `(x2, x3)`, and `0` indicates that `1` is the offset of the output variable in the second formula in `L.fomrula`,
- `ListConvInfo$iterMax`: an integer for the number of maximal iterations in the optimization of the log-marginal likelihood and the penalized log-likelihood,
- `ListConvInfo$progressPen`: if `TRUE`, information about the progress of the penalized log-likelihood maximization will be printed,
- `ListConvInfo$PenTol`: the tolerance in the maximization of the penalized log-likelihood, 
- `ListConvInfo$progressM`: if `TRUE`, information about the progress of the log-marginal likelihood maximization will be printed, 
- `ListConvInfo$MLTol`: the tolerance in the maximization of the log-marginal likelihood,
- `....`: additional arguments to supply to the function `gam()` in `mgcv`.
For additional information on `dat` and `L.formula` see the examples in Section 2.2., or the documentation of the R package `mgcv` on CRAN.

The **outputs** contained in the variable `fit` resulting from `mtgam` can be used as if `fit` were computed from the function `gam()` in `mgcv`. This can be used for plots, predictions, etc... In particular, the vector `sp` in `gam()` corresponds to the hyper-parameters for the smooth functions only, whereas `mtgam` provides an additional output vector called `reg`, which contains the values of all the hyper-parameters including those described by the non-zero values in `groupReg`. Following the example given in `groupReg` above, if we have `L.formula <- list(y ~ x1 + x2 + x3 + s(x1) + s(x2), ~ 1)` and `groupReg=NULL`, then `fit$reg` would be `(lamb1, lamb2, lamb3)`, where `lamb1` would be the hyper-parameter corresponding to the regression weights for `(x1, x2, x3)`, and `lamb2` would be associated to the regression weights of `s(x1)` and `lamb3` to `s(x2)`. If `groupReg <- list(c(1, 2), 0)` then `fit$reg` would be `(lamb1, lamb2, lamb3, lamb4)`, where `lamb1` would be the hyper-parameter corresponding to the regression weight for `x1`, `lamb2` to `(x2, x3)`, `lamb3` to `s(x1)` and `lamb4` to `s(x2)`.

### 2.2. Supported distributions and examples
The function `mtgam` trains probability distributions with functional parameters that have a parametrization which does not constrain the parameters range values. 

#### 2.2.1. Classical exponential family distributions
- Gaussian distribution: `fmName="gauss"` implements `N(mu, tau)`, where `mu` is the mean and `tau` is `2 log(sigma)`, `sigma` being the standard deviation,
- Poisson distribution: `fmName="poisson"` implements `Poiss(mu)`, where `mu` is the log-rate,
- Exponential distribution: `fmName="expon"` implements `Expon(mu)`, where `mu` is the log-rate,
- Gamma distribution: `fmName="gamma"` implements `Gamma(mu, tau)`, where `mu` is the log-shape and `tau` is `-log(sigma)`, `sigma` being the scale,
- Binomial distribution: `fmName="binom"` implements `Binom(mu)`, where `mu` is the logit, i.e., `log(p/(1-p))` with `p` the probability of succeess. 


Several examples can be found in the subdirectory `./simulation_paper/Multgam`, which reproduces Section 3 of the paper.

```R
n <- 1000
dat <- data.frame(y1=runif(n), y2=runif(n), x1=runif(n), x2=runif(n), x3=runif(n)) ## y1 and y2 are the outputs and x1, x2 and x3 are the inputs

k <- 20 ## dimension of the basis function
L.formula <- list(y1 ~ s(x1, bs="cr", k=k), ## cr is the cubic regression spline family of basis functions
                  ~ s(x2, bs="cc", k=k) + s(x3, bs="tp", k=k), ## tp is the thin plate regression spline
                  y2 ~ s(x4, bs="cr", k=k),
                  ~ s(x5, bs="cr", k=k)
                  )
```             

#### 2.2.2. Extreme value distribution families
- Generalized extreme value distribution: `fmName="gev"` implements `GEV(mu, tau, xi)`, where `mu` is the location, `tau` is the log-scale and `xi` is the shape,
- Generalized Pareto distribution: `fmName="gpd"` implements `GPD(mu, tau)`, where `tau` is the log-scale and `xi` is the shape,
- Point process approach in extreme value analysis: `fmName="pp"` implements `PP(mu, tau, xi)`, where `mu` is the location, `tau` is the log-scale and `xi` is the shape,
- r-Largest extreme value distribution: `fmName="rgev"` implements `rGEV(mu, tau, xi)`, where `mu` is the location, `tau` is the log-scale and `xi` is the shape.

The families `gev`, `gpd`, `rgev` provide simulation code that

Simulation
Return levels

### 2.3. Extension to new distributions

## 3. General comments
The package is under development. For 
Convergence criteria are conservative

## 4. Bugs, clarifications and suggestions
Bugs can be reported to the maintainer at yousra.elbachir@gmail.com by sending an email with:
- subject: multgam: bugs,
- content: a reproducible example and a simple description of the problem.

Further details on how to use the package or suggestions for additional extensions can be requested to the maintainer.

## 5. Citation
Acknowledge the use of `multgam` by citing the paper El-Bachir and Davison (2019).

## References
Yousra El-Bachir and Anthony C. Davison. Fast automatic smoothing for generalized additive models. *Journal of Machine Learning Research*, 20(173):1--27, 2019. Available at http://jmlr.org/beta/papers/v20/18-659.html.


