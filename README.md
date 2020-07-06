# multgam: automatic smoothing for multiple GAMs
The Rcpp package `multgam` implements the empirical Bayes optimization algorithm described in El-Bachir and Davison (2019), which trains multiple generalized additive models (GAMs) and automatically tunes their L2 regularization hyper-parameters. Moreover, `multgam` provides automatic L2 regularization (ridge penalty) for multiple parametric non-linear regression models, where the linear or non-linear functions of inputs are not necessarily smooth but their regression weights are constrained by the L2 penalty with possibly different hyper-parameters.

The package `multgam` uses R as an interface for the optimization code implemented in C++, and uses the R package `mgcv` to set up the matrix of inputs, to visualize the learned functions, and to perform predictions.

## Table of contents
[1. Installation](#install)                                     
[2. Usage](#usage)                    
    <pre><code> [2.1. Main function](#mainFunc)         
    [2.2. Supported probability distributions and examples](#supportedDistrib)          
          [2.2.1. Classical exponential family distributions](#classical)         
          [2.2.2. Extreme value distribution families](#evd)          
           [2.2.3. Examples](#examples)          
    [2.3. Extension to new distributions](#newDistrib)          
[3. General comments](#comments)          
[4. Bugs, help and suggestions](#bugs)          
[5. Citation](#citation)

<a name="install"/></a>
## 1. Installation
The package `multgam` must be installed from source as follows.
- Download the repository `multgam`.
- Run the file `./install.R`.

<a name="usage"/></a>
## 2. Usage

The output variable can be a vector or a matrix from a univariate or a multivariate probability distribution, but the log-likelihood for the full dataset must be expressed as the sum of the log-likelihoods for an individual observation. A particular case is independent random observations. 

In practice, `multgam` interprets a GAM as a multiple linear regression model whose weights are subject to the L2 penalty. When the functions of inputs are smooth, the regularization matrices are dense and represent the smoothing matrices, which are computed by the package. When the functions of inputs are weighted sums, the regularization matrices are the identity matrices, to which the user can assign different regularization hyper-parameters; see the argument `groupReg` in the function `mtgam` in Section 2.1. 

<a name="mainFunc"/></a>
### 2.1. Main function

Train a multiple generalized additive model using the function `mtgam` as follows
```R
fit <- mtgam(dat, L.formula, fmName="gauss", lambInit=NULL, betaInit=NULL, groupReg=NULL, 
             iterMax=200, progressPen=FALSE, PenTol=.Machine$double.eps^.5, progressML=FALSE, MLTol=1e-07, ...)
``` 
with **arguments**:
- `dat`: a list or a data frame whose columns contain the input and the output variables used in `L.formula`; family specific considerations can be found in Section 2.2.,
- `L.formula`: a list of as many formulae as there are output variables having additive structures linking the input variables,
- `fmName`: a character variable for the name of the probability distribution of the output variables: `"gauss"` for the Gaussian distribution, `"poisson"` for the Poisson distribution, `"binom"` for the binomial distribution, `"expon"` for the exponential distribution, `"gamma"` for the gamma distribution, `"gev"` for the generalized extreme value distribution, `"gpd"` for the generalized Pareto distribution, `"pp"` for the point process approach in extreme value analysis, `"rgev"` for the r-largest extreme value distribution. Details on their parametrization and specific considerations can be found in Section 2.2.,
- `lambInit`: a vector of starting values for the L2 regularization hyper-parameters. This should contain as many values as non-zero elements supplied to `groupReg`, in addition to the number of smooth functions. Default values are provided,
- `betaInit`: a vector of starting values for the regression weights. Default values are provided,
- `groupReg`: a list of length `L.formula` which indicates how to regularize the regression weights of the input variables in the multiple parametric regression models described in each formula of `L.formula`. Each element of `groupReg` is a vector associated to a formula, and contains the numbers of successive input variables in that formula whose regression weights share the same hyper-parameter. The value `0` in place of a vector indicates that the regression weight corresponding to that input variable is an offset, and so should not be penalized. If `NULL`: the regression weights of a smooth function of inputs share the same hyper-parameter, but different smooth functions have different hyper-parameters, and all the remaining non-smooth functions share the same hyper-parameter. 
For example, if we have `L.formula <- list(y ~ x1 + x2 + x3 + s(x1) + s(x2), ~ 1)`, the argument `groupReg=NULL` would correspond to one hyper-parameter associated with the regression weights of the triple `(x1, x2, x3)`, one hyper-parameter for `s(x1)`, one hyper-parameter for `s(x2)` and no hyper-parameters on the offset. However, if the regression weight for the input variable `x1` is constrained by an L2 penalty, and `x2` and `x3` share the same hyper-parameter, then the corresponding argument should be `groupReg <- list(c(1, 2), 0)`, where `1` corresponds to having one hyper-parameter on the regression weight of `x1`, `2` to having one hyper-parameter on the pair `(x2, x3)`, and `0` indicates that `1` is the offset of the output variable in the second formula in `L.fomrula`,
- `iterMax`: an integer for the number of maximal iterations in the optimization of the log-marginal likelihood and the penalized log-likelihood,
- `progressPen`: if `TRUE`, information about the progress of the penalized log-likelihood maximization will be printed,
- `PenTol`: the tolerance in the maximization of the penalized log-likelihood, 
- `progressML`: if `TRUE`, information about the progress of the log-marginal likelihood maximization will be printed, 
- `MLTol`: the tolerance in the maximization of the log-marginal likelihood,
- `....`: additional arguments to supply to the function `gam()` in `mgcv`.
For additional information on `dat` and `L.formula` see the examples in Section 2.2., or the documentation of the R package `mgcv` on CRAN.

The **outputs** contained in the variable `fit` resulting from `mtgam` can be used as if `fit` were computed from the function `gam()` in `mgcv`. This can be used for plots, predictions, etc... In particular, the vector `sp` in `gam()` corresponds to the hyper-parameters for the smooth functions only, whereas in `mtgam`, `sp` contains the values of all the hyper-parameters including those described by the non-zero values in `groupReg`. Following the example given in `groupReg` above, if we have `L.formula <- list(y ~ x1 + x2 + x3 + s(x1) + s(x2), ~ 1)` and `groupReg=NULL`, then `fit$reg` would be `(lamb1, lamb2, lamb3)`, where `lamb1` would be the hyper-parameter corresponding to the regression weights for `(x1, x2, x3)`, and `lamb2` would be associated to the regression weights of `s(x1)` and `lamb3` to `s(x2)`. If `groupReg <- list(c(1, 2), 0)` then `fit$reg` would be `(lamb1, lamb2, lamb3, lamb4)`, where `lamb1` would be the hyper-parameter corresponding to the regression weight for `x1`, `lamb2` to `(x2, x3)`, `lamb3` to `s(x1)` and `lamb4` to `s(x2)`. Further details can be found at point 1 of Section 2.2.3.

<a name="supportedDistrib"/></a>
### 2.2. Supported probability distributions and examples
The function `mtgam` trains probability distributions with functional parameters whose parametrization does not constrain the parameters range values. 

<a name="classical"/></a>
#### 2.2.1. Classical exponential family distributions
- Gaussian distribution: `fmName="gauss"` implements `N(mu, tau)`, where `mu` is the mean and `tau` is `2 log(sigma)`, `sigma` being the standard deviation,
- Poisson distribution: `fmName="poisson"` implements `Poiss(mu)`, where `mu` is the log-rate,
- Exponential distribution: `fmName="expon"` implements `Expon(mu)`, where `mu` is the log-rate,
- Gamma distribution: `fmName="gamma"` implements `Gamma(mu, tau)`, where `mu` is the log-shape and `tau` is `-log(sigma)`, `sigma` being the scale,
- Binomial distribution: `fmName="binom"` implements `Binom(mu)`, where `mu` is the logit, i.e., `log(p/(1-p))` with `p` the probability of success.

<a name="evd"/></a>
#### 2.2.2. Extreme value distribution families
- Generalized extreme value distribution: `fmName="gev"` implements `GEV(mu, tau, xi)`, where `mu` is the location, `tau` is the log-scale and `xi` is the shape,
- Generalized Pareto distribution: `fmName="gpd"` implements `GPD(mu, tau)`, where `tau` is the log-scale and `xi` is the shape,
- Point process approach in extreme value analysis: `fmName="pp"` implements `PP(mu, tau, xi)`, where `mu` is the location, `tau` is the log-scale and `xi` is the shape. In the case `pp`, the output variable `y` (say) in the argument `dat` of the function `mtgam` should be a matrix of size `nx(N+2)`, where `n` is the sample size and `N` is the length of the largest block. The first column of the matrix `dat$y` should contain the vector of the `n` block sizes, the second column should be the vector of the `n` thresholds and the remaining columns should be filled with the threshold exceedances and `NA` values when the size `n_i` of the `i`-th block contains fewer exceedances than `N`, i.e., when `n_i<N`,  
- r-Largest extreme value distribution: `fmName="rgev"` implements `rGEV(mu, tau, xi)`, where `mu` is the location, `tau` is the log-scale and `xi` is the shape. In the case `rgev`, the output variable `y` (say) in the argument `dat` of the function `mtgam` should be a matrix of size `nxr`, where `n` is the sample size and `r` is the number of r largest extremal data per block of GEV. The values in each of the rows should be sorted in ascending order.

Data from the families `gev`, `gpd` and `rgev` can be simulated by the function
```R
simExtrem(mu=NULL, sigma=NULL, xi=NULL, r=NULL, family="gev")
```
with **arguments**:
- `mu`: a vector of location parameters for the full dataset,
- `sigma`: a vector of scale parameters for the full dataset,
- `xi`: a vector of shape parameters for the full dataset,
- `r`: an integer for the number of r largest extremal data per block of GEV data,
- `family`: a character variable which can take `"gev"`, `"gpd"` or `"rgev"`, 

and **output**: 
- if `family="gev"` or `family="gpd"`: a vector of length `mu`, which contains the generated data,
- if `family="rgev"`: a matrix of size `nxr`, where `n` is the length of `mu` and `r` is the number of r largest extremal data per block of GEV. The values in each of the rows are sorted in ascending order.


Return levels (quantiles) from the families `gev` and `gpd` can be computed by the function
```R
returnLevel(prob=NULL, mu=NULL, sigma=NULL, xi=NULL, family="gev")
```
with **arguments**:
- `prob`: a scalar for the probability for which the return level is computed,
- `mu`: a vector of location parameters for the full dataset,
- `sigma`: a vector of scale parameters for the full dataset,
- `xi`: a vector of shape parameters for the full dataset,
- `family`: a character variable which can take `"gev"` or `"gpd"`, 

and **output**:
- a vector of return levels corresponding to the probability `prob` and the functional parameters `mu`, `sigma` and `xi`.

<a name="examples"/></a>
#### 2.2.3. Examples: 
The following examples include:
  1. the usage of `groupReg` on the Gaussian model for example,
  2. the training of a multiple generalized additive models on the supported distributions,
  3. the definition of `dat` for the PP model (in pseudo-code).
  
These three steps can be found in `./examples.R`.

<a name="newDistrib"/></a>
### 2.3. Extension to new distributions
New families of distributions can be implemented by the user and added to `multgam`, but for a numerically stable implementation, it is preferable to contact the maintainer at yousra.elbachir@gmail.com who can do this for you.

<a name="comments"/></a>
## 3. General comments
- The package is under development.  
- The convergence criteria are conservative, if the training does not converge, increase `MLTol` to `1e-06` or `1e5`. If this still does not converge, please report the error to maintainer following Section 4. 

<a name="bugs"/></a>
## 4. Bugs, help and suggestions
Bugs can be reported to the maintainer at yousra.elbachir@gmail.com by sending an email with:
- subject: multgam: bugs,
- content: a reproducible example and a simple description of the problem.

Further details on how to use the package or suggestions for additional extensions can be requested to the maintainer.

<a name="citation"/></a>
## 5. Citation
Acknowledge the use of `multgam` by citing the paper El-Bachir and Davison (2019).

## References
Yousra El-Bachir and Anthony C. Davison. Fast automatic smoothing for generalized additive models. *Journal of Machine Learning Research*, 20(173):1--27, 2019. Available at http://jmlr.org/beta/papers/v20/18-659.html.


