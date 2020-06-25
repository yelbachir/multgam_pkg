# multgam: automatic smoothing for multiple GAMs
The Rcpp package `multgam` implements the empirical Bayes optimization algorithm described in El-Bachir and Davison (2019), which trains multiple generalized additive models (GAMs) and automatically tunes their L2 regularization hyper-parameters. In particular, `multgam` also provides automatic L2 regularization (ridge penalty) for multiple parametric non-linear regression, i.e., non-smooth functions of inputs. The package `multgam` uses R as an interface to the optimization code implemented in C++, and uses the R package `mgcv` to set up the matrix of inputs and to visualize the learned smooth functions and perform predictions. As a toy example, `multgam` trains models with the following structure:

############ check this
Y_i ~ F(\mu_i, \tau_i), where the Y_i are random (vector of) variables generated from a probability distribution F with parameters \mu_i and \tau_i such that:
\mu_i = \beta_{10} + f_{11}(x_{i1}) + ... f_{1p}(x_p) or eventually \mu_i = \beta_0 + \beta_1 w_1 + ... + \beta_r w_r + f_2(x_2) + ... f_p(x_p)
\tau_i = \beta_{20} + f_{21}(z_{i1}) + ... f_q(z_q), 
such that the regression coefficents of the f_j and those of the w_j are subject to the L2 penalty. 
################

## Table of content


## 1. Installation
The package `multgam` must be installed from source as follows.
- Download the repository `multgam`.
- Run the file `./install.R`.

## 2. Usage

The package trains univariate and multivariate probability distributions whose parameters are represented by sums of unknown smooth functions to be learned. The log-likelihood of the vector of output variables should be expressed as the sum of the contribution of the individual output variables, a particular case is independent random variables. In practice, `multgam` interprets a GAM as a multiple non-linear regression model whose coefficients are subject to the L2 penalty. In the case of smooth functions, the regularization matrices are dense and represent the smoothing matrices (computed by the software). In the case of non-smooth functions, the regularization matrices are the identity matrices to which the user can assign different regularization hyper-parameters for different non-smooth functions; see the argument `groupReg` in the main function `mtgam` below. 

### 2.1. Main function

Train a multiple generalized additive model using the `mtgam` method as follows
```R
fit <- mtgam(dat, L.formula, fmName="gauss", lambInit=NULL, betaInit=NULL, groupReg=NULL, 
             ListConvInfo=list("iterMax"=200, "progressPen"=FALSE, "PenTol"=.Machine$double.eps^.5, "progressML"=FALSE, "MLTol"=1e-07), ...)
``` 
with arguments:
- `dat`: a list or a data frame whose columns contain the input and the output variables used in `L.formula`,
- `L.formula`: a list of as many formulae as there are output variables having additive structures linking the input variables,
- `fmName`: a character variable for the name of the probability distribution of the output variables, further details can be found in Section 2.2.,
- `lambInit`: vector of starting values for the L2 regularization hyper-parameters,
- `betaInit`: vector of starting values for the regression coefficients,
- `groupReg`: list of as many vectors as there are non-smooth functions in the parametric multiple regression model. Each element of this list should be a vector or a scalar (indicating the intercept) contains as many values as
of how to regularize the parametric forms, i.e., one lambda for a group of beta or one lambda per beta? default value is the same hyper-parameter for all parametric regression coefficients,
- `ListConvInfo$iterMax`: number of maximal iterations for the optimization of the log-marginal likelihood and the penalized log-likelihood,
- `ListConvInfo$progressPen`: if `TRUE`, information about the progress of maximization of the penalized log-likelihood will be printed,
- `ListConvInfo$PenTol`: tolerance for the maximization of the penalized log-likelihood, 
- `ListConvInfo$progressM`: if TRUE, information about the progress of the maximization of the log-marginal likelihood will be printed, 
- `ListConvInfo$MLTol`: tolerance for the maximization of the log-marginal likelihood for the L2 regularization hyper-parameters,
- ....: additional arguments to supply to the function `gam()` in `mgcv`.
For additional information on `dat` and `L.formula` see the examples below, or the documentation for the R package `mgcv` in CRAN.

The output `fit` of the function `mtgam` can be used as if it were computed from the function `gam` in `mgcv`. This includes plots, predictions, etc...


. For example: 
```R
n <- 1000
dat <- data.frame(y1=runif(n), y2=runif(n), x1=runif(n), x2=runif(n), x3=runif(n)) ## y1 and y2 are the outputs and x1, x2 and x3 are the inputs
```
- L.formula: a list of formulae linking the output to the input variables. Each output variable must have an additive structure with smooth functions of inputs. The argument `L.formula` is supplied to the package `mgcv`, so this must conform to the documentation in `mgcv`. For example if ($y_1$, $y_2$) is a random vector following a bivariate distribution such that  whose  and : 
```R
k <- 20 ## dimension of the basis function
L.formula <- list(y1 ~ s(x1, bs="cr", k=k), ## cr is the cubic regression spline family of basis functions
                  ~ s(x2, bs="cc", k=k) + s(x3, bs="tp", k=k), ## tp is the thin plate regression spline
                  y2 ~ s(x4, bs="cr", k=k),
                  ~ s(x5, bs="cr", k=k)
                  )
```             

The additive structure must be in the form of expression (1) in the paper.


### 2.2. Supported distributions
#### 2.2.1. Classical exponential family distributions
#### 2.2.2. Extreme value distribution families
Likelihood definition
Simulation
Return levels

### 2.3. Example distributions
Several examples can be found in the subdirectory `./simulation_paper/Multgam`, which reproduces Section 3 of the paper.

## 3. Extension to new distributions

## 4. General comments
The package is under development. For 

## 5. Bugs
Bugs can be reported to the maintainer at yousra.elbachir@gmail.com by sending an email with:
- subject: multgam: bugs,
- content: a reproducible example and a simple description of the problem.

## 6. Citation
Acknowledge the use of `multgam` by citing the paper El-Bachir and Davison (2019).

## References
Yousra El-Bachir and Anthony C. Davison. Fast automatic smoothing for generalized additive models. *Journal of Machine Learning Research*, 20(173):1--27, 2019. Available at http://jmlr.org/beta/papers/v20/18-659.html.


