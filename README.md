# multgam: automatic smoothing for multiple GAMs
The Rcpp package `multgam` implements the double statistical optimization algorithm described in El-Bachir and Davison (2019), which automatically fits and smooths multiple generalized additive models (GAMs). This uses R as an interface to the optimization code implemented in C++, and uses the R package `mgcv` to set up the matrix of inputs and to visualize the learned smooth functions and perform predictions.

## 1. Installation
The package `multgam` must be installed from source as follows.
- Download the repository `multgam`.
- Run the file `./install.R`.

## 2. Usage

The package fits univariate and multivariate probability distributions whose parameters are represented by sums of unknown smooth functions to be learned. The (vector of) output variables are assumed to be independent.

### 2.1. Main function

Train a multiple generalized additive model using the `mtgam` method as follows
```R
fit <- mtgam(dat, L.formula, fmName="gauss", lambInit=NULL, betaInit=NULL, groupReg=NULL, 
             ListConvInfo=list("iterMax"=200, "progressPen"=FALSE, "PenTol"=.Machine$double.eps^.5, "progressML"=FALSE, "MLTol"=1e-07), ...)
``` 
with arguments:
- dat: a data frame whose columns contain the input and the output variables. This must conform to the documentation in `mgcv`,
- L.formula: a list of formulae linking the outputs to the input variables. Each output variable must have an additive structure with smooth functions of inputs. The argument `L.formula` is supplied to the package `mgcv`, so this must conform to the documentation in `mgcv`



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
- fmName: name of the probability distribution
- lambInit: 
- betaInit: 
- ListConvInfo: 

The additive structure must be in the form of equation (1) in the paper. 


- in case you're interested in spatial analysis, the tensor product family in mgcv is not (yet) supported by the optimization since the M-step does not have an analytical solution.


### 2.2. Supported distributions
#### 2.2.1. Classical exponential family distributions
#### 2.2.2. Extreme value distribution families

### 2.3. Example distributions
Several examples can be found in the subdirectory `./simulation_paper/Multgam`, which reproduces Section 3 of the paper.


### 2.3. Plots

## 3. Extension to new distributions

## 4. General comments
The package is under development. 

## 5. Bugs
Bugs can be reported to yousra.elbachir@gmail.com by sending an email with:
- subject: multgam: bugs,
- content: a reproducible example and a simple description of the problem.

## 6. Citation
Acknowledge the use of `multgam` by citing the paper El-Bachir and Davison (2019).

## References
Yousra El-Bachir and Anthony C. Davison. Fast automatic smoothing for generalized additive models. *Journal of Machine Learning Research*, 20(173):1--27, 2019. Available at http://jmlr.org/beta/papers/v20/18-659.html.


