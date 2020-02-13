# multgam: automatic smoothing for multiple GAMs
The Rcpp package `multgam` implements the double statistical optimization method for automatically smoothing multiple generalized additive models as described in El-Bachir and Davison (2019). This uses R as an interface to the optimization code implemented in C++, and uses the R package `mgcv` to set up the inputs and to visualize the outputs.

## 1. Installation
The package `multgam` must be installed from source as follows.
- Download the repository `multgam`.
- Run the file `./install.R`.

## 2. Usage
### 2.1. Main function

Fit the model using
```R
mtgam(dat, L.formula, fmName="gev", lambInit=NULL, betaInit=NULL, ListConvInfo=list("iterMax"=500, "progressPen"=FALSE, "PenTol"=.Machine$double.eps^.5, "progressML"=FALSE, "MLTol"=1e-07), ...)
```

- in case you're interested in spatial analysis, the tensor product family in mgcv is not (yet) supported by the optimization since the M-step does not have an analytical solution.
### 2.2. Supported distributions

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


