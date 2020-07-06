setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) ## set working directory to the location of the running file 
setwd("..")

library(Rcpp)

compileAttributes("multgam")

install.packages("multgam", repos=NULL, type="source")
