#########################
#### Main function ######
#########################
# make sure that L.formula contains different covariate names. This is important for the prediction
mtgam <- function(dat, L.formula, fmName="gev", lambInit=NULL, betaInit=NULL, ListConvInfo=list("iterMax"=500, "progressPen"=FALSE, "PenTol"=.Machine$double.eps^.5, "progressML"=FALSE, "MLTol"=1e-07), ...){
  # dat : dataframe input as for gam
  # L.formula : list of fomulas as for gam where y is the original dataset
  # beta.init should be feasible
  
  if(length(ListConvInfo)!=5){
    stop("Check how ListConvInfo should be defined")
  }
  storage.mode(ListConvInfo$iterMax) <- "integer"
  
  if(length(L.formula)>1){
   
    G <- gam(L.formula, family=mvn(length(L.formula)), data=dat, fit=FALSE, ...)
    
  } else {
    
    G <- gam(L.formula[[1]], data=dat, fit=FALSE, ...)
    
  }
  
  rm(dat)
  
  TnS <- length(G$S) # total nb of smoothing matrices
  if(TnS==0){
    stop("There is no smooth terms in your model!")
  }
  
  
  G$m <- NULL; G$min.sp <- NULL; G$pearson.extra <- NULL; G$dev.extra <- NULL; G$n.true <- NULL; G$intercept <- NULL; G$off <- NULL; G$S <- NULL
  G$n.paraPen <- NULL; G$lsp0 <- NULL; G$n <- NULL; G$prior.weights <- G$w; G$w <- NULL; G$offset <- NULL; G$xlevels <- NULL; G$model <- G$mf
  G$mf <- NULL; G$call <- G$cl; G$cl <- NULL; G$am <- NULL
  
  if(is.list(G$formula)){ 
    attr(G$formula, "lpi") <- attr(G$X, "lpi") 
  }
  attr(G$pred.formula, "full") <- reformulate(all.vars(G$terms))
  
  i <- which(G$cmX==1)
  D <- length(i)
  pFull <- matrix(NA, nrow=2, ncol=D)
  pFull[1,] <- i-1
  pp <- ncol(G$X)
  
  if(D>1){
    dm1 <- D-1
    for(d in 1:dm1){
      pFull[2,d] <- i[d+1]-i[d]
    }
    pFull[2,D] <- pp-i[D]+1
  } else if(D==1) {
    
    pFull[2,D] <- pp
    
  } else {
    stop("D cannot be <= 0")
  }
  
  storage.mode(pFull) <- "integer"
  
  # family initialization
  if(fmName == "gev"){
    
    fmName <- 0
    
    } else if(fmName == "gauss"){
    
        fmName <- 1
        
        } else if(fmName == "anglogit") {
          
          fmName <- 2
          
        } else if(fmName == "poisson") {
            
          fmName <- 3
          
          storage.mode(G$y) <- "double"
          
          } else if(fmName == "gamma") {
            
            fmName <- 4
            
            } else if(fmName == "binom") {
              
              fmName <- 5
              
              storage.mode(G$y) <- "double"
              
            } else if(fmName == "expon") {
              
              fmName <- 6
              
            } else if (fmName == "newFam"){
            
            fmName <- 7
            
            if(is.null(betaInit)){
              stop("Provide a ", D, " dimensional vector with initial values for the unpenalized params")
              } else {
                betaInit2 <- betaInit[1]
                for(i in 1:D){
                  betaInit2 <- c(betaInit2, rep(0, pFull[2,i]-1))
                  }
                betaInit <- betaInit2
                }
            } else {
              stop("Distribution family not supported!")
              }
  
  # extract smooths, ranks and indices of the 1st beta's corresponding to the smoothing matrices
  SList <- vector("list", TnS)
  pS <- matrix(NA, nrow=2, ncol=TnS)
  nB <- length(G$smooth) # nb of blocks
  ncBlock <- matrix(NA, nrow=2, ncol=nB)
  l <- 0
  s <- 0
  
  for(j in 1:nB){ # each block
    
    ncBlock[1,j] <- s
    nS <- length(G$smooth[[j]]$S) # G$smooth[[j]]$dim
    ncBlock[2,j] <- nS
    s <- s + nS
    
    for(i in 1:nS){ # inside a block
      
      SList[[i+l]] <- G$smooth[[j]]$S[[i]]
      pS[1,i+l] <- G$smooth[[j]]$first.para-1
      pS[2,i+l] <- (G$smooth[[j]]$last.para-G$smooth[[j]]$first.para)+1
    }
    l <- l + nS
  }
  storage.mode(pS) <- "integer"
  storage.mode(ncBlock) <- "integer"
  
  # initialization for lambda
  if(is.null(lambInit)){
    lambInit <- rep(-1, TnS)
  }
  
  if(is.null(betaInit)){
    betaInit <- rep(-1000000, pp)
  }
  
  ##  y <- matrix(y) # no need for this. It is received by c++ as a matrix
  
  # call c++ code
  fit <- mtgamcpp(betaInit, lambInit, SList, G$X, G$y, pFull, pS, G$rank, ncBlock, fmName, ListConvInfo)
  names(fit$coefficients) <- G$term.names
  names(fit$edf) <- G$term.names
  G$term.names <- NULL
  G$rank <- sum(!fit$bdrop)
  names(fit$sp) <- names(G$sp) 
  G$sp <- fit$sp
  fit$sp <- NULL
  G <- c(G, fit)
  
  # outputs useful for prediction and plotting
  G$scale <- 1
  G$scale.estimated <- FALSE
  G$sig2 <- 1
  G$na.action <- attr(G$model, "na.action")
  G$df.residual <- nrow(G$X) - sum(G$edf)
  
  class(G) <- "gam"
  environment(G$formula) <- environment(G$pred.formula) <- environment(G$terms) <- environment(G$pterms) <- .GlobalEnv

  return(G)
}