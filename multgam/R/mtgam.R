#########################
#### Main function ######
#########################
# make sure that L.formula contains different covariate names. This is important for the prediction
mtgam <- function(dat, L.formula, fmName="gauss", lambInit=NULL, betaInit=NULL, groupReg=NULL, 
                  iterMax=200, progressPen=FALSE, PenTol=.Machine$double.eps^.5, progressML=FALSE, MLTol=1e-07, ...){
  # dat : dataframe input as for mgcv::gam
  # L.formula : list of fomulas as for mgcv::gam where y is the (matrix of) response variables
  # beta.init should be feasible
  # groupReg: list of vectors of how to regularize the parametric forms, i.e., one lambda for a group of beta or one lambda per beta? 
  #           default value is the same lambda for all parametric beta
  
  ###########################
  #### 1. Model set-up ######
  ###########################
  ListConvInfo <- list("iterMax"=iterMax, "progressPen"=progressPen, "PenTol"=PenTol, "progressML"=progressML, "MLTol"=MLTol)
  storage.mode(ListConvInfo$iterMax) <- "integer"
  
  if(length(L.formula)>1){
   
    G <- gam(L.formula, family=mvn(length(L.formula)), data=dat, fit=FALSE, ...)
    
  } else {
    
    G <- gam(L.formula[[1]], data=dat, fit=FALSE, ...)
  }
  
  rm(dat)
  
  TnS <- length(G$S) # total nb of smoothing matrices
  G$y <- as.matrix(G$y)
  G$m <- NULL; G$min.sp <- NULL; G$pearson.extra <- NULL; G$dev.extra <- NULL; G$n.true <- NULL; G$intercept <- NULL; G$off <- NULL; G$S <- NULL
  G$n.paraPen <- NULL; G$lsp0 <- NULL; G$n <- NULL; G$prior.weights <- G$w; G$w <- NULL; G$offset <- NULL; G$xlevels <- NULL; G$model <- G$mf
  G$mf <- NULL; G$call <- G$cl; G$cl <- NULL; G$am <- NULL
  
  if(is.list(G$formula)){ 
    attr(G$formula, "lpi") <- attr(G$X, "lpi") 
  }
  attr(G$pred.formula, "full") <- reformulate(all.vars(G$terms))
  
  ##################################
  #### 2. Optimization set-up ######
  ##################################
  
  ## generalities: pFull
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
  
  ## parametric forms: pPar, sPar, sRank
  nPar <- 0
  if(is.null(groupReg)){ ## all the parametric forms share the same regularization param
    
    pPar <- matrix(NA, nrow=2, ncol=D) ## indices of beta corresponding to the non-smooth inputs
    for(i in 1:D){
      l <- length(G$assign[[i]])
      if(l>1){ ## there are parametric terms in addition to the intercept
        nPar <- nPar+1
        pPar[1,nPar] <- pFull[1,i] + 1
        pPar[2,nPar] <- l-1
        }
    }
    
    if( (nPar>0) & (nPar<D) ){
      pPar <- as.matrix(pPar[,1:nPar])
      }
    
  } else { ## the parametric forms are regularized according to the user's choice
    L <- length(unlist(groupReg))
    pPar <- matrix(NA, nrow=2, ncol=L) ## indices of beta corresponding to the non-smooth inputs
    for(i in 1:D){
      
      l <- groupReg[[i]][1]
      if(l>0){
        nPar <- nPar+1
        pPar[1,nPar] <- pFull[1,i] + 1
        pPar[2,nPar] <- l
          
        nGr <- length(groupReg[[i]]) 
        if(nGr>1){
          
          for(j in 2:nGr){
            nPar <- nPar+1
            pPar[1,nPar] <- pPar[1,nPar-1] + pPar[2,nPar-1] 
            pPar[2,nPar] <- groupReg[[i]][j]
            }
        }
        
      }
    }
    
    if( (nPar>0) & (nPar<L) ){
      pPar <- as.matrix(pPar[,1:nPar])
      }
    }
  
  if(nPar>0){
    
    sPar <- vector("list", nPar)
    sRank <- vector("numeric", nPar)
    for(i in 1:nPar){
      sPar[[i]] <- diag(pPar[2,i])
      sRank[i] <- pPar[2,i]
    } 
  }
  

  ## smooth forms
  # extract smooths, ranks and indices of the beta corresponding to the smoothing matrices
  if(TnS>0){
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
  } else {
    
    ncBlock <- NA
  }
  storage.mode(ncBlock) <- "integer"

  
  ## group parametric forms and smooth forms: adjust pS, SList, G$rank and TnS
  if( (nPar>0) & (TnS>0) ){ ## if there are parametric and smooth forms 
    
    ## 1. augment pS, SList, G$rank with the corresponding parametric forms
    pS <- cbind(pPar, pS)
    TnS <- ncol(pS)
    SList <- c(sPar, SList)
    G$rank <- c(sRank, G$rank)
    
    ## 2. adjust pS, SList, G$rank and TnS to the full model matrix, i.e. parametric and smooth terms
    i <- order(pS[1,]) ## order((pS[1,])) indexes of the sorted values in the original vector pS[1,]
    pS <- pS[,i] 
    SList <- SList[i]
    G$rank <- G$rank[i]
    
  } else if( (nPar>0) & (TnS==0) ){ ## if there are parametric but no smooth forms 
    
    ## 1. adjust pS, SList, G$rank with the corresponding parametric forms
    pS <- pPar
    TnS <- ncol(pS)
    SList <- sPar
    G$rank <- sRank
    
  } else if( (nPar==0) & (TnS==0) ){ ## if there are no parametric and no smooth forms 
    stop("This package deals with L2 regularization, so there must be parametric or smooth terms to penalize")
  }
  
  storage.mode(pS) <- "integer"
  
  #############################
  #### 2. Initialization ######
  #############################
  ## family initialization
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
              
            } else if(fmName == "gpd") {
              
              fmName <- 7
              
              } else if(fmName == "pp") {
                
                fmName <- 8
                
              } else if(fmName == "rgev") {
                
                if(ncol(G$y) == 1){
                    fmName <- 0
                  } else {
                    fmName <- 9
                }
                 
              } else if(fmName == "newFam"){
                
                fmName <- 10
                
                if(is.null(betaInit)){
                  
                  stop("Provide a ", D, " dimensional vector with initial values for the functional params")
                  
                  } else {
                    
                    betaInit2 <- betaInit[1]
                    
                    for(i in 1:D){
                      
                      betaInit2 <- c(betaInit2, rep(0, pFull[2,i]-1))
                      
                      }
                    
                    betaInit <- betaInit2
                    }
            
              } else {
                
                stop("Distribution family not supported! Check out the help page at https://github.com/yelbachir/multgam to add a new family")
                
                }
  
  ## initialization for parameters
  if(is.null(lambInit)){
    lambInit <- rep(-1, TnS)
  }
  
  if(is.null(betaInit)){
    betaInit <- rep(-1000000, pp)
  }

  #######################
  #### 3. c++ code ######
  #######################
  fit <- mtgamcpp(betaInit, lambInit, SList, G$X, G$y, pFull, pS, G$rank, ncBlock, fmName, ListConvInfo)
  names(fit$coefficients) <- G$term.names
  names(fit$edf) <- G$term.names
  G$term.names <- NULL
  G$rank <- sum(fit$bdrop)
  ##names(fit$sp) <- names(G$sp) ## true when there are only smooth functions!
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
# need to add
#"Vc", "edf2","edf1", "deviance"