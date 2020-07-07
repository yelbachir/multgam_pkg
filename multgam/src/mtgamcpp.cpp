/**************************************
 **************************************
 ******
 ******  Main c++ function called by R
 ******  Intermediate functions :
 ******  mtgamcpp
 ******
 **************************************
 **************************************/

#include <RcppEigen.h>

#include "../inst/include/MLOptim.hpp"

#include "../inst/include/commonVarFamily.hpp" // extra argument, i.e. n, to family binomial

using namespace Eigen; 

//============================================
// use compileAtrributes to get the RcppExport cpp and R versions when using
// [[Rcpp::export]]
SEXP mtgamcpp(SEXP betaInitR, SEXP lambInitR, SEXP SListR, SEXP XR, SEXP yR, SEXP pFullR, SEXP pSR, SEXP rankSR, SEXP ncBlockR, SEXP fmNameR, SEXP ListConvInfoR)
{
    const Map<VectorXd> betaInit(Rcpp::as<Map<VectorXd> >(betaInitR));
    const Map<VectorXd> lambInit(Rcpp::as<Map<VectorXd> >(lambInitR));
    const Map<MatrixXd> X(Rcpp::as<Map<MatrixXd> >(XR));
    const Map<MatrixXd> y(Rcpp::as<Map<MatrixXd> >(yR));
    const Map<MatrixXi> pFull(Rcpp::as<Map<MatrixXi> >(pFullR));
    const Map<MatrixXi> pS(Rcpp::as<Map<MatrixXi> >(pSR));
    MatrixXi ncBlock(Rcpp::as<Map<MatrixXi> >(ncBlockR)); // indices of non-converged blocks of smooths: each column contains (ind_lamb1, dim_block) where ind_lamb1 is the index w.r.t full lambda of the 1st lamb of the smooth function and dim_block is the nb of lamb in that block
    const Map<VectorXd> rankS(Rcpp::as<Map<VectorXd> >(rankSR));
    
    const int fmName(Rcpp::as<int>(fmNameR));
    const Rcpp::List ListConvInfo(ListConvInfoR);
    const convInfo conv({ListConvInfo["iterMax"], ListConvInfo["progressPen"], ListConvInfo["PenTol"], ListConvInfo["progressML"], ListConvInfo["MLTol"]});
    
    try {
		/***********************
		 *  initialization
         ***********************/
         
         // convert List of the R smoothing matrices to std::vector of Eigen::MatrixXd
         const Rcpp::List SList(SListR);
         const int q(SList.size());
         vecMatXd Svec(q);
         
         for(int i(0); i<q; ++i){
			 Rcpp::NumericMatrix tmp = SList[i];
			 Map<MatrixXd> tmpMap(&tmp(0,0), tmp.nrow(), tmp.ncol());
			 Svec[i] = MatrixXd(tmpMap);
			 };
			 
		 // struct of functors pointing to the log-lik and its derivatives
		 if(fmName == 5){ // extra argument, i.e. n, to family binomial
			 
			 nSizeBinom = y.rows();
			 };
		 StructFamily Family(familyChoice[fmName]);
			 
		 // initialization of outputs passed by ref to MLOptim
		 VectorXd lamb(lambInit);
		 VectorXd beta(betaInit);
		 if(betaInit(0) == -1000000){
			 beta = Family.betaInit(y, pFull);
			 };
		 MatrixXd theta(MatrixXd::Zero(y.rows(), pFull.cols())); // in n x D matrix
		 const int pF(beta.size());
		 MatrixXd Hp(MatrixXd::Zero(pF, pF));
		 VectorXd Up(VectorXd::Zero(pF));
		 double PenL(0.0);
		 VectorXi areIden(VectorXi::LinSpaced(pF,0,pF-1));
		 int nIter(0);
        
        MLOptim(lamb, beta, theta, Hp, Up, PenL, areIden, nIter, Family, Svec, X, y, pFull, pS, rankS, ncBlock, conv);
			 		
		// return full space, not only the identifibale one
		const int pFIden(areIden.size());
		Rcpp::LogicalVector bdrop(pF, true);
		MatrixXd Vp(MatrixXd::Zero(pF,pF)); // is HpInv
		if(pFIden < pF){ // if some are non-identif
			
			// indices of dropped beta's
			for(int i(0); i<pFIden; ++i){
				bdrop[areIden[i]] = false;
				};
  
			// compute HpInv first
			MatrixXd HpInv(pFIden, pFIden);
			HpInv.noalias() = Hp.ldlt().solve(MatrixXd::Identity(pFIden, pFIden));
			
			// update to full space
			VectorXd betaOut(VectorXd::Zero(pF));
			for(int j(0); j<pFIden; ++j){ // iterate over identifiable beta's and keep non-identifibable ones to 0
				betaOut(areIden(j)) = beta(j);
				Vp.col(areIden(j)) = HpInv.col(j);
				Vp.row(areIden(j)) = HpInv.row(j);
				};
				
			beta = betaOut;
				
			} else { // if all are identif only need to compute Vp here
				Vp.noalias() = Hp.ldlt().solve(MatrixXd::Identity(pF, pF));
				};
			
		// compute edf and Ve
		MatrixXd S(MatrixXd::Zero(pF,pF)); 
		for(int j(0); j<q; ++j){ // compute S as block matrix
			S.block(pS(0,j), pS(0,j), pS(1,j), pS(1,j)).noalias() += lamb(j) * Svec[j];
			};
		MatrixXd Fp(MatrixXd::Identity(pF,pF) - Vp * S); //Fp.noalias() = MatrixXd::Identity(pF,pF) - Vp * S; p167 Wood(2006)
//		VectorXd edf(Fp.diagonal()); //edf(Fp.trace()); //edf = pF - Tr(Vp*S)
//		MatrixXd Ve(Fp * Vp); //Ve.noalias() = Fp * Vp; eq 4.34 Wood(2006)
			
		return Rcpp::List::create(Rcpp::Named("coefficients")=beta, Rcpp::Named("fitted.values")=theta, Rcpp::Named("sp")=lamb, Rcpp::Named("edf")=Fp.diagonal(),
		Rcpp::Named("bdrop")=bdrop, Rcpp::Named("l")=PenL, Rcpp::Named("Vp")=Vp, Rcpp::Named("Ve")=Fp * Vp, Rcpp::Named("niter")=nIter);
		
		} catch(std::exception& ex){
			forward_exception_to_r(ex);
			} catch(...){
				::Rf_error("C++ error");
				}
				
				return R_NilValue; // -Wall
}
