/**************************************
 **************************************
 ******
 ******  Main c++ optimization function
 ******  called by mtgamcpp
 *
 ******  Goal: ML optimization algorithm :
 ******  estimate the smoothing pars using
 ******  the approximate EM
 ******
 **************************************
 **************************************/

#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>

#include "../inst/include/MLOptim.hpp"
#include "../inst/include/PenOptim.hpp"
#include "../inst/include/commonDefs.hpp"
#include "../inst/include/smoothOptim.hpp"

using namespace Eigen;

/**************************************
 *
 * ML optimization algorithm :
 * estimate the smoothing pars using the approximate EM
 *
 * 
 **************************************/

/*****************
 *
 * Initialization
 *
 *****************/

void computeLambInit(const MatrixXd& Hp, VectorXd& lambInit, const Map<MatrixXi>& pS, const vecMatXd& SList)
{
	/* given betaInit, compute the corresponding lambInit */

	// Hp is XtHX where H is unpenalized hessian
	const int q(pS.cols());
	int k(0), p(0);
	std::vector<double> a(Hp.rows());
	double tmp(0.0);

	//int j;
	//#pragma omp parallel for num_threads(std::min(omp_get_max_threads(), q)) private(j)
	//for(j = 0; j<q; ++j){ // iterate over the smoothing matrices
	for(int j(0); j<q; ++j){ // iterate over the smoothing matrices

		k = 0;
		p = pS(1,j);
		for(int i(0); i<p; ++i){ // iterate over the diagonal of Sj

			tmp = sqrTol * SList[j].maxCoeff();
			if(std::abs(SList[j](i,i)) > tmp){
				a[k] = std::abs(Hp(i,i)/SList[j](i,i));
				++k;
				};
			};

		// take the median
		std::nth_element(a.begin(), a.begin() + k/2, a.begin()+k);
		lambInit(j) = a[k/2];
		};
};

/**************************************
 *
 * compute dTheta
 *
 **************************************/

void computedTheta(vecMatXd& dHdTheta, const Map<MatrixXd>& y, const MatrixXd& theta, const StructFamily& Family)
{
    const int n(y.rows());
    int i;

	#pragma omp parallel shared(y, theta, dHdTheta) private(i)
    {
		#pragma omp for schedule(static)
        for(i = 0; i<n; ++i){
            // compute dHidTthetai in vector form as the sum of the inner products sum_{i=1}^D <X_i^d, dbeta^d> * dHidTthetai
            dHdTheta[i] = Family.Deriv3i(y.row(i), theta.row(i));
        };
    }
};


/**************************************
 *
 * Computations in the reduced space of identifiable beta
 *
 **************************************/

// reduce arguments to the corresponding identifiable regression coefs and optimize pen log-lik again
structIdentif Reduce2Identif(VectorXd& beta, MatrixXd& theta, MatrixXd& Hp, VectorXd& Up, double& PenL, const StructFamily& Family, const VectorXd& lamb, const VectorXi& areIden, VectorXd& rankSIden,
                             const vecMatXd& SList, const Map<MatrixXd>& X, const Map<MatrixXi>& pFull, const Map<MatrixXi>& pS, const int& HpRank, const Map<MatrixXd>& y, const convInfo& conv)
{
    // beta (input) is the full beta that will be reduced
    // work on a betaRed which will be the intermediate beta and set beta = betaRed at the end
    // do not form XIden by copying relevant cols of X but use pointers to relevant cols

    // prepare output
    const int n(y.rows()), D(pFull.cols()), q(pS.cols());
    structIdentif out = { SList, MatrixXd::Zero(n, HpRank), MatrixXi::Zero(2, D), MatrixXi::Zero(2, q), vecMatXd(n) };
    for(int ii(0); ii<q; ++ii){
        out.SListIden[ii].setZero(); // keep the initialization by SList because we only at compile time the size of SList[ii]
    };

    // prepare for loop
    int d(0);
    VectorXd betaRed(HpRank); // will become beta at the end
    int b(pS(0,0)-1), i(0), k(0);
    int j(0), jjm1(HpRank-1);
    int tmp(0);
    bool inSj(true), endSj(true);


    // main loop
    for(int jj(0); jj<jjm1; ++jj){ // iterate over identifiable ones

        // fill pFullIden : index of the 1st beta corresponding to the d-th functional param
        if((d < D) and (areIden(jj) > pFull(0,d)-1)){
            out.pFullIden(0,d) = jj;
            ++d;
        };

        // update non-smooth terms, i.e., XIden and betaRed
        out.XIden.col(jj) = X.col(areIden(jj));
        betaRed(jj) = beta(areIden(jj));

        // update smooth terms, i.e., pS, SList, rankSIden
        // if else restricting to the beta's of the smoothing matrices
        if(j < q){ // if it hasn't reached the last smoothing matrix

            tmp = b+pS(1,j)+1; // (length Sj)-1
            inSj = ((areIden(jj) > b) and (areIden(jj) < tmp));
            if(inSj){ // if areIden(jj) is in Sj (including bounds)

                // if crossing Sj for the 1st time
                if(i==0){ // index of the identifiable beta of the current smoothing matrix j
                    out.pSIden(0,j) = jj;
                };

                while(areIden(jj) > pS(0,j)+i){ ++i; }; // get index i of the identifiable beta of the current smoothing matrix j corresponding to areIden(jj)
                out.SListIden[j].col(k) = SList[j].col(i);
                out.SListIden[j].row(k) = SList[j].row(i);
                ++i; // index of the next identifiable beta of the current smoothing matrix j
                ++k; // nb of identifiable beta of the current smoothing matrix j

            };

            endSj = (inSj and (areIden(jj+1) > tmp-1));
            if(endSj){ // summarize the findings for the current smoothing matrix j before going through the next one

                out.pSIden(1,j) = k;
                if(k==0){
                    Rcpp::stop("Regression params of the smooth ", j , " are not identifiable! This smooth function may be non-significant, remove it from the model formula and try again");
                };

                out.SListIden[j].conservativeResize(k,k);
                if(k != pS(1,j)){ // if SListIden[j] has changed, compute its new rank
                    ColPivHouseholderQR<MatrixXd> SjQR(k,k);
                    SjQR.setThreshold(sqrTol);
                    SjQR.compute(out.SListIden[j].selfadjointView<Lower>());
                    rankSIden(j) = SjQR.rank();
                    if(rankSIden(j)==0){
                        Rcpp::stop("Regression params of the smooth " , j , " are not identifiable! This smooth function may be non-significant, remove it from the model formula and try again");
                    };
                };

                ++j; // update to the next smoothing matrix

                if(j < q){
                    b = pS(0,j)-1;
                    i = 0;
                    k = 0;
                };
            }; // end of summary
        }; // end of not reaching the last smoothing matrix

    }; // end main loop

    // compute corresponding components for areIden(jjm1=HpRank-1)
    out.XIden.col(jjm1) = X.col(areIden(jjm1));
    betaRed(jjm1) = beta(areIden(jjm1));

    if(j < q){ // there are still smooth components

        if(inSj and (areIden(jjm1) < tmp) ){ // areIden(jjm1-1) and areIden(jjm1) are both in the same Sj
            while(areIden(jjm1) > pS(0,j)+i){ ++i; };
            out.SListIden[j].col(k) = SList[j].col(i);
            out.SListIden[j].row(k) = SList[j].row(i);
            ++i;
            ++k;
            out.pSIden(1,j) = k;
            if(k==0){
                Rcpp::stop("Regression params of the smooth " , j , " are not identifiable! This smooth function may be non-significant, remove it from the model formula and try again");
            };
            out.SListIden[j].conservativeResize(k,k);
            if(k != pS(1,j)){ // if SListIden[j] has changed, compute its new rank
                ColPivHouseholderQR<MatrixXd> SjQR(k,k);
                SjQR.setThreshold(sqrTol);
                SjQR.compute(out.SListIden[j].selfadjointView<Lower>());
                rankSIden(j) = SjQR.rank();
                if(rankSIden(j)==0){
                    Rcpp::stop("Regression params of the smooth " , j , " are not identifiable! This smooth function may be non-significant, remove it from the model formula and try again");
                };
            };
        } else { // areIden(jjm1) is in some Sj

            while(j < q){
                tmp = b+pS(1,j)+1;
                inSj = (areIden(jjm1) > b) and (areIden(jjm1) < tmp);

                if(inSj){
                    if(i==0){ out.pSIden(0,j) = jjm1; };
                    while(areIden(jjm1) > pS(0,j)+i){ ++i; };
                    out.SListIden[j].col(0) = SList[j].col(i);
                    out.SListIden[j].row(0) = SList[j].row(i);
                    out.SListIden[j].conservativeResize(0,0);
                    rankSIden(j) = 1; // need to check this
                    out.pSIden(1,j) = 1;
                    j = q; // to break the loop
                    }; // end inSj

                    ++j;
                }; // end while
            };
        }; // end assignation of areIden(jjm1)


        int dm1(D-1);
        for(int dd(0); dd<dm1; ++dd){
            out.pFullIden(1,dd) = out.pFullIden(0,dd+1)-out.pFullIden(0,dd);
            };
        out.pFullIden(1,D-1) = HpRank-out.pFullIden(0,D-1);
        beta = betaRed;

    /* iterate again for having the final reduced beta */
    Hp.resize(HpRank, HpRank);
    Hp.setZero();
    Up.resize(HpRank);
    Up.setZero();
    PenL = 0.0;

    initialUnPen(out.XtXIden, beta, theta, Hp, Up, PenL, out.XIden, y, out.pFullIden, Family);
    PenllOptim(beta, theta, Hp, Up, PenL, Family, lamb, lamb, y, out.SListIden, out.XIden, out.XtXIden, out.pFullIden, out.pSIden, conv);

    return out;
};

/**************************************
 *
 * Computations of elements of log-lik
 *
 **************************************/
// compute theta, unpenalized log-lik, gradient and negative hessian
// need not be templated because it is called in MLOptim for Map<...>
void computeUnPen(const VectorXd& beta, MatrixXd& theta, MatrixXd& Hp, VectorXd& Up, double& PenL, const Map<MatrixXd>& y,
                  const Map<MatrixXi>& pFull, const Map<MatrixXd>& X, const vecMatXd& XtX, const StructFamily& Family)
{

    const int pF(Hp.rows()), n(y.rows());

    Hp.setZero();
    Up.setZero();
    PenL = 0.0;
    int i;

	#pragma omp parallel shared(X, pFull, beta, theta, y, XtX, Hp, Up) private(i)
    {
        const int D(pFull.cols());
        VectorXd gradi(D);
        VectorXd Hivec(D*(D+1)/2);
        int indv(-1);
        MatrixXd Hp_priv(MatrixXd::Zero(pF,pF));
        VectorXd Up_priv(VectorXd::Zero(pF));

		#pragma omp for schedule(static) nowait reduction(+ : PenL)
        for(i = 0; i<n; ++i){ // iterate over the data y_i's

        	for(int d(0); d<D; ++d){
        					theta(i, d) = X.row(i).segment(pFull(0,d),pFull(1,d)) * beta.segment(pFull(0,d),pFull(1,d));
        				};

            PenL += Family.logLik(y.row(i), theta.row(i)); // this is the log-lik but will be updated as penalized log-lik
            // compute grad w.r.t beta
            gradi = Family.Deriv1i(y.row(i), theta.row(i)); // Hp = -hess will be updated after the loop
            for(int d(0); d<D; ++d){
                //Up.segment(pFull(0,d), pFull(1,d)).noalias() += gradi(d) * X.row(i).transpose().segment(pFull(0,d), pFull(1,d));
                Up_priv.segment(pFull(0,d), pFull(1,d)).noalias() += gradi(d) * X.row(i).transpose().segment(pFull(0,d), pFull(1,d));
            };

            // compute sum_i XtHXi
            indv = -1;
            Hivec = Family.Deriv2i(y.row(i), theta.row(i));
            for(int dc(0); dc<D; ++dc){ // iterate over columns
                for(int dl(dc); dl<D; ++dl){ // iterate over rows
                    ++indv;
                    //Hp.block(pFull(0,dl), pFull(0,dc), pFull(1,dl), pFull(1,dc)).noalias() += Hivec(indv)*XtX[i].block(pFull(0,dl), pFull(0,dc), pFull(1,dl), pFull(1,dc));
                    Hp_priv.block(pFull(0,dl), pFull(0,dc), pFull(1,dl), pFull(1,dc)).noalias() += Hivec(indv)*XtX[i].block(pFull(0,dl), pFull(0,dc), pFull(1,dl), pFull(1,dc));
                };
            }; // end of computation of XtHXi
        }; // end of iteration over the data y_i's

		#pragma omp critical
        Hp += Hp_priv;
        Up += Up_priv;
    }

    Hp = -Hp; // so far Hp is XtHX
    // make Hp symmetric as all the computations were performed only on the lower triangular part
    Hp.triangularView<Upper>() = Hp.transpose();
};



/**************************************
 *
 * main function
 *
 **************************************/
 
void MLOptim(VectorXd& lamb, VectorXd& beta, MatrixXd& theta, MatrixXd& Hp, VectorXd& Up, double& PenL, VectorXi& areIden, int& nIter, const StructFamily& Family, const vecMatXd& SList,
             const Map<MatrixXd>& X, const Map<MatrixXd>& y, const Map<MatrixXi>& pFull, const Map<MatrixXi>& pS, const Map<VectorXd>& rankS, MatrixXi& ncBlock, const convInfo& conv)
{
    // lamb (input) is initial lambda, lamb (output) is the best lamb
    // beta (input) is initial beta, beta (output) is the best beta
    // Hp (input) is 0, Hp (output) is Hp(best beta)
    // PenL (input) is 0, PenL (output) is Lp(best beta)
    // areIden (input) is the vector of indices of the full beta, areIden (output) is the indices of the best identifiable beta

    /******************
     * General strategy : after every PenllOptim call, check identifiability of the beta's
     * if Hp is full rank, use the space of identifiable beta (the original components) to compute fixed_k
     * if not, get indices of identifiable beta's and reduce beta, HpInv, PenL, pS, SList, X, pFull to
     * identifiable space and optimize penalized log-lik again
     ******************/

    const int n(y.rows());
    vecMatXd XtX(n);

    /* initialUnPen can be performed by blocks in the WORKERS, the reduction + should be done in the MANAGER */
    initialUnPen(XtX, beta, theta, Hp, Up, PenL, X, y, pFull, Family); // XtX, theta, oldPen, Up, Hp
    if(lamb(0) < 0.0){  // if lambda not supplied
    	computeLambInit(Hp, lamb, pS, SList);
    };

    /* given lamb, optimize penalized log-lik to get beta as starting beta for the EM */
    PenllOptim(beta, theta, Hp, Up, PenL, Family, lamb, lamb, y, SList, X, XtX, pFull, pS, conv);
    // beta has become beta_k, Hp has become Hp(beta_k), oldPen has become Lp(beta_k)

    /* compute c_k on the space of identifiable beta */
    // set-ups for beta
    const int pF(beta.size());
    ColPivHouseholderQR<MatrixXd> HpQR(pF, pF);
    HpQR.setThreshold(sqrTol);
    HpQR.compute(Hp.selfadjointView<Lower>());
    int HpRank(HpQR.rank());
    LDLT<MatrixXd> ldlHp(pF); // stable Cholesky decomposition of Hp
    MatrixXd HpInv(pF, pF);
    vecMatXd dHdTheta(n);
    VectorXd oldBeta(beta);
    double oldPen(PenL);

    // set-ups for lambda
    VectorXd rankSIden(rankS), rankSRef(rankS);
    const int q(lamb.size());
    VectorXd oldFixed(q); // c_k in the paper

    /*****************************
     * computation of oldFixed_k, i.e., c_k in the paper:
     * need to adjust beta, HpInv, PenL, pS, SList, X, pFull to identifiable space after estimation from PenllOptim
     * pass oldFit and rankSiden by reference to the function Reduce2Identif,
     * which modifies them directly when called,
     * and outputs the corresponding identifiable arguments
     ******************************/

    // define a named lambda to avoid code repetition in the if-else statement for the identifiable space of beta
    auto c_kVec = [&] (const Ref<const MatrixXi>& pS, const vecMatXd& SList, const Ref<const MatrixXd>& X, const vecMatXd& XtX, const Ref<const MatrixXi>& pFull, const int& p)
		  {

              ldlHp.compute(Hp.selfadjointView<Lower>());
              HpInv.resize(p, p);
              HpInv = ldlHp.solve(MatrixXd::Identity(p, p));
              computedTheta(dHdTheta, y, theta, Family);

              // 1 per node
              for(int j(0); j<q; ++j){ // iterate over non-converged lambda's
                  oldFixed(j) = computeBSBTrj(Family, pS.col(j), oldBeta, SList[j], HpInv, X, XtX, pFull, y, dHdTheta);
              };

              return oldFixed;
          };

    if(HpRank < pF){ // reduce beta, X, SList, pFull, pS, rankSIden to identifiable ones

    	// identify identifiable beta
    	areIden = HpQR.colsPermutation().indices(); // indices of all the pivots
    	areIden.conservativeResize(HpRank); // extract only significant ones
    	std::sort(areIden.data(), areIden.data()+HpRank); // sort them in increasing order

        structIdentif Iden(Reduce2Identif(oldBeta, theta, Hp, Up, oldPen, Family, lamb, areIden, rankSIden, SList, X, pFull, pS, HpRank, y, conv));
        rankSRef = rankSIden;
        oldFixed = c_kVec(Iden.pSIden, Iden.SListIden, Iden.XIden, Iden.XtXIden, Iden.pFullIden, HpRank);

    } else { // all beta's are identifiable so X, SList, pFull, pS, pF should be original ones
        oldFixed = c_kVec(pS, SList, X, XtX, pFull, pF);
    };

    /* initialization for the updating loop */
    bool converged(false);
    int nnCv(q), ncTmp(0); // nb of non-converged lamb
    VectorXi NC(VectorXi::LinSpaced(q,0,q-1)); // indices w.r.t. full lamb of non-converged lamb
    VectorXd oldLamb(lamb);
    VectorXd diffLamb(q);
    VectorXd fixed(oldFixed); // c_k+1 in the paper
    double oldDiffL(0.0), newDiffL(0.0);  // used in convergence test

    // define a named lambda to avoid code repetition in the if-else statement for the identifiable space of beta
    auto c_kConv = [&] (const Ref<const MatrixXi>& pS, const vecMatXd& SList, const Ref<const MatrixXd>& X, const vecMatXd& XtX, const Ref<const MatrixXi>& pFull, const int& p)
		  {
			  double diffLp(std::abs(PenL-oldPen));

              /* test convergence */
              // (no significant change in lp) or (small change in lp but significant change in lambda)
              if( (diffLp < conv.PenTol*std::abs(oldPen)) or ((diffLp < 0.1) and (newDiffL > 1.01 * oldDiffL)) ){ 
                  converged = true;

              } else { /* check whether c_k+1 has converged, otherwise keep the corresponding lambda */
					   
					   /* compute c_k+1 for non-converged lambda */
					   computedTheta(dHdTheta, y, theta, Family);
					   ldlHp.compute(Hp.selfadjointView<Lower>());
					   HpInv.resize(p, p);
					   HpInv = ldlHp.solve(MatrixXd::Identity(p, p));
					   
					   // define NC as indices w.r.t. full lambda of non-converged lambda
					   ncTmp = 0;
					   
					   /* if no significant change in lamb, reduce MLTol until change becomes significant */
					   for(int j(0); j<nnCv; ++j){ // iterate over non-converged lambda's, 1 per worker
						   
						   /* if significant changes in lamb_j, update its value using c_k+1 */
						   if(std::abs(lamb(NC(j))-oldLamb(NC(j))) > conv.MLTol*std::abs(PenL) ){ // non-converged lambda_j
							   fixed(NC(j)) = computeBSBTrj(Family, pS.col(NC(j)), beta, SList[NC(j)], HpInv, X, XtX, pFull, y, dHdTheta); // c_k+1 in the paper;
							   NC(ncTmp) = NC(j); // update top elements of NC by indices of non-converged lambda
							   ++ncTmp;
							   };
						}; // end of iteration over nnCv
						
						nnCv = ncTmp;
					}; // end of convergence test
							
				return fixed;
          };

    // updating loop
    while( (!converged) && (nIter < conv.iterMax)){
        ++nIter;
        
        for(int j(0); j<nnCv; ++j){ // iterate over non-converged lambda's
            lamb(NC(j)) = rankSRef(NC(j))/oldFixed(NC(j));
        };
        
        /*
         *****************************
         * input of PenllOptim : identifiable beta_k (where the unindentifiables have been replaced with 0s) whereas SList, X, pF, pS are the original ones
         * output of PenllOptim : identifiable beta_k+1 and Hp_k+1
         *****************************
         */

        diffLamb = lamb-oldLamb;
        newDiffL = diffLamb.lpNorm<Infinity>(); // used in convergence test
        
        if(HpRank < pF){ // augment beta_k with 0s corresponding to unidentifiable components to get beta_k+1 full size from PenllOptim

            // fill unidentifiable beta's with 0's and update bestBeta
            beta.resize(pF);
            beta.setZero();
            for(int jj(0); jj<HpRank; ++jj){ // iterate over identifiable beta's
                beta(areIden(jj)) = oldBeta(jj);
            };
            Hp.resize(pF, pF);
            Up.resize(pF);
            computeUnPen(beta, theta, Hp, Up, PenL, y, pFull, X, XtX, Family);
            diffLamb = lamb;
        };

        PenllOptim(beta, theta, Hp, Up, PenL, Family, diffLamb, lamb, y, SList, X, XtX, pFull, pS, conv);

        // check identifiability of the beta's :
        // if Hp is full rank, use the beta and the original SList, X, etc... to compute fixed_k+1
        // if Hp is not full rank, get indices of identifiable beta's and reduce beta, SList, X, etc... to identifiable ones and optimize penalized log-like again
        HpQR.compute(Hp.selfadjointView<Lower>());
        HpRank = HpQR.rank();

        // compute fixed_k+1 for non-converged lambda's depending on whether all the beta's are identifiable or not
        if(HpRank < pF){ // reduce beta, X, SList, pFull, pS, rankSIden to identifiable ones

            rankSIden = rankS;
            // identify identifiable beta
    	    areIden = HpQR.colsPermutation().indices(); // indices of all the pivots
            areIden.conservativeResize(HpRank); // extract only significant ones
            std::sort(areIden.data(), areIden.data()+HpRank); // sort them in increasing order
            
            structIdentif Iden(Reduce2Identif(beta, theta, Hp, Up, PenL, Family, lamb, areIden, rankSIden, SList, X, pFull, pS, HpRank, y, conv));
            rankSRef = rankSIden;
            fixed = c_kConv(Iden.pSIden, Iden.SListIden, Iden.XIden, Iden.XtXIden, Iden.pFullIden, HpRank);

        } else {

            fixed = c_kConv(pS, SList, X, XtX, pFull, pF);
            rankSRef = rankS;
        };

        // for debugging
        if(conv.progressML){
            Rcpp::Rcout << nIter << " max|Diff(lamb)| " << newDiffL << " max|gradML| " << (fixed-oldFixed).lpNorm<Infinity>() << " Diff(Pen) "
            << PenL-oldPen << std::endl;
        };

        // test full convergence
        if(!converged){ // need to iterate again
			oldDiffL = newDiffL;
			oldLamb = lamb;
            oldBeta = beta;
            oldPen = PenL;
            oldFixed = fixed;
            };

    }; // end of main updating loop

    if(!converged){
        Rcpp::stop("MLOptim has reached the limit number of iterations without converging. Trace the code using progressML=TRUE");
    };

};
