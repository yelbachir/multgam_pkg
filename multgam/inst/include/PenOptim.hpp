/**************************************
 **************************************
 ******
 ******  Penalized log-lik optimization
 ******  based on Newton--Raphson algo
 ******
 **************************************
 **************************************/

#ifndef PENOPTIM_HPP_
#define PENOPTIM_HPP_

#ifdef _OPENMP
#include <omp.h>
#endif

#include <RcppEigen.h>
#include <string>
#include <iostream>
#include <cmath>

#include "commonTypedefs.hpp"
#include "convInfo.hpp"
#include "Family.hpp"

using namespace Eigen;

/**************************************
 *
 * Computation of : (through overloaded function)
 * - unpen log-lik
 * - unpen gradient and -hessian
 * - unpen log-lik, gradient and -hessian
 *
 **************************************/

// compute unpenalized log-lik
// must be templated because of Reduce2Identif which takes Map<...> args and their ... counterparts
template <typename DerivedA, typename DerivedB>
double computeUnPen(const MatrixBase<DerivedA>& X, const MatrixBase<DerivedB>& pFull, const VectorXd& beta, MatrixXd& theta, const Map<MatrixXd>& y,
                    const StructFamily& Family)
{
    const int D(pFull.cols()), n(y.rows());
    double Pen(0.0);
    int i;

	#pragma omp parallel shared(X, pFull, beta, theta, y) private(i)
    {
		#pragma omp for schedule(static) nowait reduction(+ : Pen)
        for(i = 0; i<n; ++i){
            for(int d(0); d<D; ++d){
                theta(i, d) = X.row(i).segment(pFull(0,d),pFull(1,d)) * beta.segment(pFull(0,d),pFull(1,d));
            };

            Pen += Family.logLik(y.row(i), theta.row(i));
        };
    }

    return Pen;
};


// compute unpenalized gradient and negative hessian
// must be templated because of Reduce2Identif which takes Map<...> args and their ... counterparts
template <typename DerivedA, typename DerivedB>
void computeUnPen(const MatrixXd& theta, MatrixXd& Hp, VectorXd& Up, const Map<MatrixXd>& y,
                  const MatrixBase<DerivedA>& pFull, const MatrixBase<DerivedB>& X, const vecMatXd& XtX, const StructFamily& Family)
{
    const int pF(Hp.rows()), n(y.rows()), D(pFull.cols());

    Hp.setZero();
    Up.setZero();
    int i;

	#pragma omp parallel shared(X, pFull, theta, y, XtX, Hp, Up) private(i)
    {
        VectorXd gradi(D);
        VectorXd Hivec(D*(D+1)/2);
        int indv(-1);
        MatrixXd Hp_priv(MatrixXd::Zero(pF,pF));
        VectorXd Up_priv(VectorXd::Zero(pF));

		#pragma omp for schedule(static) nowait
        for(i = 0; i<n; ++i){ // iterate over the data y_i's

            // compute grad w.r.t beta
            gradi = Family.Deriv1i(y.row(i), theta.row(i)); // Hp = -hess will be updated after the loop
            for(int d(0); d<D; ++d){
                //Up.segment(pFull(0,d), pFull(1,d)).noalias() += gradi(d) * X.row(i).transpose().segment(pFull(0,d), pFull(1,d));
                Up_priv.segment(pFull(0,d), pFull(1,d)).noalias() += gradi(d) * X.row(i).transpose().segment(pFull(0,d), pFull(1,d));
            };

            // compute XtHXi, i.e., non-penalized hess w.r.t beta
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
    Hp.triangularView<Upper>() = Hp.transpose();
};

// compute unpenalized log-lik, gradient and negative hessian
// must be templated because of Reduce2Identif which takes Map<...> args and their ... counterparts
template <typename DerivedA, typename DerivedB>
void computeUnPen(const MatrixXd& theta, MatrixXd& Hp, VectorXd& Up, double& PenL, const Map<MatrixXd>& y,
                  const MatrixBase<DerivedA>& pFull, const MatrixBase<DerivedB>& X, const vecMatXd& XtX, const StructFamily& Family)
{

    const int pF(Hp.rows());
    const int n(y.rows());

    Hp.setZero();
    Up.setZero();
    PenL = 0.0;
    int i;

	#pragma omp parallel shared(X, pFull, theta, y, XtX, Hp, Up) private(i)
    {
        const int D(pFull.cols());
        VectorXd gradi(D);
        VectorXd Hivec(D*(D+1)/2);
        int indv(-1);
        MatrixXd Hp_priv(MatrixXd::Zero(pF,pF));
        VectorXd Up_priv(VectorXd::Zero(pF));

		#pragma omp for schedule(static) nowait reduction(+ : PenL)
        for(i = 0; i<n; ++i){ // iterate over the data y_i's

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
 * Modified Cholesky correction of Hp
 *
 **************************************/
 
void solveModChol(MatrixXd& Hp, VectorXd& delta); // decompose and solve
void solveStablModChol(MatrixXd& Hp, VectorXd& b); // precondition decompose and solve

/**************************************
 *
 * Main NR algo
 *
 **************************************/	

void UpdateNewton(MatrixXd& Hp, VectorXd& delta);


// must be templated because of Reduce2Identif which takes Map<...> args and their ... counterparts
template <typename DerivedA, typename DerivedB, typename DerivedC>
void PenllOptim(VectorXd& beta, MatrixXd& theta, MatrixXd& Hp, VectorXd& Up, double& PenL, const StructFamily& Family,
                const VectorXd& diffLamb, const VectorXd& lamb, const Map<MatrixXd>& y, const vecMatXd& SList,
                const MatrixBase<DerivedA>& X, const vecMatXd& XtX, const MatrixBase<DerivedB>& pFull, const MatrixBase<DerivedC>& pS, const convInfo& conv)
{
    // beta (input) is initial beta
    // beta (output) is best beta
    // usage:
    // for every call to PenllOptim, beta should be initial beta, Hp Up and PenL should be un-penlized ones

    /*********************************
     *
     * initialization
     *
     ********************************/

    // compute penalized forms of oldPen, Up and Hp based on unpenalized ones
    const int q(pS.cols());
    vecMatXd Sl(q);
    double oldPen(PenL);
    const int pF(beta.size());
    //int j;

    //#pragma omp parallel num_threads(std::min(omp_get_max_threads(), q)) private(j)
    //{
    	//MatrixXd Hp_priv(MatrixXd::Zero(pF, pF));
    	//VectorXd Up_priv(VectorXd::Zero(pF));
    	//double pen(0.0);

    	//#pragma omp for schedule(static) nowait
    	//for(j = 0; j<q; ++j){
	for(int j(0); j<q; ++j){

    		MatrixXd SlTmp(SList[j] * diffLamb(j)); // size of Sl is adjusted automatically by Eigen
    		Sl[j].noalias() = SList[j] * lamb(j);
    		VectorXd Sbl(SlTmp * beta.segment(pS(0,j), pS(1,j))); // size of Sbl is adjusted automatically by Eigen
    		//pen -= 0.5*beta.segment(pS(0,j),pS(1,j)).dot(Sbl);
    		//Up_priv.segment(pS(0,j), pS(1,j)) -= Sbl;
    		//Hp_priv.block(pS(0,j), pS(0,j), pS(1,j), pS(1,j)) += SlTmp;
		oldPen -= 0.5*beta.segment(pS(0,j),pS(1,j)).dot(Sbl);
        	Up.segment(pS(0,j), pS(1,j)) -= Sbl;
        	Hp.block(pS(0,j), pS(0,j), pS(1,j), pS(1,j)) += SlTmp;
    	};

    	//#pragma omp critical
    	//oldPen += pen;
    	//Up += Up_priv;
    	//Hp += Hp_priv;
    //}

    // set-ups
    std::string w;
    VectorXd delta(pF);
    bool converged(false);
    int iter(0);
    VectorXd oldBeta(beta);

    // lambda fct for step size reduction
    auto stepSizeReduc = [&] () {

	int k_iter(0);

        while( (PenL < oldPen) and (k_iter < 21) ){

            delta *= 0.5;
            beta = oldBeta + delta;
            PenL = computeUnPen(X, pFull, beta, theta, y, Family);
            for(int j(0); j<q; ++j){
                PenL -= 0.5*beta.segment(pS(0,j), pS(1,j)).dot(Sl[j] * beta.segment(pS(0,j),pS(1,j)));
            };
            ++k_iter;

        };
    };

    /*********************************
     *
     * updating loop
     *
     *********************************/

    while(!converged and (iter < conv.iterMax)){
        ++iter;

        // Newton update
        delta = Up;
        UpdateNewton(Hp, delta);
        beta = oldBeta + delta;
        PenL = computeUnPen(X, pFull, beta, theta, y, Family);
        for(int j(0); j<q; ++j){
            PenL -= 0.5*beta.segment(pS(0,j), pS(1,j)).dot(Sl[j] * beta.segment(pS(0,j),pS(1,j)));
        };
        w = "No-step-reduction";

        /*********************************
         *
         * step halving if needed
         *
         *********************************/

        // if the update has not increased the penalized loglik use step halving
        if(PenL < oldPen){

            // Newton step halving
            stepSizeReduc();
            w = "Newton";

            // if Newton step halving failed try steepest ascent
            if(PenL < oldPen){

                delta = Up;
                beta = oldBeta + delta;
                PenL = computeUnPen(X, pFull, beta, theta, y, Family);
                for(int j(0); j<q; ++j){ // Lp(beta)
                    PenL -= 0.5*beta.segment(pS(0,j), pS(1,j)).dot(Sl[j] * beta.segment(pS(0,j),pS(1,j)));
                };
                stepSizeReduc(); // end of steepest ascent
                w = "Steep";
            };
        }; // end of step halving

        /*********************************
         * test convergence in case penalized loglik has increased
         *********************************/

        // compute penalized grad and hess
        computeUnPen(theta, Hp, Up, y, pFull, X, XtX, Family);
        for(int j(0); j<q; ++j){ // compute penalties
            VectorXd Sbl(Sl[j] * beta.segment(pS(0,j), pS(1,j)));
            Up.segment(pS(0,j), pS(1,j)) -= Sbl;
            Hp.block(pS(0,j), pS(0,j), pS(1,j), pS(1,j)) += Sl[j];
        };

        // track progress for debugging
        if(conv.progressPen){
			Rcpp::Rcout << iter << " " << w << " : Diff(Pen) " << PenL-oldPen << ", max|Up| " << Up.lpNorm<Infinity>() << std::endl;
			};
        
        //if( (std::abs(PenL-oldPen) < sqrTol*penConv) and ((oldBeta-beta).lpNorm<Infinity>() < sqrTol*(beta.lpNorm<Infinity>())) and (Up.lpNorm<Infinity>() < MLTol*penConv ) ){
        if( (std::abs(PenL-oldPen) < conv.PenTol*std::abs(oldPen)) and ( Up.lpNorm<Infinity>() < conv.MLTol*std::abs(oldPen) ) ){
            converged = true;
        } else {
            if( (PenL < oldPen) or (std::abs(PenL-oldPen) < conv.PenTol) ){ // no improvement made after step-halving and gradient is still significantly different from 0
                Rcpp::stop("PenllOptim did not converge after <steep-PenllOptim>. Trace the code using progressPen=TRUE!");
            };
            oldBeta = beta;
            oldPen = PenL;
        };

    }; // end updating loop from beta_k to beta_k+1

    if(!converged){
        Rcpp::stop("PenllOptim has reached the limit number of iterations without converging. Trace the code using progressPen=TRUE");
    };
};





#endif /* PENOPTIM_HPP_ */

