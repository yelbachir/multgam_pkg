#ifndef MLOPTIM_HPP_
#define MLOPTIM_HPP_

#ifdef _OPENMP
#include <omp.h>
#endif

#include <RcppEigen.h>

#include "convInfo.hpp"
#include "Family.hpp"
#include "commonTypedefs.hpp"

using namespace Eigen;

struct structIdentif {

    vecMatXd SListIden;
    MatrixXd XIden;
    MatrixXi pFullIden;
    MatrixXi pSIden;
    vecMatXd XtXIden;
    
};

/*****************
 *
 * Initialization
 *
 *****************/
template <typename DerivedA, typename DerivedB>
void initialUnPen(vecMatXd& XtX, const VectorXd& beta, MatrixXd& theta, MatrixXd& Hp, VectorXd& Up, double& PenL, const MatrixBase<DerivedA>& X,
		const Map<MatrixXd>& y, const MatrixBase<DerivedB>& pFull, const StructFamily& Family)
{
	/* computation of XtX, theta, PenL, Hp, Up going through the data only once */

	const int n(y.rows()), D(pFull.cols()), pF(beta.size());
	int i;

	#pragma omp parallel private(i)
	{
		VectorXd gradi(D);
		VectorXd Hivec(D*(D+1)/2);
		int indv(-1);
		MatrixXd Hp_priv(MatrixXd::Zero(pF,pF));
		VectorXd Up_priv(VectorXd::Zero(pF));

		#pragma omp for schedule(static) nowait reduction(+ : PenL)
		for(i = 0; i<n; ++i){ // iterate over the data y_i's

			XtX[i] = MatrixXd(pF,pF).setZero().selfadjointView<Lower>().rankUpdate(X.row(i).transpose());

			for(int d(0); d<D; ++d){ // compute thetai in vector form as the the inner product <X_i^d, beta^d>_{d=1, ..., D}
				theta(i, d) = X.row(i).segment(pFull(0,d),pFull(1,d)) * beta.segment(pFull(0,d),pFull(1,d));
			};

			PenL += Family.logLik(y.row(i), theta.row(i)); // this is the log-lik but will be updated as penalized log-lik

			// compute grad w.r.t beta
			gradi = Family.Deriv1i(y.row(i), theta.row(i)); // Hp = -hess will be updated after the loop
			for(int d(0); d<D; ++d){
				//Up.segment(pFull(0,d), pFull(1,d)).noalias() += gradi(d) * X.row(i).transpose().segment(pFull(0,d), pFull(1,d));
				Up_priv.segment(pFull(0,d), pFull(1,d)).noalias() += gradi(d) * X.row(i).transpose().segment(pFull(0,d), pFull(1,d));
			};

			// compute sum_i XtHXi, i.e., non-penalized hess w.r.t beta
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
		
void computeLambInit(const MatrixXd& Hp, VectorXd& lambInit, const Map<MatrixXi>& pS, const vecMatXd& SList);

/**************************************
 *
 * compute dTheta
 *
 **************************************/

void computedTheta(vecMatXd& dHdTheta, const Map<MatrixXd>& y, const MatrixXd& theta, const StructFamily& Family);

/**************************************
 *
 * Computation of c_k components for lambda
 *
 **************************************/

// compute fixed components of lambda
template <typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD>
double computeBSBTrj(const StructFamily& Family, const DenseBase<DerivedA>& pSj, const VectorXd& beta, const MatrixBase<DerivedB>& Sj,
                     const MatrixXd& HpInv, const MatrixBase<DerivedC>& X, const vecMatXd& XtX, const MatrixBase<DerivedD>& pFull, const Map<MatrixXd>& y, 
                     const vecMatXd& dHdTheta)
{
    // mainly computation of dHp

    // initialize dHp by XtdHX
    VectorXd Sb(pSj(1));
    Sb.noalias() =  Sj * beta.segment(pSj(0),pSj(1)); // Sj * betaj;

    const int pF(beta.size());
    VectorXd dbeta(pF);
    // no need for -Sb as Deriv3 will be used rather than -Deriv3

    dbeta.noalias() = HpInv.middleCols(pSj(0),pSj(1)) * Sb;

    const int n(y.rows());
    MatrixXd dHp(MatrixXd::Zero(pF,pF));

    int i;
	#pragma omp parallel shared(X, pFull, beta, y, dbeta, XtX, dHp) private(i)
    {
        const int D(pFull.cols());
        VectorXd dHivec(D*(D+1)/2);
        int indv(-1);
        MatrixXd dHp_priv(MatrixXd::Zero(pF,pF));

		#pragma omp for schedule(static) nowait //reduction(+ : dHp)
        for(i = 0; i<n; ++i){ // iterate over the data y_i's

            dHivec.setZero();
            for(int d(0); d<D; ++d){
                dHivec.noalias() += (X.row(i).segment(pFull(0,d),pFull(1,d)) * dbeta.segment(pFull(0,d),pFull(1,d))) * dHdTheta[i].col(d);
            }; // end of computation of dHivec

            // compute XtdHXi i.e., non-penalized hess w.r.t beta
            indv = -1;
            for(int dc(0); dc<D; ++dc){ // iterate over columns
                for(int dl(dc); dl<D; ++dl){ // iterate over rows
                    ++indv;
                    //dHp.block(pFull(0,dl), pFull(0,dc), pFull(1,dl), pFull(1,dc)).noalias() += dHivec(indv) * XtX[i].block(pFull(0,dl), pFull(0,dc), pFull(1,dl), pFull(1,dc));
                    dHp_priv.block(pFull(0,dl), pFull(0,dc), pFull(1,dl), pFull(1,dc)).noalias() += dHivec(indv) * XtX[i].block(pFull(0,dl), pFull(0,dc), pFull(1,dl), pFull(1,dc));
                };
            }; // end of computation of XtdHXi
        }; // end of iteration over the data y_i's

		#pragma omp critical
        dHp += dHp_priv;
    }

    // make dHp symmetric as all the computations were performed only on the lower triangular part
    dHp.triangularView<Upper>() = dHp.transpose();

    // so far dHp is actually XtdHX, still needs addition of penalty terms
    dHp.block(pSj(0), pSj(0), pSj(1), pSj(1)) += Sj;

    return (beta.segment(pSj(0),pSj(1))).dot(Sb) + (HpInv * dHp).trace();
};

/**************************************
 *
 * Computations in the reduced space of identifiable beta
 *
 **************************************/

// reduce arguments to the corresponding identifiable regression coefs and optimize pen log-lik again
structIdentif Reduce2Identif(VectorXd& beta, MatrixXd& theta, MatrixXd& Hp, VectorXd& Up, double& PenL, const StructFamily& Family, const VectorXd& lamb, const VectorXi& areIden, VectorXd& rankSIden,
                             const vecMatXd& SList, const Map<MatrixXd>& X, const Map<MatrixXi>& pFull, const Map<MatrixXi>& pS, const int& HpRank, const Map<MatrixXd>& y, const convInfo& conv);

/**************************************
 *
 * Computations of elements of log-lik
 *
 **************************************/
// compute theta, unpenalized log-lik, gradient and negative hessian
void computeUnPen(const VectorXd& beta, MatrixXd& theta, MatrixXd& Hp, VectorXd& Up, double& PenL, const Map<MatrixXd>& y,
                  const Map<MatrixXi>& pFull, const Map<MatrixXd>& X, const vecMatXd& XtX, const StructFamily& Family);

/**************************************
 *
 * main EM function
 *
 **************************************/

void MLOptim(VectorXd& lamb, VectorXd& beta, MatrixXd& theta, MatrixXd& Hp, VectorXd& Up, double& PenL, VectorXi& areIden, int& nIter, const StructFamily& Family, const vecMatXd& SList,
             const Map<MatrixXd>& X, const Map<MatrixXd>& y, const Map<MatrixXi>& pFull, const Map<MatrixXi>& pS, const Map<VectorXd>& rankS, MatrixXi& ncBlock, const convInfo& conv);

#endif /* MLOPTIM_HPP_ */
