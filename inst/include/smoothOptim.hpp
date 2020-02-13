/**************************************
 *
 * optimization for smooths
 *
 **************************************/

#ifndef SMOOTHOPTIM_HPP_
#define SMOOTHOPTIM_HPP_

#ifdef _OPENMP
#include <omp.h>
#endif

#include <RcppEigen.h>
#include <iostream>
#include <string>
#include <vector>

#include "commonTypedefs.hpp"
#include "commonDefs.hpp"

using namespace Eigen;

/**************************************
 *
 * multivariate smooth
 *
 **************************************/

template <typename Derived>
void computeGradHtmpSm(Eigen::ArrayXd& G, vecMatXd& Htmp, const Eigen::ArrayXd& lambda, const Eigen::MatrixBase<Derived>& oldFixeds, const int& ind, const vecMatXd& SList)
{

  // compute grad w.r.t rho and prepare SInv * Sj for computation of -H

  const int dim(G.size());

  // compute the pseudo inverse S^-
  MatrixXd SInv(lambda(0) * SList[ind]);
  for(int j(1); j<dim; ++j){
    SInv += lambda(j) * SList[ind+j];
  };

  const int p(SInv.rows());
  BDCSVD<MatrixXd> svd(p, p);
  svd.compute(SInv.selfadjointView<Lower>(), Eigen::ComputeFullU | Eigen::ComputeFullV);
  auto sValInv = svd.singularValues();
  const double tol(sValInv.maxCoeff() * sqrTol);
  int r(0); // rank of S^-

  while( sValInv(r) > tol ){

    sValInv(r) = 1.0 / sValInv(r);
    ++r;

  };
  SInv.noalias() = svd.matrixV().leftCols(r) * sValInv.head(r).asDiagonal() * svd.matrixU().leftCols(r).transpose();

//  SelfAdjointEigenSolver<MatrixXd> eigenDec(p);
//  VectorXd eiVal(p);
//  eigenDec.compute(SInv.selfadjointView<Lower>());
//  eiVal = eigenDec.eigenvalues();

  // compute G and Htmp
  for(int j(0); j<dim; ++j){
    Htmp[j].noalias() = SInv * SList[ind+j];
    G(j) = Htmp[j].trace() - oldFixeds[j];
  };

  G *= lambda;

};


inline void computeHSm(const vecMatXd& Htmp, Eigen::MatrixXd& H, const Eigen::ArrayXd& lambda, const Eigen::ArrayXd& G)
{
  // compute negative hessian w.r.t rho

  const int dim(Htmp.size());

  // fill lower diagonal part of H
  for(int c(0); c<dim; ++c){ // iterate over columns

    for(int r(c); r<dim; ++r){ // iterate over rows

      H(r,c) = lambda(r) * lambda(c) * (Htmp[r] * Htmp[c]).trace();
    };
  };

  H.triangularView<Upper>() = H.transpose();
  H.diagonal() -= G.matrix();

};



//============ Newton--Raphson for multivariate smooths

template <typename Derived>
Eigen::ArrayXd mvSmOptim(const int& ind, const Eigen::MatrixBase<Derived>& lambda0, const Eigen::MatrixBase<Derived>& oldFixeds, const vecMatXd& SList, const int& iterMax)
  // cannot make it void and write directly in Lambda.segment as this is an rvalue...: try && on lambda
{

  const int dim(lambda0.size());

  ArrayXd G(dim);
  vecMatXd Htmp(dim);
  ArrayXd lambda(lambda0);
  computeGradHtmpSm(G, Htmp, lambda, oldFixeds, ind, SList);

  bool converged(false);
  ArrayXd oldRho(lambda.log());
  ArrayXd bestRho(oldRho);

  if ( G.matrix().lpNorm<Infinity>() < 1e-6 ){

    converged = true;

  } else {

    int iter(0);
    MatrixXd H(dim, dim);
    VectorXd delta(dim);
    std::string w;
    int k_iter; // index for step halving
    ArrayXd GTmp(dim);

    // lambda fct for step size reduction
    auto stepSizeReduc = [&] () {

      k_iter = 0;

      while( (delta.dot(GTmp.matrix()) < 0.0) and (k_iter < 21) ){

        delta *= 0.5;
        bestRho.matrix() = oldRho.matrix() + delta;
        lambda = bestRho.exp();
        computeGradHtmpSm(GTmp, Htmp, lambda, oldFixeds, ind, SList);
        ++k_iter;

      };
    };


//    const int p(H.rows());
//    SelfAdjointEigenSolver<MatrixXd> eigenDec(p);
//    VectorXd eiVal(p);
//    eigenDec.compute(H.selfadjointView<Lower>());
//    eiVal = eigenDec.eigenvalues();

    do {

      ++iter;
      computeHSm(Htmp, H, lambda, G);

      delta = H.colPivHouseholderQr().solve(G.matrix()); // Newton step
      bestRho.matrix() = oldRho.matrix() + delta;
      lambda = bestRho.exp();
      w = "No-step-reduction";

      // step halve if the new rho has not increased the E-step
      computeGradHtmpSm(GTmp, Htmp, lambda, oldFixeds, ind, SList); // GTmp = G(rho+delta)
      if(delta.dot(GTmp.matrix()) < 0.0){

        // Newton step halving
        stepSizeReduc(); //  Newton reduction
        w = "Newton";

        // if Newton step halving failed try steepest ascent
        if(delta.dot(GTmp.matrix()) < 0.0){

          delta = G;
          bestRho.matrix() = oldRho.matrix() + delta; //HQR.solve(G.matrix());
          lambda = bestRho.exp();
          computeGradHtmpSm(GTmp, Htmp, lambda, oldFixeds, ind, SList);
          stepSizeReduc(); // end of steepest ascent
          w = "Steep";
        };

        if(delta.dot(GTmp.matrix()) < 0.0){

          std::cerr << "mvSmOptim did not improve after step halving";
        };

      }; // end of step halving


      G = GTmp; // step has improved the E-step
      //std::cout << "GRAD SMOO " << G.matrix().lpNorm<Infinity>() << std::endl;
      //std::cout << "lambda " << lambda.matrix().transpose() << std::endl;

      if ( G.matrix().lpNorm<Infinity>() < 1e-6 ){

        converged = true;

      } else {

        oldRho = bestRho;

      };

    } while( (!converged) and (iter < iterMax) );
  };

  if(!converged) {
    std::cerr << "mvSmOptim did not converge";
  };

  return lambda;

};

#endif /* SMOOTHOPTIM_HPP_ */
