#ifndef FAMILYGAUSS_HPP_
#define FAMILYGAUSS_HPP_

#include <RcppEigen.h>
#include <cmath>

#include "commonDefs.hpp" // for pi = 3.14...

using namespace Eigen;

const double log2pi(std::log(2*Pi));

/**********
 *
 * log-lik
 *
 **********/

inline double logLikGauss(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
{
  //double mu(thetai(0)), tau(thetai(1)); // params of the Gauss model
  double tmp(yi(0)-thetai(0));

  return -0.5*(log2pi + thetai(1) + tmp*tmp*std::exp(-thetai(1)));
};

/************************
 *
 * gradient in vector form
 *
 *************************/
inline VectorXd Deriv1iGauss(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
{
  // lmu, ltau

  //double mu(thetai(0)), tau(thetai(1)), xi(thetai(2)); // params of the model
  double a(yi(0)-thetai(0));

  Eigen::VectorXd out(2);
  out(0) = a*std::exp(-thetai(1));
  out(1) = 0.5*(out(0)*a -1);

  return out;
}

/**************************************
 *
 * 2nd derivatives in lower triangular
 * vector form
 *
 **************************************/
inline VectorXd Deriv2iGauss(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
{ // returns vector of  lmumu, lmutau, ltautau

  //double mu(thetai(0)), tau(thetai(1)); // params of the model
  Eigen::VectorXd out(3);
  out(0) = -std::exp(-thetai(1));

  double a(yi(0)-thetai(0));
  out(1) = a*out(0);
  out(2) = 0.5*a*out(1);

  return out;
}

/**************************************
 *
 * 3rd derivatives in matrix lower triangular vector form
 *
 **************************************/
inline MatrixXd Deriv3iGauss(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
{
  /* returns a 0.5*D*(D+1)xD matrix containing the lower triangular part of
  * the matrices of the 3rd derivatives of the loglik w.r.t theta
  * d-th column corresponds to the lower triangular part of dHi/dTheta_i^{d}
  */

  /* output
  * mumumu, mumutau, mutautau
  * mumutau, mutautau, tautautau
  */

  //double mu(thetai(0)), tau(thetai(1)), xi(thetai(2)); // params of the model
  double a(yi(0)-thetai(0));

  Eigen::MatrixXd out(3,2);
  out(0,0) = 0.0;
  out(1,0) = std::exp(-thetai(1));
  out(2,0) = a*out(1,0);

  out(0,1) = out(1,0);
  out(1,1) = out(2,0);
  out(2,1) = a*out(2,0);

  return out;
}


/******************
 *
 * beta initialization
 *
 ********************/

inline VectorXd betaInitGauss(const Map<MatrixXd>& y, const Map<MatrixXi>& pFull)
{
  // compute beta_0 = (mu0, 0...0, tau0, 0...0) in R pxp

  double ymean(y.mean());
  double yvar((y.array() - ymean).matrix().squaredNorm() /(y.rows()-1));

  Eigen::VectorXd out(Eigen::VectorXd::Zero(pFull.row(1).sum()));

  out(0) = ymean; // mu0
  out(pFull(0,1)) = std::log(yvar); // tau0

  return out;
};

#endif /* FAMILYGAUSS_HPP_ */

