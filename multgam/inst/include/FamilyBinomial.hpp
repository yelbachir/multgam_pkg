#ifndef FAMILYBINOM_HPP_
#define FAMILYBINOM_HPP_

#include <RcppEigen.h>
#include <cmath>
#include <boost/math/special_functions/binomial.hpp>

#include "./commonFunctions.hpp"
#include "./commonVarFamily.hpp" // extra argument, i.e. n, to family binomial

using namespace Eigen;

/* original model : Binomial(n, p) model in terms of p = exp(tau)/(1+exp(tau)), so that thetai(0) = tau
 * parametrization : Binomial(n, tau), where tau has gam structure
 */

/**********
 *
 * log-lik
 *
 **********/
inline double logLikBinom(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
{
  // parameter of the binomial(n, p=exp(tau)/(1+exp(tau))) model where thetai(0)=tau

  return thetai(0) * yi(0) - nSizeBinom * LogOnePlusExpX(thetai(0)) + std::log( boost::math::binomial_coefficient<double>(nSizeBinom, yi(0)));
};

/************************
 *
 * derivative w.r.t. p
 *
 *************************/
inline VectorXd Deriv1iBinom(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
{
 // ltau

  Eigen::VectorXd out(1);

  out(0) = yi(0) - nSizeBinom * LogitInverse(thetai(0));

  return out;
};


/**************************************************************
 *
 * 2nd derivative w.r.t. p
 *
 **************************************************************/
inline VectorXd Deriv2iBinom(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
{
  // ltautau

  Eigen::VectorXd out(1);
  double tmp(LogitInverse(thetai(0)));
  
  out(0) = -nSizeBinom * tmp * (1 - tmp); // as logitInv'=logitInv(1-logitInv)

  return out;
};

/**************************************************************
 *
 * 3rd derivative w.r.t. to p
 *
 **************************************************************/
inline MatrixXd Deriv3iBinom(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
{
  // ltautautau

  Eigen::MatrixXd out(1,1);
  
  double tmp(LogitInverse(thetai(0)));

  out(0,0) = nSizeBinom * tmp * (tmp - 1) * (1 - 2 * tmp);

  return out;
};

/******************
 *
 * beta initialization
 *
 ********************/

inline VectorXd betaInitBinom(const Map<MatrixXd>& yi, const Map<MatrixXi>& pFull)
{
  // compute beta_0 = (tau0, 0...0)

  VectorXd out(VectorXd::Zero(pFull.row(1).sum()));
  
  double p(yi.mean()/yi.rows());
  
  out(0) = Logit(p);
  
  return out;
};

#endif /* FAMILYBINOM_HPP_ */
