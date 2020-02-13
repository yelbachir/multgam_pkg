#ifndef FAMILYGAMMA_HPP_
#define FAMILYGAMMA_HPP_

#include <RcppEigen.h>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>
#include <boost/math/special_functions/polygamma.hpp>
#include <cmath>


using namespace Eigen;

/**********
 *
 * log-lik
 *
 **********/

inline double logLikGamma(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
{
  //double tau(thetai(0)), gamma(thetai(1)); // params of the Gamma(tau=log(alpha), gamma=log(beta)) model
  double alpha(std::exp(thetai(0)));
  
  return alpha * (std::log(yi(0)) + thetai(1)) - yi(0) * std::exp(thetai(1)) - std::lgamma(alpha) - std::log(yi(0));
};

/************************
 *
 * gradient in vector form
 *
 *************************/
inline VectorXd Deriv1iGamma(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
{
  // ltau, lgamma
  double alpha(std::exp(thetai(0)));

  Eigen::VectorXd out(2);
  out(0) = alpha * (std::log(yi(0)) + thetai(1) - boost::math::digamma(alpha));
  out(1) = alpha - yi(0) * std::exp(thetai(1));

  return out;
}

/**************************************
 *
 * 2nd derivatives in lower triangular
 * vector form
 *
 **************************************/
inline VectorXd Deriv2iGamma(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
{ 
  // returns vector of ltautau, ltaugamma, lgammagamma
  double alpha(std::exp(thetai(0)));
  
  Eigen::VectorXd out(3);
  out(0) = alpha * (std::log(yi(0)) + thetai(1) - boost::math::digamma(alpha) - alpha * boost::math::trigamma(alpha));
  out(1) = alpha;
  out(2) = - yi(0) * std::exp(thetai(1));

  return out;
}

/**************************************
 *
 * 3rd derivatives in matrix lower triangular vector form
 *
 **************************************/
inline MatrixXd Deriv3iGamma(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
{
 /* returns a 0.5*D*(D+1)xD matrix containing the lower triangular part of
  * the matrices of the 3rd derivatives of the loglik w.r.t theta
  * d-th column corresponds to the lower triangular part of dHi/dTheta_i^{d}
  */

 /* output
  * col 1 = tautautau, tautaugamma, taugammagamma
  * col 2 = tautaugamma, taugammagamma, gammagammagamma
  */

  double alpha(std::exp(thetai(0)));

  Eigen::MatrixXd out(3,2);
  out(0,0) = alpha * (std::log(yi(0)) + thetai(1) - boost::math::digamma(alpha) - 3 * alpha * boost::math::trigamma(alpha) - alpha * alpha * boost::math::polygamma(2, alpha));
  out(1,0) = alpha;
  out(2,0) = 0.0;

  out(0,1) = alpha;
  out(1,1) = 0.0;
  out(2,1) = -yi(0) * std::exp(thetai(1));

  return out;
}


/******************
 *
 * beta initialization
 *
 ********************/

inline VectorXd betaInitGamma(const Map<MatrixXd>& y, const Map<MatrixXi>& pFull)
{
  // compute beta_0 = (tau0, 0...0, gamma0, 0...0) in R pxp

  double ymean(y.mean());
  double yvar((y.array() - ymean).matrix().squaredNorm() /(y.rows()-1));

  Eigen::VectorXd out(Eigen::VectorXd::Zero(pFull.row(1).sum()));
  
  double temp(ymean/yvar);

  out(0) = std::log(ymean*temp); // tau0
  out(pFull(0,1)) = std::log(temp); // gamma0

  return out;
};

#endif /* FAMILYGAMMA_HPP_ */

