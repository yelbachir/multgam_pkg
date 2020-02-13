#ifndef FAMILYPOISS_HPP_
#define FAMILYPOISS_HPP_

#include <RcppEigen.h>
#include <cmath>
#include <boost/math/special_functions/factorials.hpp>

using namespace Eigen;

/**********
 *
 * log-lik
 *
 **********/

inline double logLikPoiss(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
{
  // theta_i = log(lambda_i), where lambda_i>0

  return yi(0)*thetai(0) - std::exp(thetai(0)) - std::log(boost::math::factorial<double>(yi(0)));
};

/************************
 *
 * gradient in vector form
 *
 *************************/
inline VectorXd Deriv1iPoiss(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
{
  // returns vector of ltheta
  
  Eigen::VectorXd out(1);
  out(0) = yi(0) - std::exp(thetai(0));

  return out;
}

/**************************************
 *
 * 2nd derivatives in lower triangular
 * vector form
 *
 **************************************/
inline VectorXd Deriv2iPoiss(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
{ // returns vector of  lthetatheta

  Eigen::VectorXd out(1);
  out(0) = -std::exp(thetai(0));
  
  return out;
}

/**************************************
 *
 * 3rd derivatives in matrix lower triangular vector form
 *
 **************************************/
inline MatrixXd Deriv3iPoiss(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
{
 /* returns a 0.5*D*(D+1)xD matrix containing the lower triangular part of
  * the matrices of the 3rd derivatives of the loglik w.r.t theta
  * d-th column corresponds to the lower triangular part of dHi/dTheta_i^{d}
  */

 /* output
  * mumumu
  */

  Eigen::MatrixXd out(1,1);
  out(0,0) = -std::exp(thetai(0));

  return out;
}


/******************
 *
 * beta initialization
 *
 ********************/

inline VectorXd betaInitPoiss(const Map<MatrixXd>& y, const Map<MatrixXi>& pFull)
{
  // compute beta_0 = (theta0, 0...0)
  
  Eigen::VectorXd out(Eigen::VectorXd::Zero(pFull.row(1).sum()));
  out(0) = std::log(y.mean()); // theta0

  return out;
};

#endif /* FAMILYPOISS_HPP_ */

