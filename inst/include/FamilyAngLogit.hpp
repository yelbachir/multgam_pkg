#ifndef FAMILYANGLOGIT_HPP_
#define FAMILYANGLOGIT_HPP_

#include <RcppEigen.h>
#include <cmath>

using namespace Eigen;

/******************
 *
 * utility functions
 *
 ********************/
inline double yPowExp(const double& y, const double& tmp)
{
  // computation of y^(1+exp(-alpha))

  return std::exp(tmp*std::log(y));
};

/**********
 *
 * log-lik
 *
 **********/
inline double logLikAngLogit(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
{
  // utility variables
  double alpha(thetai(0)); // parameter of the symmetric logistic model
  double OneMy(1-yi(0)), tmp(1+std::exp(-alpha));

  return -alpha + std::log(yi(0)*OneMy)*std::expm1(-alpha) - (1+1/(1+std::exp(alpha)))*std::log(yPowExp(yi(0), tmp) + yPowExp(OneMy, tmp));
};

VectorXd Deriv1iAngLogit(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai);
VectorXd Deriv2iAngLogit(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai);
MatrixXd Deriv3iAngLogit(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai);

/******************
 *
 * beta initialization
 *
 ********************/

inline VectorXd betaInitAngLogit(const Map<MatrixXd>& yi, const Map<MatrixXi>& pFull)
{
  // compute beta_0 = (alpha0, 0...0)

  VectorXd out(VectorXd::Zero(pFull.row(1).sum()));

  return out;
};

#endif /* FAMILYANGLOGIT_HPP_ */
