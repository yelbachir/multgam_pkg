#ifndef FAMILYGEV_HPP_
#define FAMILYGEV_HPP_

#include <RcppEigen.h>

using namespace Eigen;
// utility functions
inline double fexp(const double& a, const double& b, const double& e);

// main functions
double logLikGev(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai);
VectorXd Deriv1iGev(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai);
VectorXd Deriv2iGev(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai);
MatrixXd Deriv3iGev(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai);
VectorXd betaInitGev(const Map<MatrixXd>& yi, const Map<MatrixXi>& pFull);

#endif /* FAMILYGEV_HPP_ */
