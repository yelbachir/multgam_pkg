#ifndef FAMILYGPD_HPP_
#define FAMILYGPD_HPP_

#include <RcppEigen.h>

using namespace Eigen;

// main functions
double logLikGpd(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai);
VectorXd Deriv1iGpd(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai);
VectorXd Deriv2iGpd(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai);
MatrixXd Deriv3iGpd(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai);
VectorXd betaInitGpd(const Map<MatrixXd>& yi, const Map<MatrixXi>& pFull);

#endif /* FAMILYGPD_HPP_ */
