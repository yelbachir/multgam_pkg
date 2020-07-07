#ifndef FAMILYPP_HPP_
#define FAMILYPP_HPP_

#include <RcppEigen.h>

using namespace Eigen;

// main functions
double logLikPp(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai);
VectorXd Deriv1iPp(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai);
VectorXd Deriv2iPp(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai);
MatrixXd Deriv3iPp(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai);
VectorXd betaInitPp(const Map<MatrixXd>& yi, const Map<MatrixXi>& pFull);

#endif /* FAMILYPP_HPP_ */
