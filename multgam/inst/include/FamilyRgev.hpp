#ifndef FAMILYRGEV_HPP_
#define FAMILYRGEV_HPP_

#include <RcppEigen.h>

using namespace Eigen;

// utility functions
inline double fexpRgev(const double& a, const double& b, const double& e);

// GEV derivative functions
VectorXd Deriv1iG(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai);
VectorXd Deriv2iG(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai);
MatrixXd Deriv3iG(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai);

// main functions
double logLikRgev(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai);
VectorXd Deriv1iRgev(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai);
VectorXd Deriv2iRgev(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai);
MatrixXd Deriv3iRgev(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai);
VectorXd betaInitRgev(const Map<MatrixXd>& yi, const Map<MatrixXi>& pFull);

#endif /* FAMILYRGEV_HPP_ */
