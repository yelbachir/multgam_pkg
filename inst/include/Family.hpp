#ifndef FAMILY_HPP_
#define FAMILY_HPP_


#include <RcppEigen.h> 
#include <functional> 
#include <array> 

using namespace Eigen;

// typedefs
typedef std::function<double(const Ref<const RowVectorXd>&, const Ref<const RowVectorXd>& )> FamilyLL;
typedef std::function<VectorXd(const Ref<const RowVectorXd>&, const Ref<const RowVectorXd>& )> FamilyD12;
typedef std::function<MatrixXd(const Ref<const RowVectorXd>&, const Ref<const RowVectorXd>& )> FamilyD3;
typedef std::function<VectorXd(const Map<MatrixXd>&, const Map<MatrixXi>& )> FamilyBetaInit;


struct StructFamily
{
  FamilyLL logLik;
  FamilyD12 Deriv1i;
  FamilyD12 Deriv2i;
  FamilyD3 Deriv3i;
  FamilyBetaInit betaInit;

};


const int nbFamily = 7;
extern const std::array<StructFamily, nbFamily> familyChoice;

#endif /* FAMILY_HPP_ */
