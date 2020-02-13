#include "../inst/include/FamilyAngLogit.hpp"

/************************
 *
 * derivative w.r.t. the unique dependence parameter alpha
 *
 *************************/
VectorXd Deriv1iAngLogit(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
{
  /* output for parameter = (alpha)
  * vector (lalpha)
  */

  // utility variables
  double alpha(thetai(0)); // parameter of the symmetric logistic model
  double expA(std::exp(-alpha)), OneMy(1-yi(0)), tmpInv(1/(1+std::exp(alpha)));
  double tmp(1+expA), tmp0(yPowExp(yi(0), tmp)), tmp1(yPowExp(OneMy, tmp)), tmp2(tmp0 + tmp1);

  Eigen::VectorXd out(1);

  out(0) = -1 - expA*std::log(yi(0)*OneMy) + (tmpInv - tmpInv*tmpInv)*std::log(tmp2) + (1+tmpInv)*expA*(tmp0*std::log(yi(0)) +  tmp1*std::log(OneMy))/tmp2;

  return out;
};

/**************************************************************
 *
 * 2nd derivative w.r.t. the unique dependence parameter alpha
 *
 **************************************************************/
VectorXd Deriv2iAngLogit(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
{
  /* output for parameter = (alpha)
  * vector (lalpha)
  */

  // utility variables
  double alpha(thetai(0)); // parameter of the symmetric logistic model
  double expA(std::exp(-alpha)), OneMy(1-yi(0)), tmpInv(1/(1+std::exp(alpha))), tmpInv2(tmpInv*tmpInv);
  double tmp(1+expA), tmp0(yPowExp(yi(0), tmp)), tmp1(yPowExp(OneMy, tmp)), tmp2(tmp0 + tmp1);
  double logi(std::log(yi(0))), tmp0log(tmp0*logi), logMi(std::log(OneMy)), tmp1log(tmp1*logMi);
  double logPrim(-expA*(tmp0log + tmp1log)/tmp2), log2Prim(expA*(tmp0log*(1+expA*logi) + tmp1log*(1+expA*logMi))/tmp2 - logPrim*logPrim);

  Eigen::VectorXd out(1);

  out(0) = expA*std::log(yi(0)*OneMy) - (tmpInv - 3*tmpInv2 + 2*tmpInv*tmpInv2)*std::log(tmp2) + 2*(tmpInv - tmpInv2)*logPrim - (1+tmpInv)*log2Prim;

  return out;
};


/**************************************************************
 *
 * 3rd derivative w.r.t. the unique dependence parameter alpha
 *
 **************************************************************/
MatrixXd Deriv3iAngLogit(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
{
  /* output for parameter = (alpha)
  * vector (lalpha)
  */

  // utility variables
  double alpha(thetai(0)); // parameter of the symmetric logistic model
  double OneMy(1-yi(0)), expA(std::exp(-alpha)), tmpInv(1/(1+std::exp(alpha))), tmpInv2(tmpInv*tmpInv), tmpInv3(tmpInv*tmpInv2);
  double tmp(1+expA), tmp0(yPowExp(yi(0), tmp)), tmp1(yPowExp(OneMy, tmp)), tmp2(tmp0 + tmp1);
  double logi(std::log(yi(0))), tmp0log(tmp0*logi), logMi(std::log(OneMy)), tmp1log(tmp1*logMi);
  double logPrim(-expA*(tmp0log + tmp1log)/tmp2), log2Ratio(expA*(tmp0log*(1+expA*logi) + tmp1log*(1+expA*logMi))/tmp2), log2Prim(log2Ratio - logPrim*logPrim);
  double e2(expA*expA), e3(expA*e2), log3Prim(-(tmp0log*(3*e2*logi + expA + e3*logi*logi) + tmp1log*(3*e2*logMi + expA + e3*logMi*logMi))/tmp2 - logPrim*(log2Ratio + 2*log2Prim));

  Eigen::MatrixXd out(1,1);

  out(0,0) = -expA*std::log(yi(0)*OneMy) + (tmpInv -7*tmpInv2 +12*tmpInv3 - 6*tmpInv3*tmpInv)*std::log(tmp2) - 3*(tmpInv -3*tmpInv2 + 2*tmpInv3)*logPrim + 3*(tmpInv - tmpInv2)*log2Prim - (1+tmpInv)*log3Prim;

  return out;
};
