#include <cmath>

#include "../inst/include/FamilyGev.hpp"
#include "../inst/include/commonDefs.hpp"


using namespace Eigen;

/******************
 *
 * utility functions
 *
 ********************/
inline double fexp(const double& a, const double& b, const double& e)
{
  // used in the computation of exp(a+b*log1p(z)), where e=log1p(z)

  return std::exp(a+b*e);
}

/**********
 *
 * log-lik
 *
 **********/
double logLikGev(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
{
  // utility variables
  double mu(thetai(0)), tau(thetai(1)), xi(thetai(2)); // params of the GEV model
  double z((yi(0)-mu)*std::exp(-tau));

  if(std::abs(xi) <= std::pow(MachinePrec, 0.3)){

    return -tau-z-std::exp(-z);

  } else {

    z *= xi;
    //			if( ((1+z) < sqrTol) or (xi < -1) ){
    if((1.0+z) < sqrTol){
      return -infiniteLL;

    } else {
      double L(std::log1p(z)), xi1(1.0/xi);
      return -tau - L*(xi1 + 1.0) - std::exp(-L*xi1);
    };
  };
}


/************************
 *
 * gradient in vector form
 *
 *************************/
VectorXd Deriv1iGev(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
{
  /* output for pars = (mu, tau, xi)
  * vector (lmu, ltau, lxi)
  */

  double mu(thetai(0)), tau(thetai(1)), xi(thetai(2)); // params of the model
  double a(yi(0)-mu), b(a*std::exp(-tau));

  if(std::abs(xi) <= std::pow(MachinePrec, 0.3)){ // derivative components w.r.t thetai when xi==0

    // utility variables
    double e(std::expm1(-b));

    VectorXd out(3);
    out(0) = -e*std::exp(-tau);
    out(1) = a*out(0)-1.0;
    out(2) = -b*(1.0+.5*e*b);

    return out;

  } else { // derivative components w.r.t theta when xi !=0

    //utility variables
    double L(std::log1p(b*xi)), xi1(1.0/xi);
    double e(std::expm1(-L*xi1));
    double exi1(e*xi1), f1(fexp(-tau,-1.0,L));

    VectorXd out(3);
    out(0) = -f1*(e-xi);
    out(1) = a*out(0) -1.0;
    out(2) = a*f1*(exi1-1.0)-L*exi1*xi1;

    return out;

  };
}

/**************************************
 *
 * 2nd derivatives in lower triangular
 * vector form
 *
 **************************************/
VectorXd Deriv2iGev(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
{
  /* output for pars = (mu, tau, xi)
  * vector (lmumu, lmutau, lmuxi, ltautau, ltauxi, lxixi)
  */

  double mu(thetai(0)), tau(thetai(1)), xi(thetai(2)); // params of the model
  double a(yi(0)-mu), b(a*std::exp(-tau));

  if(std::abs(xi) <= std::pow(MachinePrec, 0.3)){ // derivative components w.r.t thetai when xi==0

    // utility variables
    double e(std::expm1(-b)), et(std::exp(-tau)), eb(std::exp(-b)), b2(b*b);

    VectorXd out(6);
    out(0) = -std::exp(-2.0*tau-b);
    out(1) = a*out(0)+e*et;
    out(2) = et*(b*(e-.5*b*eb)+1.0);
    out(3) = a*out(1);
    out(4) = a*out(2);
    out(5) = b2*(1.0-eb*.25*b2+2.0*e*b/3);

    return out;

  } else { // derivative components w.r.t theta when xi !=0

    //utility variables
    double xi1(1.0/xi), L(std::log1p(b*xi));
    double e(std::expm1(-L*xi1));
    double exi1(e*xi1), f1(fexp(-tau,-1.0,L)), f2(fexp(-2.0*tau,-2.0,L)), f0(fexp(0,-xi1,L));
    double f0xi1(f0*xi1), Lf0xi1(L*f0xi1), Lf0xi12(Lf0xi1*xi1), lmu(-f1*(e-xi));

    VectorXd out(6);
    out(0) = (xi+1.0)*f2*(xi-f0);
    out(1) = a*out(0) -lmu;
    out(2) = a*f2*(e + f0xi1-xi) + f1*(1.0-Lf0xi12);
    out(3) = a*(out(0)*a - lmu);
    out(4) = a*out(2);
    out(5) = f2*(1.0-f0xi1*xi1-exi1)*a*a + (2.0*L*exi1 + 2.0*a*f1*(Lf0xi1 - e) - L*Lf0xi12)*xi1*xi1;

    return out;

  };
}

/**************************************
 *
 * 3rd derivatives in matrix
 * lower triangular vector form
 *
 **************************************/

MatrixXd Deriv3iGev(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
{
  /* returns a 0.5*D*(D+1)xD (where D is the dimension of thetai) matrix containing the lower triangular part of
  * the matrices of the 3rd derivatives of the loglik w.r.t theta
  * d-th column corresponds to the lower triangular part of dHi/dTheta_i^{d}
  */

  /* output for pars = (mu, tau, xi)
  * col 1 = (mumumu, mumutau, mumuxi, mutautau, mutauxi, muxixi),
  * col 2 = (mumutau, mutautau, mutauxi, tautautau, tautauxi, tauxixi),
  * col 3 = (mumuxi, mutauxi, muxixi, tautauxi, tauxixi, xixixi)
  */

  double mu(thetai(0)), tau(thetai(1)), xi(thetai(2)); // params of the model
  double a(yi(0)-mu), b(a*std::exp(-tau));


  if(std::abs(xi) <= std::pow(MachinePrec,0.3)){ // derivative components w.r.t thetai when xi==0

    // utility variables
    double e(std::expm1(-b)), et(std::exp(-tau)), eb(std::exp(-b)), b2(b*b);
    double beb(b*eb), b3(b*b2), lmumu(-std::exp(-2*tau-b));

    MatrixXd out(6,3);
    out(0,0) = -std::exp(-3.0*tau-b);
    out(1,0) = a*out(0,0)-2.0*lmumu;
    out(2,0) = std::exp(-2.0*tau)*(-e+beb*(2.0-.5*b));
    out(3,0) = out(0,0)*a*a -3.0*a*lmumu -e*et;
    out(4,0) = a*out(2,0)-et*(b*(e-.5*beb)+1.0);
    out(5,0) = b*et*(2.0*(b-1.0)+beb*(-2.0+5.0*b/3.0-.25*b2));

    out(0,1) = out(1,0);
    out(1,1) = out(3,0);
    out(2,1) = out(4,0);
    out(3,1) = a*out(3,0);
    out(4,1) = a*out(4,0);
    out(5,1) = a*out(5,0);

    out(0,2) = out(2,0);
    out(1,2) = out(4,0);
    out(2,2) = out(5,0);
    out(3,2) = out(4,1);
    out(4,2) = out(5,1);
    out(5,2) = (2.0 - 1.5*e*b + (b2 -b3/8.0)*eb )*b3;

    return out;

  } else { // derivative components w.r.t theta when xi !=0

    //utility variables
    double L(std::log1p(b*xi)), xi1(1/xi), f0(fexp(0,-xi1,L)), f1(fexp(-tau,-1,L)), f2(fexp(-2*tau,-2,L));
    double f3(fexp(-3*tau,-3,L)), Lxi1(L*xi1), e(std::expm1(-Lxi1)), f0xi1(f0*xi1), xi12(xi1*xi1);
    double Lf0xi1(L*f0xi1), a2(a*a), u(xi+1), v(1+xi1), lmumu(u*f2*(xi-f0));

    MatrixXd out(6,3);
    out(0,0) = (2.0*xi*lmumu*f1) - u*fexp(-3.0*tau,-3.0-xi1,L);
    out(1,0) = a*out(0,0)-2.0*lmumu;
    out(2,0) = a*f3*(2.0*xi*(e-xi) + (3.0+xi1)*f0) + f2*(2.0*xi-e-v*Lf0xi1);
    out(3,0) = out(0,0)*a2 -3.0*a*lmumu -f1*(e-xi);
    out(4,0) = a*out(2,0) -a*f2*(e + f0xi1-xi) - f1*(1.0-Lxi1*f0xi1);
    out(5,0) = (a*f3*(2.0*u-v*(1.0+v)*f0) + 2.0*f2*(f0*(L*v-1.0)*xi12-1.0))*a + (2.0-Lxi1)*Lxi1*xi12*fexp(-tau,-1.0-xi1,L);

    out(0,1) = out(1,0);
    out(1,1) = out(3,0);
    out(2,1) = out(4,0);
    out(3,1) = a*out(3,0);
    out(4,1) = a*out(4,0);
    out(5,1) = a*out(5,0);

    out(0,2) = out(2,0);
    out(1,2) = out(4,0);
    out(2,2) = out(5,0);
    out(3,2) = out(4,1);
    out(4,2) = out(5,1);
    out(5,2) = ( (e*2.0*xi1 + f0*(3.0+xi1)*xi12 -2.0)*f3*a + (e+(2.0-L*v)*f0xi1)*3.0*xi12*f2 )*a2 + (3.0*a*f1*(2.0*e+(Lxi1-4.0)*Lf0xi1) + ( (6.0-Lxi1)*Lf0xi1 -6.0*e)*Lxi1 )*xi12*xi1;

    return out;

  };
}



/******************
 *
 * beta initialization
 *
 ********************/

VectorXd betaInitGev(const Map<MatrixXd>& yi, const Map<MatrixXi>& pFull)
{
  // compute beta_0 = (mu0, 0...0, tau0, 0...0, xi0, 0...0)

  double ymean(yi.mean());
  double yvar(((yi.array() - ymean).matrix().squaredNorm()) /(yi.rows()-1.0));
  double s(std::sqrt(6.0*yvar)/Pi); // se(X) when xi == 0

  VectorXd out(Eigen::VectorXd::Zero(pFull.row(1).sum()));
  out(0) = ymean - s*0.5772157; // mu0
  out(pFull(0,1)) = std::log(s); // tau0
  out(pFull(0,2)) = 0.1; // xi0

  return out;
}
