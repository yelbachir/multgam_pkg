#include <cmath>

#include "../inst/include/FamilyGpd.hpp"
#include "../inst/include/commonDefs.hpp"


using namespace Eigen;

/**********
 *
 * log-lik
 *
 **********/
double logLikGpd(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
{
  // utility variables
  double tau(thetai(0)), xi(thetai(1)); // params of the GPD model
  double z(yi(0)*std::exp(-tau)), xi1(1.0/xi);
  
  if(yi(0) < 0.0){
	  return -infiniteLL;
  } else {
	  if(std::abs(xi) <= std::pow(MachinePrec, 0.3)){
		  
		  return -tau-z;
		  
		  } else if( (xi>0.0) or ((xi<0.0) and (yi(0) < -std::exp(tau)*xi1)) ){
		  
				return -tau-std::log1p(z*xi)*(xi1+1.0);
		  
		  } else {
			  return -infiniteLL;
			  };
	  };
}

/************************
 *
 * gradient in vector form
 *
 *************************/
VectorXd Deriv1iGpd(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
{
 /* output for pars = (tau, xi)
  * vector (ltau, lxi)
  */

  double tau(thetai(0)), xi(thetai(1)); // params of the model
  double z(yi(0)*std::exp(-tau));

  if(std::abs(xi) <= std::pow(MachinePrec, 0.3)){ // derivative components w.r.t thetai when xi==0

    VectorXd out(2);
    out(0) = z-1.0;
    out(1) = z*(.5*z-1.0);
    
    return out;

  } else { // derivative components w.r.t theta when xi !=0

    VectorXd out(2);
    double L(std::log1p(z*xi)), c((1.0+xi)*yi(0)*std::exp(-tau-L));
    out(0) = c-1.0;
    
    double xi1(1.0/xi);
    out(1) = (L*xi1-c)*xi1;

    return out;

  };
}

/**************************************
 *
 * 2nd derivatives in lower triangular
 * vector form
 *
 **************************************/
VectorXd Deriv2iGpd(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
{
 /* output for pars = (tau, xi)
  * vector (ltautau, ltauxi, lxixi)
  */

  double tau(thetai(0)), xi(thetai(1)); // params of the model
  double z(yi(0)*std::exp(-tau));

  if(std::abs(xi) <= std::pow(MachinePrec, 0.3)){ // derivative components w.r.t thetai when xi==0

    VectorXd out(3);
    out(0) = -z;
    out(1) = z*(1.0-z);
    out(2) = z*z*(1.0-2.0*z/3.0);

    return out;

  } else { // derivative components w.r.t theta when xi !=0

    VectorXd out(3);
    
    double a((xi+1.0)), L(std::log1p(z*xi)), f(-tau-L), b(yi(0)*yi(0)*std::exp(2.0*f)), c(yi(0)*std::exp(f));
    out(0) = a*(xi*b-c);
    out(1) = c-a*b;
    
    double xi1(1.0/xi);
    out(2) = 2.0*(c-L*xi1)*(xi1*xi1) + (1.0+xi1)*b;
    
    return out;

  };
}

/**************************************
 *
 * 3rd derivatives in matrix
 * lower triangular vector form
 *
 **************************************/

MatrixXd Deriv3iGpd(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
{
 /* returns a 0.5*D*(D+1)xD (where D is the dimension of thetai) matrix containing the lower triangular part of
  * the matrices of the 3rd derivatives of the loglik w.r.t theta
  * d-th column corresponds to the lower triangular part of dHi/dTheta_i^{d}
  */

 /* output for pars = (tau, xi)
  * col 1 = (tautautau, tautauxi, tauxixi),
  * col 2 = (tautauxi, tauxixi, xixixi),
  */

  double tau(thetai(0)), xi(thetai(1)); // params of the model
  double z(yi(0)*std::exp(-tau));


  if(std::abs(xi) <= std::pow(MachinePrec,0.3)){ // derivative components w.r.t thetai when xi==0
    
    MatrixXd out(3,2);
    out(0,0) = z;
    out(1,0) = z*(2.0*z-1.0);
    
    double z2(z*z);
    out(2,0) = 2.0*z2*(z-1.0);

    out(0,1) = out(1,0);
    out(1,1) = out(2,0);
    out(2,1) = z*z2*(1.5*z-2.0);

    return out;

  } else { // derivative components w.r.t theta when xi !=0

    MatrixXd out(3,2);
    
    double d(1.0+xi), L(std::log1p(z*xi)), ff(-tau-L), c(yi(0)*std::exp(ff)), y2(yi(0)*yi(0)), a(y2*yi(0)*std::exp(3.0*ff));
    double g(2.0*xi), b(y2*std::exp(2.0*ff));
    out(0,0) = d*(c + xi*(a*g-3.0*b) );
    
    double f(a*d);
    out(1,0) = b*(2.0+3.0*xi) -c -f*g;
    out(2,0) = 2.0*(f-b);

    out(0,1) = out(1,0);
    out(1,1) = out(2,0);
    
    double xi1(1.0/xi);
    out(2,1) = ( 6.0*(L*xi1 - c)*xi1 - 3.0*b )*(xi1*xi1) - 2.0*a*(1.0+xi1);
    
    return out;

  };
}



/******************
 *
 * beta initialization
 *
 ********************/
 VectorXd betaInitGpd(const Map<MatrixXd>& yi, const Map<MatrixXi>& pFull)
{
  // compute beta_0 = (tau0, 0...0, xi0, 0...0)

  double ymean(yi.mean());
  double yvar(((yi.array() - ymean).matrix().squaredNorm()) /(yi.rows()-1.0));
  double s(std::sqrt(6.0*yvar)/Pi);

  VectorXd out(Eigen::VectorXd::Zero(pFull.row(1).sum()));
  out(0) = std::log(s); //  out(0) = .5*std::log(yvar); // tau0
  out(pFull(0,1)) = 0.1; // xi0

  return out;
}
