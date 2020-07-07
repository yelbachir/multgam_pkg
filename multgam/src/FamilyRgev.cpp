#include <cmath>

#include "../inst/include/FamilyRgev.hpp"
#include "../inst/include/commonDefs.hpp"


using namespace Eigen;

/******************
 *
 * utility functions
 *
 ********************/
inline double fexpRgev(const double& a, const double& b, const double& e)
{
  // used in the computation of exp(a+b*log1p(z)), where e=log1p(z)

  return std::exp(a+b*e);
}

/**********
 *
 * log-lik
 *
 **********/
double logLikRgev(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
{
  // utility variables
  double mu(thetai(0)), tau(thetai(1)), xi(thetai(2)); // params of the GEV model
  double etau(std::exp(-tau)), z((yi(0)-mu)*etau);
  const int r(yi.size());

  if(std::abs(xi) <= std::pow(MachinePrec, 0.3)){
	  
	  return -std::exp(-z)-r*tau-etau*(yi.sum()-r*mu);

  } else {
	  
	  const int r1(r-1);
	  double b(xi*etau), z1((yi(r1)-mu)*b);
	  z *= xi;
	  if( ((xi < 0.0) and ((1.0+z1) > sqrTol)) or ((xi > 0.0) and ((1.0+z) > sqrTol) ) ){
		  
		  double Lj(std::log1p(z)), out(Lj + std::log1p(z1));
		  for(int j(1); j<r1; ++j){  
			  out += std::log1p((yi(j)-mu)*b);
			  };
		   double xi1(1.0/xi);
		   return -out*(1.0+xi1) - r*tau - std::exp(-Lj*xi1);
			  
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
VectorXd Deriv1iG(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
{
 /* output for pars = (mu, tau, xi)
  * vector (lmu, ltau, lxi)
  */

  double mu(thetai(0)), tau(thetai(1)), xi(thetai(2)); // params of the model
  double a(yi(0)-mu), b(a*std::exp(-tau));

  if(std::abs(xi) <= std::pow(MachinePrec, 0.3)){ // derivative components w.r.t thetai when xi==0

    // utility variables
    double e(std::expm1(-b));

    Eigen::VectorXd out(3);
    out(0) = -e*std::exp(-tau);
    out(1) = a*out(0)-1.0;
    out(2) = -b*(1.0+.5*e*b);

    return out;

  } else { // derivative components w.r.t theta when xi !=0

    //utility variables
    double L(std::log1p(b*xi)), xi1(1.0/xi);
    double e(std::expm1(-L*xi1));
    double exi1(e*xi1), f1(fexpRgev(-tau,-1.0,L));

    VectorXd out(3);
    out(0) = -f1*(e-xi);
    out(1) = a*out(0) -1.0;
    out(2) = a*f1*(exi1-1.0)-L*exi1*xi1;

    return out;

  };
}

VectorXd Deriv1iRgev(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
{
 /* output for pars = (mu, tau, xi)
  * vector (lmu, ltau, lxi)
  */
  
  double mu(thetai(0)), tau(thetai(1)), xi(thetai(2)); // params of the model
  double etau(std::exp(-tau));
  const int r(yi.size()), r1(r-1);
  
  if(std::abs(xi) <= std::pow(MachinePrec, 0.3)){ // derivative components w.r.t thetai when xi==0
    
    // utility variables
    double aj(0.0), Saj(0.0), Saj2(0.0);
    for(int j(1); j<r; ++j){
		aj = yi(j)-mu;
		Saj += aj;
		Saj2 += aj*aj;
		};
		
	VectorXd out(Deriv1iG(yi, thetai));
    out(0) += r1*etau;
    Saj *= etau;
    out(1) += Saj - r1;
    out(2) += 0.5*std::exp(-2.0*tau)*Saj2 - Saj;

    return out;

  } else { // derivative components w.r.t theta when xi !=0
    
    //utility variables
    double b(xi*etau), aj(0.0), Lj(0.0), SLj(0.0), ej(0.0), Sej(0.0), Sajej(0.0);
    for(int j(1); j<r; ++j){
		aj = yi(j)-mu;
		Lj = std::log1p(aj*b);
		SLj += Lj;
		ej = std::exp(-tau-Lj);
		Sej += ej;
		Sajej += aj*ej;
		};

	double xip1(xi+1.0), xi1(1.0/xi);
    VectorXd out(Deriv1iG(yi, thetai));
    out(0) += xip1*Sej;
    out(1) += xip1*Sajej - r1;
    out(2) += SLj*xi1*xi1 - (1.0+xi1)*Sajej;

    return out;

  };
}

/**************************************
 *
 * 2nd derivatives in lower triangular
 * vector form
 *
 **************************************/
VectorXd Deriv2iG(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
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
    double exi1(e*xi1), f1(fexpRgev(-tau,-1.0,L)), f2(fexpRgev(-2.0*tau,-2.0,L)), f0(fexpRgev(0,-xi1,L));
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

VectorXd Deriv2iRgev(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
{
 /* output for pars = (mu, tau, xi)
  * vector (lmumu, lmutau, lmuxi, ltautau, ltauxi, lxixi)
  */

  double mu(thetai(0)), tau(thetai(1)), xi(thetai(2)); // params of the model
  double etau(std::exp(-tau));
  const int r(yi.size());

  if(std::abs(xi) <= std::pow(MachinePrec, 0.3)){ // derivative components w.r.t thetai when xi==0

    // utility variables
    double aj(0.0), Saj(0.0), aj2(0.0), Saj2(0.0), Saj3(0.0);
    for(int j(1); j<r; ++j){
		aj = yi(j)-mu;
		Saj += aj;
		aj2 = aj*aj;
		Saj2 += aj2;
		Saj3 += aj*aj2;
		};

	double retau((r-1)*etau), etau2(std::exp(-2.0*tau));
    VectorXd out(Deriv2iG(yi, thetai));
    out(1) -= retau;
    out(2) += retau - etau2*Saj;
    
    Saj *= etau;
    out(3) -= Saj;
    
    Saj2 *= etau2;
    out(4) += Saj - Saj2;
    out(5) += Saj2 - 2.0*std::exp(-3.0*tau)*Saj3/3.0;
	
    return out;

  } else { // derivative components w.r.t theta when xi !=0
	  
	//utility variables
    double b(etau*xi), aj(0.0), Lj(0.0), SLj(0.0), ej(0.0), Sej(0.0), Sajej(0.0), Sej2(0.0), Sajej2(0.0), Saj2ej2(0.0);
    for(int j(1); j<r; ++j){
		aj = yi(j)-mu;
		Lj = std::log1p(aj*b);
		SLj += Lj;
		Lj = -Lj-tau;
		ej = std::exp(Lj);
		Sej += ej;
		Sajej += aj*ej;
		ej = std::exp(2.0*Lj);
		Sej2 += ej;
		ej *= aj;
		Sajej2 += ej;
		Saj2ej2 += aj*ej;	
		};
    
    double xi1(1.0/xi), xip1(1.0+xi);
    VectorXd out(Deriv2iG(yi, thetai));
    
    out(0) += xi*xip1*Sej2;
    Sajej2 *= xip1;
    out(1) += xi*Sajej2 - xip1*Sej;
    out(2) +=  Sej - Sajej2;
    
    out(3) += xip1*(xi*Saj2ej2 - Sajej);
    out(4) += Sajej - xip1*Saj2ej2;
    out(5) +=  2.0*(Sajej - SLj*xi1)*xi1*xi1 + (1.0+xi1)*Saj2ej2;
	
    return out;

  };
}

/**************************************
 *
 * 3rd derivatives in matrix
 * lower triangular vector form
 *
 **************************************/
MatrixXd Deriv3iG(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
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

    MatrixXd out(MatrixXd::Zero(6,3));
    out(0,0) = -std::exp(-3.0*tau-b);
    out(1,0) = a*out(0,0)-2.0*lmumu;
    out(2,0) = std::exp(-2.0*tau)*(-e+beb*(2.0-.5*b));
    out(3,0) = out(0,0)*a*a -3.0*a*lmumu -e*et;
    out(4,0) = a*out(2,0)-et*(b*(e-.5*beb)+1.0);
    out(5,0) = b*et*(2.0*(b-1.0)+beb*(-2.0+5.0*b/3.0-.25*b2));

	/* fill only the unique elements and leave the others zero as Deriv3iRgev completes the symmtric part */
    out(3,1) = a*out(3,0);
    out(4,1) = a*out(4,0);
    out(5,1) = a*out(5,0);

    out(5,2) = (2.0 - 1.5*e*b + (b2 -b3/8.0)*eb )*b3;

    return out;

  } else { // derivative components w.r.t theta when xi !=0

    //utility variables
    double L(std::log1p(b*xi)), xi1(1/xi), f0(fexpRgev(0,-xi1,L)), f1(fexpRgev(-tau,-1,L)), f2(fexpRgev(-2*tau,-2,L));
    double f3(fexpRgev(-3*tau,-3,L)), Lxi1(L*xi1), e(std::expm1(-Lxi1)), f0xi1(f0*xi1), xi12(xi1*xi1);
    double Lf0xi1(L*f0xi1), a2(a*a), u(xi+1), v(1+xi1), lmumu(u*f2*(xi-f0));

    MatrixXd out(MatrixXd::Zero(6,3));
    out(0,0) = (2.0*xi*lmumu*f1) - u*fexpRgev(-3.0*tau,-3.0-xi1,L);
    out(1,0) = a*out(0,0)-2.0*lmumu;
    out(2,0) = a*f3*(2.0*xi*(e-xi) + (3.0+xi1)*f0) + f2*(2.0*xi-e-v*Lf0xi1);
    out(3,0) = out(0,0)*a2 -3.0*a*lmumu -f1*(e-xi);
    out(4,0) = a*out(2,0) -a*f2*(e + f0xi1-xi) - f1*(1.0-Lxi1*f0xi1);
    out(5,0) = (a*f3*(2.0*u-v*(1.0+v)*f0) + 2.0*f2*(f0*(L*v-1.0)*xi12-1.0))*a + (2.0-Lxi1)*Lxi1*xi12*fexpRgev(-tau,-1.0-xi1,L);

	/* fill only the unique elements and leave the others zero as Deriv3iRgev completes the symmtric part */
    out(3,1) = a*out(3,0);
    out(4,1) = a*out(4,0);
    out(5,1) = a*out(5,0);

    out(5,2) = ( (e*2.0*xi1 + f0*(3.0+xi1)*xi12 -2.0)*f3*a + (e+(2.0-L*v)*f0xi1)*3.0*xi12*f2 )*a2 + (3.0*a*f1*(2.0*e+(Lxi1-4.0)*Lf0xi1) + ( (6.0-Lxi1)*Lf0xi1 -6.0*e)*Lxi1 )*xi12*xi1;

    return out;

  };
}

MatrixXd Deriv3iRgev(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
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
  double etau(std::exp(-tau));
  const int r(yi.size()), r1(r-1);

  if(std::abs(xi) <= std::pow(MachinePrec,0.3)){ // derivative components w.r.t thetai when xi==0
	  
	// utility variables
    double aj(0.0), Saj(0.0), aj2(0.0), Saj2(0.0), Saj3(0.0), Saj4(0.0);
    for(int j(1); j<r; ++j){
		aj = yi(j)-mu;
		Saj += aj;
		aj2 = aj*aj;
		Saj2 += aj2;
		Saj3 += aj*aj2;
		Saj4 += aj2*aj2;
		};

	double retau(r1*etau), etau2(std::exp(-2.0*tau)), etau3(2.0*std::exp(-3.0*tau));
    MatrixXd out(Deriv3iG(yi, thetai));
    out(2,0) += etau2*r1;    
    out(3,0) += retau;
    
    etau2 *= 2.0;
    double Sa2(etau2*Saj);
    out(4,0) += Sa2 - retau;
    out(5,0) += etau3*Saj2 - Sa2;
    
    out(0,1) = out(1,0);
    out(1,1) = out(3,0);
    out(2,1) = out(4,0);
    
    Saj *= etau;
    out(3,1) += Saj;
    
    Saj2 *= etau2;
    out(4,1) += Saj2 - Saj;
    
    Saj3 *= etau3;
    out(5,1) += Saj3 - Saj2;

    out(0,2) = out(2,0);
    out(1,2) = out(4,0);
    out(2,2) = out(5,0);
    out(3,2) = out(4,1);
    out(4,2) = out(5,1);
    out(5,2) += 1.5*std::exp(-4.0*tau)*Saj4 - Saj3;
        
    return out;

  } else { // derivative components w.r.t theta when xi !=0
	  
	 //utility variables
    double b(xi*etau), aj(0.0), Lj(0.0), SLj(0.0), ej(0.0), Sej(0.0), Sajej(0.0), Sej2(0.0), Sajej2(0.0), aj2(0.0), Saj2ej2(0.0), Sej3(0.0), Sajej3(0.0), Saj2ej3(0.0), Saj3ej3(0.0);
    for(int j(1); j<r; ++j){
		aj = yi(j)-mu;
		Lj = std::log1p(aj*b);
		SLj += Lj;
		Lj = -tau - Lj;
		ej = std::exp(Lj);
		Sej += ej;
		Sajej += aj*ej;
		ej = std::exp(2.0*Lj);
		Sej2 += ej;
		Sajej2 += aj*ej;
		aj2 = aj*aj;
		Saj2ej2 += aj2*ej;
		ej = std::exp(3.0*Lj);
		Sej3 += ej;
		Sajej3 += aj*ej;
		ej *= aj2;
		Saj2ej3 += ej;
		Saj3ej3 += aj*ej;
		};

	double xip1(1.0+xi), xi2(2.0*xi), xixip12(xi2*xip1), xi23(3.0*xi+2.0);
    MatrixXd out(Deriv3iG(yi, thetai));
    out(0,0) += xi*xixip12*Sej3;
    out(1,0) += xixip12*(-Sej2 + xi*Sajej3);
    out(2,0) += (xi2+1.0)*Sej2 - xixip12*Sajej3;
    out(3,0) += xip1*(Sej + xi*(-3.0*Sajej2 + xi2*Saj2ej3));
    out(4,0) += - Sej + xi23*Sajej2 - xixip12*Saj2ej3;
    out(5,0) += 2.0*(-Sajej2 + xip1*Saj2ej3);

    out(0,1) = out(1,0);
    out(1,1) = out(3,0);
    out(2,1) = out(4,0);
    out(3,1) += xip1*(Sajej + xi*(-3.0*Saj2ej2 + xi2*Saj3ej3));
    out(4,1) += - Sajej + xi23*Saj2ej2 - xixip12*Saj3ej3;
    out(5,1) += 2.0*(-Saj2ej2 + xip1*Saj3ej3);

    out(0,2) = out(2,0);
    out(1,2) = out(4,0);
    out(2,2) = out(5,0);
    out(3,2) = out(4,1);
    out(4,2) = out(5,1);
    
    double xi1(1.0/xi), xi12(xi1*xi1);
    out(5,2) += 6.0*(SLj*xi1 - Sajej)*xi12*xi1 - 3.0*Saj2ej2*xi12 - 2.0*(1.0+xi1)*Saj3ej3;
	
    return out;

  };
}



/******************
 *
 * beta initialization
 *
 ********************/
VectorXd betaInitRgev(const Map<MatrixXd>& yi, const Map<MatrixXi>& pFull)
{
  // compute beta_0 = (mu0, 0...0, tau0, 0...0, xi0, 0...0)
  
  const int n(yi.rows()), n1(n-1), n12(n1*(n-2));
  
  VectorXd ySort(yi.col(yi.cols()-1));
  std::sort(ySort.data(), ySort.data()+n);
  
  double tmp(0.0), mu0(0.0), tau0(0.0), xi0(0.0); 
  for(int j(0); j<n; ++j){
	  tmp = ySort(j)/n;
	  mu0 += tmp;
	  tmp *= j;
	  tau0 += tmp/n1;
	  xi0 += (j-1)*tmp/n12;
	  };
  
  double sigma0((2.0*tau0 - mu0)*0.1/0.07669918); // gamma(1 - shape0) * (2^shape0 - 1) = 0.07669918
   
  VectorXd out(Eigen::VectorXd::Zero(pFull.row(1).sum()));
  out(0) = mu0 - 0.686287*sigma0; // -0.686287 = (1 - gamma(1 - 0.1))/0.1;
  out(pFull(0,1)) = std::log(sigma0); // std::log(s); // tau0
  out(pFull(0,2)) = 0.1; // xi0

  return out;
}

//VectorXd betaInitRgev(const Map<MatrixXd>& yi, const Map<MatrixXi>& pFull)
//{
  // compute beta_0 = (mu0, 0...0, tau0, 0...0, xi0, 0...0)
  
  //const int r1(yi.cols()-1);
//  double ymean((yi.col(0).mean())), ymin(yi.minCoeff()); //ymean(yi.minCoeff()); //(yi.col(0).mean());
//  double yvar(((yi.col(0).array() - ymean).matrix().squaredNorm()) /(yi.rows()-1.0));
//  double s(std::sqrt(6.0*yvar)/Pi); // se(X) when xi == 0

//  VectorXd out(Eigen::VectorXd::Zero(pFull.row(1).sum()));
//  out(0) = ; //ymean - s*0.5772157;// + s*0.5772157; // mu0
//  out(pFull(0,1)) = std::log(s); // tau0
//  out(pFull(0,2)) = 0.01; // xi0

//  return out;
//}
