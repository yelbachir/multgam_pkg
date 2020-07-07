#include <cmath>

#include "../inst/include/FamilyPp.hpp"
#include "../inst/include/commonDefs.hpp"


using namespace Eigen;

/**********
 *
 *
 * log-lik
 *
 **********/
double logLikPp(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
{
  // yi(0) is the size the i-th block
  // yi(1) is the threshold
  // utility variables
  double mu(thetai(0)), tau(thetai(1)), xi(thetai(2)); // params of the GEV model
  double etau(std::exp(-tau)), au(yi(1)-mu);
  const int nu(yi(0)), nup1(nu+1), nup2(nup1+1);

  if(std::abs(xi) <= std::pow(MachinePrec, 0.3)){
	  
	  double out(0.0);
	  for(int j(2); j<nup2; ++j){
		  out += yi(j);
		  };
	  return -std::exp(-au*etau)-nu*tau-etau*(out-nu*mu);

  } else {
	  
	  double b(xi*etau);
	  double yiMax(yi.segment(1,nup1).maxCoeff()), yiMin(yi.segment(1,nup1).minCoeff());
	  if( ((xi < 0.0) and ((1.0+b*(yiMax-mu)) > sqrTol)) or ((xi > 0.0) and ((1.0+b*(yiMin-mu)) > sqrTol)) ){
		  
		  double out(0.0);
		  for(int j(2); j<nup2; ++j){  
			  out += std::log1p((yi(j)-mu)*b);
			  };
		   double xi1(1.0/xi);
		   return -out*(1.0+xi1) - nu*tau - std::exp(-log1p(au*b)*xi1);
			  
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
VectorXd Deriv1iPp(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
{
 /* output for pars = (mu, tau, xi)
  * vector (lmu, ltau, lxi)
  */

  double mu(thetai(0)), tau(thetai(1)), xi(thetai(2)); // params of the model
  double etau(std::exp(-tau));
  const int nu(yi(0)), nup2(nu+2);
  
  if(std::abs(xi) <= std::pow(MachinePrec, 0.3)){ // derivative components w.r.t thetai when xi==0
    
    // utility variables
    double aj(0.0), Saj(0.0), Saj2(0.0);
    for(int j(2); j<nup2; ++j){
		aj = yi(j)-mu;
		Saj += aj;
		Saj2 += aj*aj;
		};
		
	VectorXd out(3);
	aj = yi(1)-mu;
	double eaj(std::exp(-tau-aj*etau));
	
    out(0) = nu*etau - eaj;
    Saj *= etau;
    eaj *= aj;
    out(1) = Saj - nu - eaj;
    out(2) = 0.5*(std::exp(-2.0*tau)*Saj2 - aj*etau*eaj) - Saj;

    return out;

  } else { // derivative components w.r.t theta when xi !=0
    
    //utility variables
    double b(xi*etau), aj(0.0), Lj(0.0), SLj(0.0), ej(0.0), Sej(0.0), Sajej(0.0);
    for(int j(2); j<nup2; ++j){
		aj = yi(j)-mu;
		Lj = std::log1p(aj*b);
		SLj += Lj;
		ej = std::exp(-tau-Lj);
		Sej += ej;
		Sajej += aj*ej;
		};

	double xip1(xi+1.0), xi1(1.0/xi), xi11(1.0+xi1);	
    aj = yi(1)-mu; 
	Lj = std::log1p(aj*b);
    ej = std::exp(-tau-Lj*xi11);
    
    VectorXd out(3);
    out(0) = xip1*Sej - ej;
    
    ej *= aj;
    out(1) = xip1*Sajej - nu - ej;
    out(2) = xi1*( xi1*(SLj - Lj*std::exp(-Lj*xi1)) + ej) - xi11*Sajej;

    return out;

  };
}

/**************************************
 *
 * 2nd derivatives in lower triangular
 * vector form
 *
 **************************************/
VectorXd Deriv2iPp(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
{
 /* output for pars = (mu, tau, xi)
  * vector (lmumu, lmutau, lmuxi, ltautau, ltauxi, lxixi)
  */

  double mu(thetai(0)), tau(thetai(1)), xi(thetai(2)); // params of the model
  double etau(std::exp(-tau));
  const int nu(yi(0)), nup2(nu+2);
  
  if(std::abs(xi) <= std::pow(MachinePrec, 0.3)){ // derivative components w.r.t thetai when xi==0

    // utility variables
    double aj(0.0), Saj(0.0), aj2(0.0), Saj2(0.0), Saj3(0.0);
    for(int j(2); j<nup2; ++j){
		aj = yi(j)-mu;
		Saj += aj;
		aj2 = aj*aj;
		Saj2 += aj2;
		Saj3 += aj*aj2;
		};

	aj = yi(1)-mu;
	aj2 = aj*aj;
	double retau(nu*etau), tau2(2.0*tau), etau2(std::exp(-tau2)), eaj(aj*etau), eaj1(std::exp(-tau-eaj)), 
	eaj2(std::exp(-tau2-eaj)), tau3(3.0*tau), eaj3(aj2*std::exp(-tau3-eaj));
	
    VectorXd out(6);
    out(0) = -eaj2; 
    
    eaj2 *= aj;
    out(1) = -retau - eaj2 + eaj1;
    out(2) = retau - etau2*Saj + eaj2 - 0.5*eaj3;
    
    Saj *= etau;
    eaj2 *= aj;
    out(3) = -Saj + aj*eaj1 - eaj2;
    
    Saj2 *= etau2;
    eaj3 *= aj;
    out(4) = Saj - Saj2 + eaj2 - 0.5*eaj3;
    out(5) = Saj2 + 2.0*(eaj3 - std::exp(-tau3)*Saj3)/3.0 - 0.25*aj2*aj2*std::exp(-4.0*tau-eaj);
	
    return out;

  } else { // derivative components w.r.t theta when xi !=0
	  
	//utility variables
    double b(etau*xi), aj(0.0), Lj(0.0), SLj(0.0), ej(0.0), Sej(0.0), Sajej(0.0), Sej2(0.0), Sajej2(0.0), Saj2ej2(0.0);
    for(int j(2); j<nup2; ++j){
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
    
    double xi1(1.0/xi), xi2(xi1*xi1), xip1(1.0+xi), xi11(1.0+xi1);
    aj = yi(1)-mu;
    Lj = std::log1p(aj*b);
    ej = std::exp(-tau-Lj*xi11);
    
    double ej2(std::exp(-2.0*tau-Lj*(2.0+xi1)));
    VectorXd out(6);
    
    double lmumu(ej2*xip1); 
    out(0) = xip1*xi*Sej2 - lmumu;
    
    Sajej2 *= xip1;
    
    double lmutau(ej - aj*lmumu);
    out(1) = lmutau + xi*Sajej2 - xip1*Sej;
    
    ej2 *= aj;
    Lj *= xi1;
    
    double lmuxi(xi11*ej2 - Lj*ej*xi1);
    out(2) =  Sej - Sajej2 + lmuxi;
    out(3) = xip1*(xi*Saj2ej2 - Sajej) + aj*lmutau;
    out(4) = Sajej - xip1*Saj2ej2 + aj*lmuxi;
    out(5) = 2.0*(Sajej - SLj*xi1)*xi2 + xi11*Saj2ej2 + xi2*Lj*(2.0-Lj)*std::exp(-Lj) + (2.0*xi2*(Lj-1.0)*ej - xi1*xi11*ej2)*aj;
	
    return out;

  };
}

/**************************************
 *
 * 3rd derivatives in matrix
 * lower triangular vector form
 *
 **************************************/

MatrixXd Deriv3iPp(const Ref<const RowVectorXd>& yi, const Ref<const RowVectorXd>& thetai)
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
  const int nu(yi(0)), nup2(nu+2);
  
  if(std::abs(xi) <= std::pow(MachinePrec,0.3)){ // derivative components w.r.t thetai when xi==0
	  
	// utility variables
    double aj(0.0), Saj(0.0), aj2(0.0), Saj2(0.0), Saj3(0.0), Saj4(0.0);
    for(int j(2); j<nup2; ++j){
		aj = yi(j)-mu;
		Saj += aj;
		aj2 = aj*aj;
		Saj2 += aj2;
		Saj3 += aj*aj2;
		Saj4 += aj2*aj2;
		};
		
	aj = yi(1)-mu;
	aj2 = aj*aj;
	
	double retau(nu*etau), tau2(2.0*tau), etau2(std::exp(-tau2)), tau3(3.0*tau), etau3(2.0*std::exp(-tau3)), tau4(4.0*tau), 
	eaj(aj*etau), ej(std::exp(-tau-eaj)), ej2(std::exp(-tau2-eaj)), ej3(std::exp(-tau3-eaj)), ej4(aj2*std::exp(-tau4-eaj));
    MatrixXd out(MatrixXd::Zero(6,3));
    
    out(0,0) = -ej3;
    
    ej3 *= aj;
    out(1,0) = 2.0*ej2 - ej3;
    out(2,0) = etau2*nu - 0.5*ej4 + 2.0*ej3 - ej2;    
    
    ej3 *= aj;
    ej2 *= aj;
    out(3,0) = retau - ej + 3.0*ej2 - ej3; 
    
    etau2 *= 2.0;
    ej4 *= aj;
    aj2 = aj2*aj2;
    double Sa2(etau2*Saj), ej5(aj2*std::exp(-5.0*tau-eaj)), lmutauxi(-2.0*ej2+2.5*ej3-0.5*ej4), lmuxixi(-2.0*ej3+5.0*ej4/3.0-0.25*ej5);
    out(4,0) = Sa2 - retau + lmutauxi;
    out(5,0) = etau3*Saj2 - Sa2 + lmuxixi;
    
    out(0,1) = out(1,0);
    out(1,1) = out(3,0);
    out(2,1) = out(4,0);
    
    ej2 *= aj;
    ej3 *= aj;
    Saj *= etau;
    out(3,1) = Saj - aj*ej + 3.0*ej2 - ej3;
    
    Saj2 *= etau2;
    out(4,1) = Saj2 - Saj + aj*lmutauxi;
    
    Saj3 *= etau3;
    out(5,1) = Saj3 - Saj2 + aj*lmuxixi;

    out(0,2) = out(2,0);
    out(1,2) = out(4,0);
    out(2,2) = out(5,0);
    out(3,2) = out(4,1);
    out(4,2) = out(5,1);
    out(5,2) = 1.5*(std::exp(-tau4)*Saj4 - aj*ej4) - Saj3 + aj*(ej5 - 0.125*aj2*aj*std::exp(-6.0*tau-eaj));
        
    return out;

  } else { // derivative components w.r.t theta when xi !=0
	  
	 //utility variables
    double b(xi*etau), aj(0.0), Lj(0.0), SLj(0.0), ej(0.0), Sej(0.0), Sajej(0.0), Sej2(0.0), Sajej2(0.0), aj2(0.0), Saj2ej2(0.0), Sej3(0.0), Sajej3(0.0), Saj2ej3(0.0), Saj3ej3(0.0);
    for(int j(2); j<nup2; ++j){
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
		
	aj = yi(1)-mu;
	Lj = std::log1p(aj*b);
		
	double xip1(1.0+xi), xi2(2.0*xi), xixip12(xi2*xip1), xi23(3.0*xi+2.0), xi1(1.0/xi), xi12(xi1*xi1), xi11(1.0+xi1), tl(tau+Lj);
	
	Lj *= xi1;
	ej = std::exp(-tl-Lj);
	
	double e2j(std::exp(-2.0*tl-Lj)), e3j(std::exp(-3.0*tl-Lj)), lmumumu(xip1*(xi2+1.0)*e3j);
    MatrixXd out(MatrixXd::Zero(6,3));
    out(0,0) = xi*xixip12*Sej3 - lmumumu;
    
    lmumumu *= aj;
    out(1,0) = xixip12*(-Sej2 + xi*Sajej3) + 2.0*xip1*e2j - lmumumu;
    e3j *= aj;
    
    double lmumuxi(xip1*(xi11+1.0)*e3j), u(Lj*xi11+1.0);
    
    out(2,0) = (xi2+1.0)*Sej2 - xixip12*Sajej3 - u*e2j + lmumuxi;
    
    e2j *= aj;
    
    double lmutautau(-ej + 3.0*xip1*e2j - aj*lmumumu);
    out(3,0) = xip1*(Sej + xi*(-3.0*Sajej2 + xi2*Saj2ej3)) + lmutautau;
    
    lmumuxi *= aj;
    double lmutauxi(-(xi11+u)*e2j+Lj*ej*xi1+lmumuxi);
    out(4,0) = - Sej + xi23*Sajej2 - xixip12*Saj2ej3 + lmutauxi;
    
    lmumuxi *= xi1;
    double lmuxixi(xi1*(Lj*xi1*(2.0-Lj)*ej+2.0*e2j*(u-xi11))-lmumuxi);
    out(5,0) = 2.0*(-Sajej2 + xip1*Saj2ej3) + lmuxixi;

    out(0,1) = out(1,0);
    out(1,1) = out(3,0);
    out(2,1) = out(4,0);
    out(3,1) = xip1*(Sajej + xi*(-3.0*Saj2ej2 + xi2*Saj3ej3)) + aj*lmutautau;
    out(4,1) = -Sajej + xi23*Saj2ej2 - xixip12*Saj3ej3 + aj*lmutauxi;
    out(5,1) = 2.0*(-Saj2ej2 + xip1*Saj3ej3) + aj*lmuxixi;

    out(0,2) = out(2,0);
    out(1,2) = out(4,0);
    out(2,2) = out(5,0);
    out(3,2) = out(4,1);
    out(4,2) = out(5,1);
    out(5,2) = xi1*(Lj*xi12*std::exp(-Lj)*(Lj*(6.0-Lj)-6.0) + aj*(3.0*xi1*( ej*xi1*(Lj*(Lj-4.0)+2.0) + e2j*(2.0*xi11-u)) + lmumuxi)) 
				+ 3.0*xi12*( 2.0*xi1*(SLj*xi1 - Sajej) - Saj2ej2) - 2.0*xi11*Saj3ej3;
				
    return out;

  };
}


/******************
 *
 * beta initialization
 *
 ********************/
 
VectorXd betaInitPp(const Map<MatrixXd>& yi, const Map<MatrixXi>& pFull)
{
  // compute beta_0 = (mu0, 0...0, tau0, 0...0, xi0, 0...0)
  
  const int n(yi.rows());
  VectorXd yMax(n);
  
  for(int i(0); i<n; ++i){
	  //yMax(i) = yi.block(i,2,1,yi(i,0)).maxCoeff();
	  yMax(i) = yi.row(i).segment(2,yi(i,0)).maxCoeff();
	  };
  double ymean(yMax.mean());
  
  double yvar(((yMax.array() - ymean).matrix().squaredNorm()) /(n-1.0));
  double s(std::sqrt(6.0*yvar)/Pi); // se(X) when xi == 0

  VectorXd out(Eigen::VectorXd::Zero(pFull.row(1).sum()));
  out(0) = ymean - s*0.5772157; // mu0
  out(pFull(0,1)) = std::log(s); // tau0
  out(pFull(0,2)) = 0.1; // xi0

  return out;
}

