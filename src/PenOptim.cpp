/*
 * Optimization of the penalized log-likelihood
 */

#include "../inst/include/PenOptim.hpp"
#include "../inst/include/commonDefs.hpp"



using namespace Eigen;

/**************************************
 *
 * Modified Cholesky correction of Hp
 *
 **************************************/

void solveModChol(MatrixXd& Hp, VectorXd& delta)
{
  // compute and solve modified Cholesky with pivoting so that the final matrix is positive def and well-conditioned
  // idea : compute E, L, D such that P'Hp P + E = LDL', where E is diagonal
  // factorization is made in place such that the lower triangular of Hp is L and its diagonal is D
  // implementation is according to Algorithm MC p 111 from Gill, Murray and Wright (2003): Pratical Optimization

  /**********************
  * compute modified Cholesky of Hp
  **********************/

  // start MC1
  const int n(Hp.rows());
  double beta2(0.0), gamma(0.0), xi(0.0);

  for(int c(0); c<n; ++c){ // iterate over the cols

    // maximum magnitude of the diagonal
    beta2 = std::abs(Hp(c,c));
    if(beta2 > gamma){
      gamma = beta2;
    };

    // maximum magnitude of the off-diagonal
    for(int r(c+1); r<n; ++r){ // iterate over the rows as Eigen is col-major

      beta2 = std::abs(Hp(r,c));
      if(beta2 > xi){
        xi = beta2;
      };
    };
  }; // end for

  beta2 = xi/std::sqrt(n*n-1);
  if(gamma > beta2){
    beta2 = gamma;
  };

  if(sqrTol > beta2){
    beta2 = sqrTol;
  };
  // end MC1

  MatrixXd C(MatrixXd::Zero(n,n));
  C.diagonal() = Hp.diagonal(); // MC2

  double theta(0.0);
  int q(0);

  for(int j(0); j<n; ++j){

    // start MC3
    gamma = 0.0;
    for(int i(j); i<n; ++i){ // find the smallest q s.t |C_qq| = max_j<=i<=n |C_ii|

      xi = std::abs(C(i,i));
      if(xi > gamma){
        gamma = xi;
        q = i;
      };
    };

    if(j != q){ // permute row and col
      Hp.row(j).swap(Hp.row(q));
      Hp.col(j).swap(Hp.col(q));
    };
    // end MC3

    // start MC4
    for(int s(0); s<j; ++s){ // fill in L, i.e. lower triangular part of Hp
      Hp(j,s) = C(j,s)/Hp(s,s);
    };

    theta = 0.0;
    for(int i(j+1); i<n; ++i){

      // fill in C
      gamma = 0.0;
      for(int s(0); s<j; ++s){
        gamma += Hp(j,s) * C(i,s);
      };
      C(i,j) = Hp(i,j) - gamma;

      // compute theta_j
      gamma = std::abs(C(i,j));
      if(gamma > theta){
        theta = gamma;
      };
    };
    // end MC4

    // start MC5
    Hp(j,j) = std::abs(C(j,j));
    gamma = theta*theta/beta2;
    if(Hp(j,j) < gamma){
      Hp(j,j) = gamma;
    };
    if(Hp(j,j) < sqrTol){
      Hp(j,j) = sqrTol;
    };
    // end MC5

    // start MC6
    for(int i(j+1); i<n; ++i){
      C(i,i) -= C(i,j)*C(i,j)/Hp(j,j);
    };
    // end MC6

  }; // end main loop on j

  /**********************
  * compute delta = Hp^-1 Up
  **********************/

  // delta is the gradient
  // only lower triangular part of Hp is used
  // Hp x = delta => LDL' x = delta:
  // 1. Forward solve for y : Ly = delta
  // 2. set  y = y/D
  // 3. Backward solve for delta : L'delta = y


  // 1. solve Ly = delta in place: fill in delta with y
  for(int r(1); r < n; ++r){ // iterate over the rows
    for(int c(0); c < r; ++c){ // iterate over the cols
      delta(r) -= Hp(r, c) * delta(c);
    };
  };

  // 2. set  y = y/D in place: y is delta
  delta.array() /= Hp.diagonal().array();

  // 3. solve L'delta = y in place: fill in delta with y, which is delta
  for(int r(n-2); r >= 0; --r){ // iterate over the rows
    for (int c(r+1); c < n; ++c){ // iterate over the cols
      delta(r) -= Hp(c, r) * delta(c);
    };
  };

};


void solveStablModChol(MatrixXd& Hp, VectorXd& b)
{
  // compute modified Cholesky of preconditioned matrix with pivoting so that the final matrix is positive def and well-conditioned
  // idea : compute E, L, D such that P'Hp P + E = LDL', where E is diagonal
  // factorization is made in place such that the lower triangular of Hp is L and its diagonal is D
  // implementation follows Algorithm MC p 111 from Gill, Murray and Wright (2003): Pratical Optimization

  /**********************
  * compute modified Cholesky of D^-1/2 Hp D^-1/2, where D = diagonal( 1/sqrt(|Hp_ii|) )
  **********************/

  // start MC1
  const int n(Hp.rows());
  double beta2(0.0), gamma(0.0), xi(0.0);
  VectorXd Dcond(n);
  Dcond = Hp.diagonal().array().abs().rsqrt(); // preconditioner D^-1/2
  Hp = Dcond.asDiagonal() * Hp * Dcond.asDiagonal(); // precondition Hp

  for(int c(0); c<n; ++c){ // iterate over the cols

    // maximum magnitude of the diagonal
    beta2 = std::abs(Hp(c,c));
    if(beta2 > gamma){
      gamma = beta2;
    };

    // maximum magnitude of the off-diagonal
    for(int r(c+1); r<n; ++r){ // iterate over the rows as Eigen is col-major

      beta2 = std::abs(Hp(r,c));
      if(beta2 > xi){
        xi = beta2;
      };
    };
  }; // end for

  beta2 = xi/std::sqrt(n*n-1);
  if(gamma > beta2){
    beta2 = gamma;
  };

  if(sqrTol > beta2){
    beta2 = sqrTol;
  };
  // end MC1

  MatrixXd C(MatrixXd::Zero(n,n));
  C.diagonal() = Hp.diagonal(); // MC2

  double theta(0.0);
  int q(0);

  for(int j(0); j<n; ++j){

    // start MC3
    gamma = 0.0;
    for(int i(j); i<n; ++i){ // find the smallest q s.t |C_qq| = max_j<=i<=n |C_ii|

      xi = std::abs(C(i,i));
      if(xi > gamma){
        gamma = xi;
        q = i;
      };
    };

    if(j != q){ // permute row and col
      Hp.row(j).swap(Hp.row(q));
      Hp.col(j).swap(Hp.col(q));
    };
    // end MC3

    // start MC4
    for(int s(0); s<j; ++s){ // fill in L, i.e. lower triangular part of Hp
      Hp(j,s) = C(j,s)/Hp(s,s);
    };

    theta = 0.0;
    for(int i(j+1); i<n; ++i){

      // fill in C
      gamma = 0.0;
      for(int s(0); s<j; ++s){
        gamma += Hp(j,s) * C(i,s);
      };
      C(i,j) = Hp(i,j) - gamma;

      // compute theta_j
      gamma = std::abs(C(i,j));
      if(gamma > theta){
        theta = gamma;
      };
    };
    // end MC4

    // start MC5
    Hp(j,j) = std::abs(C(j,j));
    gamma = theta*theta/beta2;
    if(Hp(j,j) < gamma){
      Hp(j,j) = gamma;
    };
    if(Hp(j,j) < sqrTol){
      Hp(j,j) = sqrTol;
    };
    // end MC5

    // start MC6
    for(int i(j+1); i<n; ++i){
      C(i,i) -= C(i,j)*C(i,j)/Hp(j,j);
    };
    // end MC6

  }; // end main loop on j

  /**********************
  * compute delta = Hp^-1 Up
  **********************/

  // b is the gradient
  // Hp x = b => Dcond^1/2 LDL' Dcond^1/2 x = b:
  // 1. Forward solve for y : Ly = Dcond^-1/2 b
  // 2. Backward solve for b : L'b = D^-1 y
  // 3. Set b = Dcond^-1/2 b

  // 1. solve Ly = Dcond^-1/2 b in place: fill in b with y
  for(int r(0); r < n; ++r){ // iterate over the rows

    b(r) = Dcond(r) * b(r);
    for (int c(0); c < r; ++c){ // iterate over the cols
      b(r) -= Hp(r, c) * b(c);
    };
  };

  // 2. solve L'b = D^-1 y in place
  for(int r(n-1); r >= 0; --r){ // iterate over the rows

    b(r) /= Hp(r,r);
    for (int c(r+1); c < n; ++c){ // iterate over the cols
      b(r) -= Hp(c, r) * b(c);
    };
  };

  // 3. set b = Dcond^-1/2 b in place
  b.array() *= Dcond.array();

};

/**************************************
 *
 * Main NR algo
 *
 **************************************/

void UpdateNewton(MatrixXd& Hp, VectorXd& delta)
{ // Newton update

    // declarations related to the eigendecomposition of Hp
    const int pF(Hp.rows());
    SelfAdjointEigenSolver<MatrixXd> eigenDec(pF);
    VectorXd eiVal(pF);

    VectorXd Dcond(pF);
    Dcond = Hp.diagonal().array().abs().rsqrt(); // preconditioner Hp^-1/2
    Hp = Dcond.asDiagonal() * Hp * Dcond.asDiagonal(); // precondition Hp

    eigenDec.compute(Hp.selfadjointView<Lower>());
    eiVal = eigenDec.eigenvalues(); // eigenvalues in increasing order automatically

    // fill the negative eigenvalues by the smallest positive but significant one
    int iii(0);
    while( (iii < pF) and (eiVal(iii) < sqrTol) ){ // get index (i.e., iii) of the 1st positive and significant eigenvalue

        eiVal(iii) = sqrTol;
        ++iii;
    };

    if(iii<pF){ // if there are positive and significant eigenvalues
        double tmp(eiVal(iii));
        for(int j(0); j<iii; ++j){ // set negative and too small eigenvalues to the value of the 1st positive signifiant one
            eiVal(j) = tmp;
        };
    };

    delta = Dcond.asDiagonal() * eigenDec.eigenvectors() * ((eigenDec.eigenvectors().transpose() * (Dcond.asDiagonal() * delta)).cwiseQuotient(eiVal));

};

