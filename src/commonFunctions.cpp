#include "../inst/include/commonFunctions.hpp"

#include <cmath>
#include <Rcpp.h>
#include <float.h>

// logit(p) = log(p/(1-p))
// The argument p must be greater than 0 and less than 1.
double Logit(const double& p)
{
    if( (p <= 0.0) || (p >= 1.0) )
    {
        Rcpp::stop("p in Binomial model must be greater than 0 and less than 1 <Logit>");
    }

    static const double smallCutOff = 0.25;

    double retval;

    if (p < smallCutOff)
    {
        // Avoid calculating 1-p since the lower bits of p would be lost.
        retval = std::log(p) - std::log1p(-p);
    }
    else
    {
		// The argument p is large enough that direct calculation is OK.
        retval = std::log(p/(1-p));
    }
    return retval;
}

// The inverse of the Logit function. Return exp(x)/(1 + exp(x)).
// Avoid overflow and underflow for extreme inputs.
double LogitInverse(const double& x)
{
    static const double X_MAX = -std::log(DBL_EPSILON);
    static const double X_MIN = std::log(DBL_MIN);
    double retval;

    if (x > X_MAX)
    {
        // For large arguments x, logit(x) equals 1 to double precision.
        retval = 1.0;  // avoids overflow of calculating e^x for large x
    }
    else if (x < X_MIN)
    {
        // logit(x) is approximately e^x for x very negative
        // and so logit would underflow when e^x underflows
        retval = 0.0;
    }
    else
    {
        // Direct calculation is safe in this range.
		// Save value to avoid two calls to e^x
        double t(std::exp(x));
        retval = t/(1+t);
    };

    return retval;
}



// return log(1 + exp(x)), preventing cancellation and overflow */
double LogOnePlusExpX(const double& x)
{
    static const double LOG_DBL_EPSILON = std::log(DBL_EPSILON);
    static const double LOG_ONE_QUARTER = -std::log(4);

    if (x > -LOG_DBL_EPSILON)
    {
        // log(exp(x) + 1) == x to machine precision
        return x;
    }
    else if (x > LOG_ONE_QUARTER)
    {
        return std::log( 1.0 + std::exp(x) );
    }
    else
    {
        // Prevent loss of precision that would result from adding small argument to 1.
        return std::log1p(std::exp(x));
    }
}
