#include <cmath>
#include <limits>

#include "../inst/include/commonDefs.hpp"

const double MachinePrec(std::numeric_limits<double>::epsilon());
const double sqrTol(std::sqrt(MachinePrec));
const double infiniteLL(std::numeric_limits<double>::max());
const double Pi(3.141593);
