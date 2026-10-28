#pragma once
#include <boost/numeric/ublas/matrix.hpp>
#include "utilfunctions.hpp"
#include <algorithm>

namespace transformer {
    namespace ublas = boost::numeric::ublas;
    ublas::matrix<double> feed_forward(
        const ublas::matrix<double>& input,
        const ublas::matrix<double>& W1,
        const ublas::matrix<double>& b1,
        const ublas::matrix<double>& W2,
        const ublas::matrix<double>& b2
    );
}
