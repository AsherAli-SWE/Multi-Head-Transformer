#include "feedforward.hpp"


namespace transformer {
    using namespace boost::numeric::ublas;

    inline matrix<double> ReLu(const matrix<double>& x) {
        matrix<double> out = x;
        //#pragma omp parallel for collapse(2)
        for (unsigned i = 0; i < x.size1(); ++i)
            for (unsigned j = 0; j < x.size2(); ++j)
                out(i, j) = std::max(0.0, x(i, j));
        return out;
    }

    matrix<double> feed_forward(
        const matrix<double>& input,
        const matrix<double>& W1,
        const matrix<double>& b1,
        const matrix<double>& W2,
        const matrix<double>& b2
    ) {
        auto x = linear(input, W1, b1);
        x = ReLu(x);
        x = linear(x, W2, b2);
        return x;
    }
}
