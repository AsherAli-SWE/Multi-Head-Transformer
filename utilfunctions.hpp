#pragma once
#include <boost/numeric/ublas/matrix.hpp>
#include <cmath>
#include <algorithm>
#include <random>

namespace transformer {
    namespace ublas = boost::numeric::ublas;
    ublas::matrix<double> softmax(const ublas::matrix<double>& x);

    inline ublas::matrix<double> scaled_dot_product_attention(
        const ublas::matrix<double>& Q,     // query matrix
        const ublas::matrix<double>& K,     // key matrix
        const ublas::matrix<double>& V     // values matrix
    ) {
        // dot product of query with all keys
        ublas::matrix<double> scores = ublas::prod(Q, ublas::trans(K));

        // calculate scale
        double scale = sqrt((double)K.size2()); // sqrt(dimension of Keys)

        // divide each by the scale
        //#pragma omp parallel for collapse(2)
        for (size_t i = 0; i < scores.size1(); ++i)
            for (size_t j = 0; j < scores.size2(); ++j)
                scores(i, j) /= scale;


        // apply softmax function
        auto attn = softmax(scores);
        
        // return the weights on the values
        return ublas::prod(attn, V);
    }

    inline ublas::matrix<double> scaled_dot_product_attention(
        const ublas::matrix<double>& Q,     // query matrix
        const ublas::matrix<double>& K,     // key matrix
        const ublas::matrix<double>& V,     // values matrix
        const ublas::matrix<double>* mask = nullptr // optional mask
    ) {
        // dot product of query with all keys
        ublas::matrix<double> scores = ublas::prod(Q, ublas::trans(K));

        // calculate scale
        double scale = sqrt((double)K.size2()); // sqrt(dimension of Keys)

        // divide each by the scale
        //#pragma omp parallel for collapse(2)
        for (size_t i = 0; i < scores.size1(); ++i)
            for (size_t j = 0; j < scores.size2(); ++j)
                scores(i, j) /= scale;

        // optionally we can mask
        if (mask) {
            //#pragma omp parallel for
            for (size_t i = 0; i < scores.size1(); ++i) {
                for (size_t j = 0; j < scores.size2(); ++j) {
                    if ((*mask)(i, j) == 0) {
                        scores(i, j) = -1e308;
                    }
                }
            }
        }
        // apply softmax function
        auto attn = softmax(scores);
        
        // return the weights on the values
        return ublas::prod(attn, V);
    }


    ublas::matrix<double> random_weight(int rows, int cols);

    inline ublas::matrix<double> linear(
        const ublas::matrix<double>& X,
        const ublas::matrix<double>& W,
        const ublas::matrix<double>& b
    ) { 
        ublas::matrix<double> out = ublas::prod(X, W);

        //#pragma omp parallel for collapse(2)
        for (std::size_t i = 0; i < out.size1(); ++i)
            for (std::size_t j = 0; j < out.size2(); ++j)
                out(i, j) += b(j, 0);
        return out;
    }

    inline ublas::matrix<double> linear(
        const ublas::matrix<double>& X,
        const ublas::matrix<double>& W
    ) { 
        ublas::matrix<double> out = ublas::prod(X, W);

        for (std::size_t i = 0; i < out.size1(); ++i)
            for (std::size_t j = 0; j < out.size2(); ++j)
                out(i, j) += 0;
        return out;
    }


    ublas::matrix<double> layer_norm(
        const ublas::matrix<double>& input,
        const ublas::matrix<double>& gamma,
        const ublas::matrix<double>& beta,
        double epsilon = 1e-5
    );

}
