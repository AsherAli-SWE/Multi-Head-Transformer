#pragma once
#include <boost/numeric/ublas/matrix.hpp>
#include <vector>
#include "utilfunctions.hpp"

namespace transformer {
    namespace ublas = boost::numeric::ublas;

    std::pair<ublas::matrix<double>, ublas::matrix<double>> multi_head_attention(
        const ublas::matrix<double>& Q_input,
        const ublas::matrix<double>& K_input,
        const ublas::matrix<double>& V_input,
        int num_heads,
        const std::vector<ublas::matrix<double>>& Wq,
        const std::vector<ublas::matrix<double>>& Wk,
        const std::vector<ublas::matrix<double>>& Wv,
        const ublas::matrix<double>& Wo,
        const ublas::matrix<double>& bo,
        const ublas::matrix<double>* mask = nullptr
    );

    struct AttentionGradients {
        std::vector<ublas::matrix<double>> dWq, dWk, dWv;
        ublas::matrix<double> dWo, dbo;
        ublas::matrix<double> d_input;  
    };


    AttentionGradients multi_head_attention_backward(
        const ublas::matrix<double>& d_output,
        const ublas::matrix<double>& Q,
        const ublas::matrix<double>& K,
        const ublas::matrix<double>& V,
        const ublas::matrix<double>& concat,
        int num_heads,
        const std::vector<ublas::matrix<double>>& Wq,
        const std::vector<ublas::matrix<double>>& Wk,
        const std::vector<ublas::matrix<double>>& Wv,
        const ublas::matrix<double>& Wo
    );
}
