#pragma once
#include <boost/numeric/ublas/matrix.hpp>
#include <vector>
#include "attention.hpp"
#include "feedforward.hpp"
#include "utilfunctions.hpp"

namespace transformer {

    class CommonSubLayer {
    public:
        CommonSubLayer(int embed_dim, int num_heads);

        // self-attention
        ublas::matrix<double> forward(const ublas::matrix<double>& input);

        ublas::matrix<double> forward(
            const ublas::matrix<double>& Q_input,
            const ublas::matrix<double>& K_input,
            const ublas::matrix<double>& V_input
        );

        ublas::matrix<double> backward(const ublas::matrix<double>& d_output);

        void sgd_step(double learning_rate);

    protected:
        int embed_dim, num_heads;

        // self-attention weights
        std::vector<ublas::matrix<double>> Wq_self, Wk_self, Wv_self;
        ublas::matrix<double> Wo_self, bo_self;

        // feedforward weights
        ublas::matrix<double> b1, W2, b2;

        ublas::matrix<double> W1, dW1, db1, dW2, db2;
        ublas::matrix<double> attn_concat;
        ublas::matrix<double> input_cache;

        std::vector<ublas::matrix<double>> dWq_self, dWk_self, dWv_self;
        ublas::matrix<double> dWo_self, dbo_self;

        // layerNorm parameters
        ublas::matrix<double> gamma1, beta1;
        ublas::matrix<double> gamma2, beta2;
        
        void initialize_weights();
    private:
        ublas::matrix<double> x1;
    };

}
