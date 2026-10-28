#pragma once
#include <boost/numeric/ublas/matrix.hpp>
#include "common_sublayer.hpp"
#include "attention.hpp"
#include "utilfunctions.hpp"

namespace transformer {

    class DecoderLayer : public CommonSubLayer {
    public:
        DecoderLayer(int embed_dim, int num_heads);

        ublas::matrix<double> masked_self_attention(
            const ublas::matrix<double>& input,
            const ublas::matrix<double>* self_mask
        );

        ublas::matrix<double> backward(
            const ublas::matrix<double>& d_output
        );

    private:
        std::vector<ublas::matrix<double>> Wq_masked, Wk_masked, Wv_masked;
        ublas::matrix<double> Wo_masked, bo_masked;

        ublas::matrix<double> gamma3, beta3;
    };

}
