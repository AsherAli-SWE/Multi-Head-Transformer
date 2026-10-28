#pragma once
#include <vector>
#include <boost/numeric/ublas/matrix.hpp>
#include "encoder_layer.hpp"
#include "decoder_layer.hpp"
#include "utilfunctions.hpp"
#include "attention.hpp"

namespace transformer {

    class Transformer {
    public:
        Transformer(int num_layers, int embed_dim, int num_heads, int vocab_size);

        ublas::matrix<double> forward(
            const std::vector<int>& encoder_ids,
            const std::vector<int>& decoder_ids
        );


        ublas::matrix<double> create_embedding_matrix(int vocab_size, int embed_dim);
        ublas::matrix<double> embed_tokens(
            const std::vector<int>& token_ids,
            const ublas::matrix<double>& embedding_matrix
        );

        void backward(const ublas::matrix<double>& predictions, const std::vector<int>& target_ids, const std::vector<int>& decoder_ids);
        void sgd_step(double learning_rate);

    private:
        int num_layers;
        int embed_dim;

        ublas::matrix<double> encoder_input;
        ublas::matrix<double> decoder_input;

        std::vector<EncoderLayer> encoders;
        std::vector<DecoderLayer> decoders;



        ublas::matrix<double> positional_encoding(int max_len, int embed_dim);
        ublas::matrix<double> create_lookahead_mask(int size);

        ublas::matrix<double> W_final;  // final projection
        ublas::matrix<double> b_final;


        ublas::matrix<double> dW_final;
        ublas::matrix<double> db_final;
        ublas::matrix<double> d_output; // same shape as predictions
        ublas::matrix<double> last_decoder_output; // decoder before final projection
        ublas::matrix<double> embedding_matrix;
        ublas::matrix<double> d_embedding;

    };

}
