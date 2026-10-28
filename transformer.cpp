#include "transformer.hpp"
#include <omp.h>

namespace transformer {

    Transformer::Transformer(int num_layers, int embed_dim, int num_heads, int vocab_size)
        : num_layers(num_layers), embed_dim(embed_dim) {
            
        
        for (int i = 0; i < num_layers; ++i) {
            encoders.emplace_back(embed_dim, num_heads);
            decoders.emplace_back(embed_dim, num_heads);
        }

        W_final = random_weight(embed_dim, vocab_size);
        b_final = ublas::zero_matrix<double>(vocab_size, 1);
        embedding_matrix = create_embedding_matrix(vocab_size, embed_dim);

    }

    ublas::matrix<double> Transformer::positional_encoding(int max_len, int embed_dim) {
        ublas::matrix<double> pe(max_len, embed_dim);

        for (int pos = 0; pos < max_len; ++pos) {
            for (int i = 0; i < embed_dim; ++i) {
                double angle = pos / std::pow(10000.0, 2.0 * (i / 2) / embed_dim);
                if (i % 2 == 0)
                    pe(pos, i) = std::sin(angle);
                else
                    pe(pos, i) = std::cos(angle);
            }
        }

        return pe;
    }

    ublas::matrix<double> Transformer::create_embedding_matrix(int vocab_size, int embed_dim) {
        return random_weight(vocab_size, embed_dim);
    }

    ublas::matrix<double> Transformer::embed_tokens(
        const std::vector<int>& token_ids,
        const ublas::matrix<double>& embedding_matrix
    ) {
        int seq_len = token_ids.size();
        int embed_dim = embedding_matrix.size2();
    
        ublas::matrix<double> embedded(seq_len, embed_dim);
        
        //#pragma omp parallel for
        for (int i = 0; i < seq_len; ++i) {
            int tok_id = token_ids[i];
            for (int j = 0; j < embed_dim; ++j) {
                embedded(i, j) = embedding_matrix(tok_id, j);
            }
        }
    
        return embedded;
    }

    ublas::matrix<double> Transformer::create_lookahead_mask(int size) {
        ublas::matrix<double> mask(size, size) = ublas::zero_matrix<double>(size, size);
        for (int i = 0; i < size; ++i)
            for (int j = 0; j <= i; ++j)
                mask(i, j) = 1.0;
        return mask;
    }


    ublas::matrix<double> Transformer::forward(
        const std::vector<int>& encoder_ids,
        const std::vector<int>& decoder_ids
    ) {

        encoder_input = embed_tokens(encoder_ids, embedding_matrix);
        decoder_input = embed_tokens(decoder_ids, embedding_matrix);
        ublas::matrix<double> lookahead_mask = create_lookahead_mask(decoder_input.size1());

        // sin positional encoding
        auto seq_len = encoder_input.size1(); // no batch, batched would be batch_size * this
        auto pe = positional_encoding(seq_len, embed_dim);
        auto enc_input = encoder_input + pe;
        auto dec_input = decoder_input + pe;

        // encoder stack
        auto enc_out = encoder_input;
        for (auto& layer : encoders) {
            enc_out = layer.forward(enc_out);
        }
        
        // decoder masked self-attention, then forward with encoder input
        auto dec_out = decoder_input;
        for (auto& layer : decoders) {
            auto masked_out = layer.masked_self_attention(dec_out, &lookahead_mask);
            dec_out = layer.forward(masked_out, enc_out, enc_out);
        }

        // save for back-propagation
        last_decoder_output = dec_out;

        // final projection
        auto x = linear(dec_out, W_final, b_final);
        return softmax(x);
    }

    void Transformer::backward(const ublas::matrix<double>& predictions, const std::vector<int>& target_ids, const std::vector<int>& decoder_ids) {
       
         //resize d_output
        d_output.resize(predictions.size1(), predictions.size2(), false);

        dW_final = ublas::zero_matrix<double>(W_final.size1(), W_final.size2());
        db_final = ublas::zero_matrix<double>(b_final.size1(), b_final.size2());

        //gradient embedding
        d_embedding = ublas::zero_matrix<double>(embedding_matrix.size1(), embedding_matrix.size2());

        for (std::size_t i = 0; i < predictions.size1(); ++i) {
            for (std::size_t j = 0; j < predictions.size2(); ++j) {
                double y = (target_ids[i] == j) ? 1.0 : 0.0;
                d_output(i, j) = predictions(i, j) - y;
            }
        }
        
        dW_final = ublas::prod(ublas::trans(last_decoder_output), d_output);
        for (std::size_t j = 0; j < db_final.size1(); ++j) {
            double sum = 0.0;
            for (std::size_t i = 0; i < d_output.size1(); ++i)
                sum += d_output(i, j);
            db_final(j, 0) = sum;
        }

        for (std::size_t i = 0; i < decoder_ids.size(); ++i) {
            int tok_id = decoder_ids[i];
            for (std::size_t j = 0; j < embed_dim; ++j) {
                for (std::size_t k = 0; k < d_output.size2(); ++k) {
                    d_embedding(tok_id, j) += d_output(i, k) * W_final(j, k);
                }
            }
        }
        
        // Backprop through decoder stack
        ublas::matrix<double> grad = ublas::prod(d_output, ublas::trans(W_final));

        for (int i = decoders.size() - 1; i >= 0; --i) {
            grad = decoders[i].backward(grad);
        }
        
    }

    void Transformer::sgd_step(double learning_rate) {
        //w_final update
        for (std::size_t i = 0; i < W_final.size1(); ++i)
            for (std::size_t j = 0; j < W_final.size2(); ++j)
                W_final(i, j) -= learning_rate * dW_final(i, j);

        //b_final update
        for (std::size_t i = 0; i < b_final.size1(); ++i)
            b_final(i, 0) -= learning_rate * db_final(i, 0);

        //embedding_matrix update
        for (std::size_t i = 0; i < embedding_matrix.size1(); ++i)
            for (std::size_t j = 0; j < embedding_matrix.size2(); ++j)
                embedding_matrix(i, j) -= learning_rate * d_embedding(i, j);

        // for (auto& layer : decoders) {
        //     layer.sgd_step(learning_rate);
        // }

    }


}
