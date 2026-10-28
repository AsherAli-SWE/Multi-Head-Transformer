#include "decoder_layer.hpp"


namespace transformer {

    DecoderLayer::DecoderLayer(int embed_dim, int num_heads)
        : CommonSubLayer(embed_dim, num_heads) {
        int head_dim = embed_dim / num_heads;
        for (int i = 0; i < num_heads; ++i) {
            
            Wq_masked.push_back(random_weight(embed_dim, head_dim));
            Wk_masked.push_back(random_weight(embed_dim, head_dim));
            Wv_masked.push_back(random_weight(embed_dim, head_dim));
        }

        Wo_masked = random_weight(embed_dim, embed_dim);
        bo_masked = ublas::zero_matrix<double>(embed_dim, 1);

        gamma3 = one_matrix(embed_dim, 1);
        beta3 = ublas::zero_matrix<double>(embed_dim, 1);
    }

    ublas::matrix<double> DecoderLayer::masked_self_attention(
        const ublas::matrix<double>& input,
        const ublas::matrix<double>* self_mask
    ) {

        auto [attn_concat, attn_output]  = multi_head_attention(
            input, input, input,
            num_heads, Wq_masked, Wk_masked, Wv_masked, Wo_masked, bo_masked, 
            self_mask
        );
        
        auto x1 = layer_norm(input + attn_output, gamma3, beta3);
        return x1;
    }


    ublas::matrix<double> DecoderLayer::backward(
        const ublas::matrix<double>& d_output
    ) {
        // proper decoder backprop would chain through attention + FFN.
        // add attention gradient logic here later.

        return d_output; // placeholder
    }

}
