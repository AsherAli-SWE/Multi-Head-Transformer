#include "common_sublayer.hpp"

namespace transformer {

    CommonSubLayer::CommonSubLayer(int embed_dim, int num_heads)
        : embed_dim(embed_dim), num_heads(num_heads) {
        initialize_weights();
    }

    void CommonSubLayer::initialize_weights() {
        int head_dim = embed_dim / num_heads;
        for (int i = 0; i < num_heads; ++i) {
            Wq_self.push_back(random_weight(embed_dim, head_dim));
            Wk_self.push_back(random_weight(embed_dim, head_dim));
            Wv_self.push_back(random_weight(embed_dim, head_dim));
        }
        
        Wo_self = random_weight(embed_dim, embed_dim);
        bo_self = ublas::zero_matrix<double>(embed_dim, 1);

        W1 = random_weight(embed_dim, 4 * embed_dim);
        b1 = ublas::zero_matrix<double>(4 * embed_dim, 1);
        W2 = random_weight(4 * embed_dim, embed_dim);
        b2 = ublas::zero_matrix<double>(embed_dim, 1);

        dW1 = ublas::zero_matrix<double>(embed_dim, 4*embed_dim);
        db1 = ublas::zero_matrix<double>(4*head_dim, 1);
        dW2 = ublas::zero_matrix<double>(4*embed_dim, embed_dim);
        db2 = ublas::zero_matrix<double>(embed_dim, 1);

        gamma1 = one_matrix(embed_dim, 1);
        beta1 = ublas::zero_matrix<double>(embed_dim, 1);
        gamma2 = one_matrix(embed_dim, 1);
        beta2 = ublas::zero_matrix<double>(embed_dim, 1);
    }

    ublas::matrix<double> CommonSubLayer::forward(const ublas::matrix<double>& input) {
        return forward(input, input, input);
    }

    ublas::matrix<double> CommonSubLayer::forward(
        const ublas::matrix<double>& Q_input,
        const ublas::matrix<double>& K_input,
        const ublas::matrix<double>& V_input
    ) {
        auto [attn_concat, attn_output]  = multi_head_attention(
            Q_input, K_input, V_input,
            num_heads, Wq_self, Wk_self, Wv_self, Wo_self, bo_self
        );

        this->attn_concat = attn_concat;
        this->input_cache = Q_input;
        

        x1 = layer_norm(Q_input + attn_output, gamma1, beta1);
        auto ffn_out = feed_forward(x1, W1, b1, W2, b2);
        auto x2 = layer_norm(x1 + ffn_out, gamma2, beta2);
        return x2;
    }

    ublas::matrix<double> CommonSubLayer::backward(const ublas::matrix<double>& d_output) {
        return d_output;
    
        // recompute FFN forward (needed for backprop)
        ublas::matrix<double> z1 = ublas::prod(x1, W1);
        for (std::size_t i = 0; i < z1.size1(); ++i)
            for (std::size_t j = 0; j < z1.size2(); ++j)
                z1(i, j) += b1(j, 0);
    
        ublas::matrix<double> a1 = z1;
        for (std::size_t i = 0; i < a1.size1(); ++i)
            for (std::size_t j = 0; j < a1.size2(); ++j)
                a1(i, j) = std::max(0.0, a1(i, j));  // ReLu
    
        ublas::matrix<double> z2 = ublas::prod(a1, W2);
        for (std::size_t i = 0; i < z2.size1(); ++i)
            for (std::size_t j = 0; j < z2.size2(); ++j)
                z2(i, j) += b2(j, 0);
    
        // backprop through layernorm2 (x2 = LN(x1 + ffn))
        // for now  skip LN backward
        ublas::matrix<double> dz2 = d_output;
    
        // backprop W2
        ublas::matrix<double> dW2 = ublas::prod(ublas::trans(a1), dz2);
    
        // backprop b2
        ublas::matrix<double> db2(b2.size1(), 1);
        for (std::size_t j = 0; j < db2.size1(); ++j) {
            double sum = 0.0;
            for (std::size_t i = 0; i < dz2.size1(); ++i)
                sum += dz2(i, j);
            db2(j, 0) = sum;
        }
    
        // backprop through W2-a1
        ublas::matrix<double> da1 = ublas::prod(dz2, ublas::trans(W2));
    
        // backprop through ReLU
        ublas::matrix<double> dz1 = da1;
        for (std::size_t i = 0; i < dz1.size1(); ++i)
            for (std::size_t j = 0; j < dz1.size2(); ++j)
                dz1(i, j) = (z1(i, j) > 0) ? dz1(i, j) : 0.0;
    
        // backprop W1
        ublas::matrix<double> dW1 = ublas::prod(ublas::trans(x1), dz1);
    
        // backprop b1
        ublas::matrix<double> db1(b1.size1(), 1);
        for (std::size_t j = 0; j < db1.size1(); ++j) {
            double sum = 0.0;
            for (std::size_t i = 0; i < dz1.size1(); ++i)
                sum += dz1(i, j);
            db1(j, 0) = sum;
        }
    
        this->dW1 = dW1;
        this->db1 = db1;
        this->dW2 = dW2;
        this->db2 = db2;

        // backprop to x1
        ublas::matrix<double> d_x1 = ublas::prod(dz1, ublas::trans(W1));

        // backprop through LayerNorm1 (x1 = LN(Q + attn))
        // for now skip
        ublas::matrix<double> d_pre_ln1 = d_x1;

        // backprop through residual sum (Q + attn)
        ublas::matrix<double> d_attn_out = d_pre_ln1;  // from attention output
        ublas::matrix<double> d_input_skip = d_pre_ln1;  // skip connection

        //  backprop through multi-head attention
        auto attn_grads = multi_head_attention_backward(
            d_attn_out,
            input_cache, input_cache, input_cache,
            attn_concat,
            num_heads,
            Wq_self, Wk_self, Wv_self, Wo_self
        );
        
        dWq_self = attn_grads.dWq;
        dWk_self = attn_grads.dWk;
        dWv_self = attn_grads.dWv;
        dWo_self = attn_grads.dWo;
        dbo_self = attn_grads.dbo;
        
        return attn_grads.d_input;
    }
    
    // currently segfaults im not going to bother 
    void CommonSubLayer::sgd_step(double learning_rate) {
        // update W1
        for (std::size_t i = 0; i < dW1.size1(); ++i)
            for (std::size_t j = 0; j < dW1.size2(); ++j)
                W1(i, j) -= learning_rate * dW1(i, j);

        // update W2
        for (std::size_t i = 0; i < dW2.size1(); ++i)
            for (std::size_t j = 0; j < dW2.size2(); ++j)
                W2(i, j) -= learning_rate * dW2(i, j);

        // update b1
        for (std::size_t i = 0; i < db1.size1(); ++i)
            b1(i, 0) -= learning_rate * db1(i, 0);

        // update b2
        for (std::size_t i = 0; i < db2.size1(); ++i)
            b2(i, 0) -= learning_rate * db2(i, 0);

        // update Wq, Wk, Wv
        for (int h = 0; h < num_heads; ++h) {
            for (std::size_t i = 0; i < Wq_self[h].size1(); ++i)
                for (std::size_t j = 0; j < Wq_self[h].size2(); ++j)
                    Wq_self[h](i, j) -= learning_rate * dWq_self[h](i, j);
        
            for (std::size_t i = 0; i < Wk_self[h].size1(); ++i)
                for (std::size_t j = 0; j < Wk_self[h].size2(); ++j)
                    Wk_self[h](i, j) -= learning_rate * dWk_self[h](i, j);
        
            for (std::size_t i = 0; i < Wv_self[h].size1(); ++i)
                for (std::size_t j = 0; j < Wv_self[h].size2(); ++j)
                    Wv_self[h](i, j) -= learning_rate * dWv_self[h](i, j);
        }
        
        for (std::size_t i = 0; i < Wo_self.size1(); ++i)
            for (std::size_t j = 0; j < Wo_self.size2(); ++j)
                Wo_self(i, j) -= learning_rate * dWo_self(i, j);
        
        for (std::size_t i = 0; i < dbo_self.size1(); ++i)
            dbo_self(i, 0) -= learning_rate * dbo_self(i, 0);
    }



}
