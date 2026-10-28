#include "attention.hpp"

namespace transformer {
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
        const ublas::matrix<double>* mask
    ) {
        int batch = Q_input.size1();
        int embed_dim = Q_input.size2();
        int head_dim = embed_dim / num_heads;

        //will store concatenated output
        ublas::matrix<double> concat(batch, embed_dim);
        
        //loop over heads, each having its own weights
        head_dim = embed_dim / num_heads;
        if (embed_dim % num_heads != 0) {

            throw std::invalid_argument("embed_dim must be divisible by num_heads");
        }

        #pragma omp parallel for
        for (int h = 0; h < num_heads; ++h) {
            //create query, key, and value matrices for this head
            auto Q = linear(Q_input, Wq[h]);
            auto K = linear(K_input, Wk[h]);
            auto V = linear(V_input, Wv[h]);
            
            // apply attention
            auto head_output = scaled_dot_product_attention(Q, K, V, mask);

            //concat each heads output
            for (int i = 0; i < batch; ++i)
                for (int j = 0; j < head_dim; ++j)
                    concat(i, h * head_dim + j) = head_output(i, j);
        }
        
        //final linear projection
        return {concat, linear(concat, Wo, bo)};
    }

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
    ) {
        int batch = Q.size1();
        int embed_dim = Q.size2();
        int head_dim = embed_dim / num_heads;

        AttentionGradients grads;
        grads.dWq.resize(num_heads);
        grads.dWk.resize(num_heads);
        grads.dWv.resize(num_heads);
        grads.dWo = ublas::zero_matrix<double>(Wo.size1(), Wo.size2());
        grads.dbo = ublas::zero_matrix<double>(Wo.size2(), 1);
        grads.d_input = ublas::zero_matrix<double>(batch, embed_dim);

        // backprop through final projection Wo 
        grads.dWo = ublas::prod(ublas::trans(concat), d_output);

        for (std::size_t j = 0; j < grads.dbo.size1(); ++j) {
            double sum = 0.0;
            for (std::size_t i = 0; i < d_output.size1(); ++i)
                sum += d_output(i, j);
            grads.dbo(j, 0) = sum;
        }

        // backprop to concatenated heads
        ublas::matrix<double> d_concat = ublas::prod(d_output, ublas::trans(Wo));

        // split d_concat by head and backprop each
        for (int h = 0; h < num_heads; ++h) {
            ublas::matrix<double> d_head(batch, head_dim);
            for (int i = 0; i < batch; ++i)
                for (int j = 0; j < head_dim; ++j)
                    d_head(i, j) = d_concat(i, h * head_dim + j);

            // placeholder: identity backprop through attention itself
            // TODO: go through dot-product + softmax chain

            // backprop through V linear layer
            grads.dWv[h] = ublas::prod(ublas::trans(Q), d_head);
            grads.d_input += ublas::prod(d_head, ublas::trans(Wv[h]));

            grads.dWq[h] = ublas::zero_matrix<double>(Wq[h].size1(), Wq[h].size2());
            grads.dWk[h] = ublas::zero_matrix<double>(Wk[h].size1(), Wk[h].size2());
        }

        return grads;
    }
    
}
