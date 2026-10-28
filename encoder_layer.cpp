#include "encoder_layer.hpp"

namespace transformer {
    namespace ublas =  boost::numeric::ublas;
    
    //nothing to construct, since encoder is entirely the common transformer_layer
    EncoderLayer::EncoderLayer(int embed_dim, int num_heads)
        : CommonSubLayer(embed_dim, num_heads) {}

}