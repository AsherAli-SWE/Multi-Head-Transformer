#pragma once
#include "common_sublayer.hpp"

namespace transformer {

    class EncoderLayer : public CommonSubLayer {
    public:
        EncoderLayer(int embed_dim, int num_heads);
    };
}
