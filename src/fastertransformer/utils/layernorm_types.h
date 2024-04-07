#pragma once

#include "src/fastertransformer/utils/assert_utils.h"
#include <string>

namespace fastertransformer {

enum class LayerNormType {
    pre_layernorm,
    post_layernorm,
    invalid_type
};

enum class NormType {
    layernorm,
    rmsnorm,
    alphanorm,
    add_bias,
    invalid_type
};

inline LayerNormType getLayerNormType(std::string layernorm_type_str) {
    if (layernorm_type_str == "pre_layernorm") {
        return LayerNormType::pre_layernorm;
    } else if (layernorm_type_str == "post_layernorm") {
        return LayerNormType::post_layernorm;
    } else {
        FT_CHECK_WITH_INFO(false, "Layernorm Type: " + layernorm_type_str + " not supported !");
    }
    return LayerNormType::invalid_type;
}

inline NormType getNormType(std::string norm_type_str) {
    if (norm_type_str == "layernorm") {
        return NormType::layernorm;
    } else if (norm_type_str == "rmsnorm") {
        return NormType::rmsnorm;
    } else if (norm_type_str == "alphanorm") {
        return NormType::alphanorm;
    } else {
        FT_CHECK_WITH_INFO(false, "Norm Type: " + norm_type_str + " not supported !");
    }
    return NormType::invalid_type;
}

} // namespace fastertransformer
