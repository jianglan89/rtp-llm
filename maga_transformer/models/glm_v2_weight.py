
import functools
from typing import List

from maga_transformer.utils.chatglm2_quantization import extract_weight_to_half
from maga_transformer.utils.model_weight import W, WeightInfo, ModelWeightInfo, \
    ModelDeployWeightInfo, CkptWeightInfo, concat_1, identity, zeros, transpose

def w_half1(ts, inter_size):
    return ts[0][:inter_size, ...].T.contiguous()

def w_half2(ts, inter_size):
    return ts[0][inter_size:, ...].T.contiguous()

class GlmV2WeightInfo(ModelDeployWeightInfo):
    def _process_meta(self, meta_dicts, weight_keys):
        if 'transformer.prefix_encoder.embedding.weight' in weight_keys:
            self._has_prefix_encoder = True

    def _get_weight_info(self):
        weights = [
            WeightInfo(W.embedding, [CkptWeightInfo('transformer.embedding.word_embeddings.weight', concat_1)], identity),
            WeightInfo(W.lm_head, [CkptWeightInfo('transformer.output_layer.weight', identity)], identity),
            WeightInfo(W.final_ln_gamma, [CkptWeightInfo('transformer.encoder.final_layernorm.weight', identity)], identity),
            WeightInfo(W.final_ln_beta, [], functools.partial(zeros, shape=[self._hidden_size])),
        ]
        if self._has_prefix_encoder:
            weights.append(WeightInfo(W.prefix_w, [CkptWeightInfo('transformer.prefix_encoder.embedding.weight', identity)], identity))

        layer_weights: List[WeightInfo] = [
            WeightInfo(W.pre_ln_gamma, [CkptWeightInfo('transformer.encoder.layers.{i}.input_layernorm.weight', identity)], identity),

            WeightInfo(W.attn_qkv_w, [CkptWeightInfo('transformer.encoder.layers.{i}.self_attention.query_key_value.weight', identity)],
                       transpose),

            WeightInfo(W.attn_qkv_b, [CkptWeightInfo('transformer.encoder.layers.{i}.self_attention.query_key_value.bias', identity)],
                       identity),

            WeightInfo(W.attn_o_w, [CkptWeightInfo('transformer.encoder.layers.{i}.self_attention.dense.weight', identity)],
                       transpose),

            WeightInfo(W.ffn_w1, [CkptWeightInfo('transformer.encoder.layers.{i}.mlp.dense_h_to_4h.weight', identity)],
                       functools.partial(w_half1, inter_size=self._inter_size)),

            WeightInfo(W.ffn_w3, [CkptWeightInfo('transformer.encoder.layers.{i}.mlp.dense_h_to_4h.weight', identity)],
                       functools.partial(w_half2, inter_size=self._inter_size)),

            WeightInfo(W.ffn_w2, [CkptWeightInfo('transformer.encoder.layers.{i}.mlp.dense_4h_to_h.weight', identity)],
                       transpose),

            WeightInfo(W.post_ln_gamma, [CkptWeightInfo('transformer.encoder.layers.{i}.post_attention_layernorm.weight', identity)],
                       identity),
        ]

        assert self._src_quantization_bit == 0 or self._src_quantization_bit in [4, 8]
        if self._src_quantization_bit in [4, 8]:
            for idx, layer_weight in enumerate(layer_weights):
                new_weight = layer_weight.weights + [CkptWeightInfo(layer_weight.weights[0].name + '_scale', functools.partial(identity, allow_empty = True))]
                layer_weights[idx] = WeightInfo(layer_weight.name, new_weight,
                                                functools.partial(extract_weight_to_half, source_bit_width = self._src_quantization_bit, sufix_func = layer_weight.process_fun))

        model_weight_info = ModelWeightInfo(layer_weights=layer_weights, weights=weights, tp_strategy=W.gpt_style_tp_strategy)
        model_weight_info.set_lora(qkv_fun=None, half1=functools.partial(w_half1, inter_size=self._inter_size),
                                       half2=functools.partial(w_half2, inter_size=self._inter_size))

        return model_weight_info
