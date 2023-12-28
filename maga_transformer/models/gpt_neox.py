from typing import Any, Dict

from maga_transformer.utils.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.utils.util import get_config_from_path
from maga_transformer.models.gpt_neox_weight import GPTNeoxWeight
from maga_transformer.models.gpt import GPT
from maga_transformer.model_factory_register import register_model

class GPTNeox(GPT):
    @staticmethod
    def get_weight_cls():
        return GPTNeoxWeight

    @staticmethod
    def _create_config(ckpt_path: str):
        config_dict = get_config_from_path(ckpt_path)
        if config_dict:
            config = GPTNeox.from_huggingface(config_dict)
        else:
            config = GptInitModelParameters(
                head_num=40,
                head_num_kv=40,
                size_per_head=128,
                layer_num=40,
                max_seq_len=4096,
                vocab_size=250752,
                eos_token_id=2,
                inter_size = 20480,
                inter_padding_size = 20480)
        config.ckpt_path = ckpt_path
        config.rotary_embedding_dim = 128
        config.rotary_embedding_style = 1
        config.has_pre_decoder_layernorm = False
        config.has_post_decoder_layernorm = True
        config.norm_type = 'rmsnorm'
        return config

    @staticmethod
    def from_huggingface(config_json: Dict[str, Any]):
        config = GptInitModelParameters(head_num=40,
                                        size_per_head=128,
                                        layer_num=40,
                                        max_seq_len=4096,
                                        vocab_size=250752)
        config.head_num = config_json['num_attention_heads']
        config.head_num_kv = config.head_num
        config.size_per_head = config_json['hidden_size'] // config_json['num_attention_heads']
        config.layer_num = config_json['num_hidden_layers']
        config.vocab_size = config_json['vocab_size']
        config.weights_data_type = config_json['torch_dtype']
        config.layernorm_eps = config_json['layer_norm_eps']
        config.inter_size = config_json['intermediate_size']
        config.inter_padding_size = config.inter_size
        config.special_tokens.bos_token_id = config_json['bos_token_id']
        config.special_tokens.eos_token_id = config_json['eos_token_id']
        if config_json.get('rope_scaling', None):
            if config_json['rope_scaling']['type'] == 'dynamic':
                config.dynamic_embedding_scalar = config_json['rope_scaling']['factor']
                config.dynamic_embedding_max_pos = config_json.get('max_position_embeddings', 2048)
        return config

register_model('gpt_neox', GPTNeox)