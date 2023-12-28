import os
import logging
import json
import torch
from maga_transformer.async_decoder_engine.async_model import AsyncModel
from maga_transformer.model_factory import ModelConfig, ModelFactory


class FakeModelLoader(object):
    def __init__(self, model_type: str, tokenizer_path: str, ckpt_path: str, weight_type: torch.dtype, async_mode: bool, max_seq_len: int=0) -> None:
        self.model_type = model_type
        self.tokenizer_path = tokenizer_path
        self.ckpt_path = ckpt_path
        self.weight_type = weight_type
        self.async_mode = async_mode
        self.max_seq_len = max_seq_len

        logging.info(f"tokenizer path: {self.tokenizer_path}")
        logging.info(f"check point path: {self.ckpt_path}")

    def load_model(self):
        config_path = os.path.join(self.ckpt_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path) as reader:
                content = reader.read()
                config_json = json.loads(content)
        else:
            raise Exception("not existed config_path ", config_path)

        model_cls = ModelFactory.get_model_cls(self.model_type)
        model_config = ModelConfig(ckpt_path=self.ckpt_path,
            model_type=self.model_type,
            tokenizer_path=self.tokenizer_path,
            async_mode=self.async_mode,
            weight_type=self.weight_type,
            max_seq_len=64,
            seq_size_per_block=8,
            gen_num_per_circle=1
            )

        raw_config = model_cls.create_config(model_config)
        raw_config.head_num = config_json.get("num_attention_heads", raw_config.head_num)
        raw_config.head_num_kv = config_json.get("num_attention_heads", raw_config.head_num_kv)
        if config_json.get('multi_query_attention', False):
            raw_config.head_num_kv = config_json['multi_query_group_num']
        raw_config.size_per_head = config_json.get("kv_channels", raw_config.size_per_head)
        raw_config.inter_size = config_json.get("ffn_hidden_size", raw_config.inter_size)
        raw_config.inter_padding_size = config_json.get("ffn_inter_padding_size", raw_config.inter_padding_size)
        raw_config.layer_num = config_json.get("num_layers", raw_config.layer_num)
        raw_config.vocab_size = config_json.get("padded_vocab_size", raw_config.vocab_size)
        raw_config.pre_seq_len = config_json.get("pre_seq_len", raw_config.pre_seq_len)

        raw_config.update_common(
            ckpt_path=self.ckpt_path,
            lora_infos=None,
            tokenizer_path=self.tokenizer_path,
            int8_mode=model_config.int8_mode,
            max_seq_len=self.max_seq_len,
            seq_size_per_block=8,
            tp_size=1,
            gen_num_per_circle=1
        )

        model = model_cls.from_config(raw_config)
        if self.async_mode:
            model = AsyncModel(model)

        return model
