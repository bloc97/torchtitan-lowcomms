# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.datasets.tokenizer.tiktoken import build_tiktoken_tokenizer
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from ..llama3.infra.parallelize import parallelize_llama
from ..llama3.infra.pipeline import pipeline_llama
from .model.args import TransformerModelArgs
from .model.model import Transformer

__all__ = [
    "parallelize_llama",
    "pipeline_llama",
    "TransformerModelArgs",
    "Transformer",
    "deepseek_v3_dense_configs",
]


deepseek_v3_dense_configs = {
    "test_100m": TransformerModelArgs(
        num_hidden_layers = 4,
        hidden_size = 1024,
        intermediate_size = 4096,
        kv_lora_rank = 64,
        q_lora_rank = 128,
        num_attention_heads = 16,
        qk_rope_head_dim = 16,
        v_head_dim = 32,
        qk_nope_head_dim = 32,
        max_position_embeddings = 2048,
        initializer_range = 0.006,
        rms_norm_eps = 1e-6,
        rope_theta = 10000.0,
        rope_scaling = None,
        attention_bias = False,
        attention_dropout = 0.0,
        dtype = "bfloat16",
    ),
    "test_1b": TransformerModelArgs(
        num_hidden_layers = 8,
        hidden_size = 2048,
        intermediate_size = 8192 + 4096,
        kv_lora_rank = 256,
        q_lora_rank = 512,
        num_attention_heads = 32,
        qk_rope_head_dim = 32,
        v_head_dim = 64,
        qk_nope_head_dim = 64,
        max_position_embeddings = 2048,
        initializer_range = 0.006,
        rms_norm_eps = 1e-6,
        rope_theta = 10000.0,
        rope_scaling = None,
        attention_bias = False,
        attention_dropout = 0.0,
        dtype = "bfloat16",
    ),
    "consilience_40b": TransformerModelArgs(
        num_hidden_layers = 61,
        hidden_size = 5120,
        intermediate_size = 36864,
        kv_lora_rank = 512,
        q_lora_rank = 1024,
        num_attention_heads = 64,
        qk_rope_head_dim = 64,
        v_head_dim = 128,
        qk_nope_head_dim = 128,
        max_position_embeddings = 2048,
        initializer_range = 0.006,
        rms_norm_eps = 1e-6,
        rope_theta = 10000.0,
        rope_scaling = None,
        attention_bias = False,
        attention_dropout = 0.0,
        dtype = "bfloat16",
    ),
}


register_train_spec(
    TrainSpec(
        name="deepseek_v3_dense",
        cls=Transformer,
        config=deepseek_v3_dense_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_tiktoken_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
)
