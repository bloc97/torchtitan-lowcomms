# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass, field

from torch import nn

from torchtitan.components.tokenizer import Tokenizer
from torchtitan.config_manager import JobConfig

from torchtitan.protocols.train_spec import BaseModelArgs
from torchtitan.tools.logging import logger


@dataclass
class TransformerModelArgs(BaseModelArgs):
    
    vocab_size: int = 129280
    hidden_size: int = 7168
    intermediate_size: int = 18432
    num_hidden_layers: int = 61
    num_attention_heads: int = 128
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    qk_nope_head_dim: int = 128
    max_position_embeddings: int = 163840
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: dict = field(
        default_factory=lambda: {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "type": "yarn",
        }
    )
    attention_bias: bool = False
    attention_dropout: float = 0.0
    pad_token_id = None
    # Added for symmetric memory
    max_seq_len: int = 4096
    dtype: str = "bfloat16"
    # Added for pipeline parallel
    num_stages: int = 1
    stage_idx: int = 0

    def update_from_config(self, job_config: JobConfig, tokenizer: Tokenizer) -> None:
        self.vocab_size = tokenizer.n_words
        self.max_seq_len = job_config.training.seq_len

    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, float]:
        nparams_embedding = 0
        nparams_dense = 0

        for name, p in model.named_parameters():
            if "embedding" in name:
                nparams_embedding += p.numel()
                nparams_dense += p.numel()
            else:
                nparams_dense += p.numel()

        nparams = nparams_dense
        logger.info(
            f"Total parameter count: dense {nparams_dense:,}, "
        )

        l, h, q, t = (
            self.num_hidden_layers,
            self.num_attention_heads,
            self.hidden_size // self.num_attention_heads,
            seq_len,
        )
        # Reasoning behind the factor of 12 for the self-attention part of the formula:
        # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
        # 2. the flash attention does 1 more matmul recomputation in the backward
        #    but recomputation should not be counted in calculating MFU           (+0)
        # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
        # 4. we follow the convention and do not account for sparsity in causal attention
        num_flops_per_token = (
            6 * (nparams_dense - nparams_embedding)
            + 12 * l * h * q * t
        )

        return nparams, num_flops_per_token
