import os
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from transformers import LlamaConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

import ttnn
from models.tt_transformers.tt.rope import RotarySetup


class LlamaRotaryEmbedding_L31(nn.Module):
    def __init__(
        self,
        dim=128,
        max_position_embeddings=8192,
        base=500000.0,
        device=None,
        scaling_factor=8.0,
        rope_type="llama3",
        config: Optional[LlamaConfig] = None,
    ):
        super().__init__()
        print("")
        print("Torch rope init.")
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            print("RoPE scaling is None.")
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        print("Note : In dynamic frequency update...")
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        print("")
        print("In RoPE forward. In shape, in type : ", x.shape, x.type)
        print("Position IDs : ", position_ids)
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """
    Rotates half the hidden dimensions of the input.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor with half of its hidden dimensions rotated.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_L31(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# from torchtune.modules import RotaryPositionalEmbeddings

TORCH_IN_DTYPE = torch.bfloat16
TORCH_OUT_DTYPE = torch.bfloat16
TT_IN_DTYPE = ttnn.bfloat16
TT_OUT_DTYPE = ttnn.bfloat16
SEQ_LEN = 32

if __name__ == "__main__":
    root = os.getcwd()
    ref_data_path = os.path.join(root, "reference_data")
    example = torch.load(os.path.join(ref_data_path, "example_with_intermediates.pt"))
    position_ids = example["args/position_ids"]
    position_ids = torch.squeeze(position_ids).tolist()
    past_keys = example["args/past_key_value"][0]
    attention_mask = example["args/attention_mask"]
    n_past = attention_mask.shape[-1] - attention_mask.shape[-2]

    device = ttnn.open_device(device_id=0)

    print("")
    rope_inputs_q_reference = example["outputs/intermediates"]["q_pre_rope"]
    rope_inputs_k_reference = example["outputs/intermediates"]["k_pre_rope"]
    print("Reference RoPE inputs (q) : ", rope_inputs_q_reference.shape, rope_inputs_q_reference.dtype)
    print("Reference RoPE inputs (k) : ", rope_inputs_k_reference.shape, rope_inputs_k_reference.dtype)
    rope_inputs_q_reference = rope_inputs_q_reference.to(TORCH_IN_DTYPE)
    rope_inputs_k_reference = rope_inputs_k_reference.to(TORCH_IN_DTYPE)
    print("Inputs cast to            : ", str(TORCH_IN_DTYPE))
    # print("Torch input shapes    : ", rope_inputs_q_reference.shape, rope_inputs_k_reference.shape)
    print("Torch input range (q)     : ", torch.min(rope_inputs_q_reference), torch.max(rope_inputs_q_reference))
    print("Torch input range (k)     : ", torch.min(rope_inputs_k_reference), torch.max(rope_inputs_k_reference))

    rope_in_q_tt = ttnn.from_torch(
        rope_inputs_q_reference,
        device=device,
        dtype=TT_IN_DTYPE,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    rope_in_k_tt = ttnn.from_torch(
        rope_inputs_k_reference,
        device=device,
        dtype=TT_IN_DTYPE,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    print("Getting reference outputs. Initializing torch RoPE..")

    # Init torch rope
    model_name = os.getenv("HF_MODEL")
    if not model_name:
        print("HF_MODEL environment variable not set")
        print("Set: export HF_MODEL='/path/to/your/llama/model'")

    model_path = Path(model_name)
    print(f"Checking model path for config.json and .safetensors : {model_path}")

    # Check if the directory exists
    if not model_path.exists():
        print(f"Model directory does not exist: {model_path}")
    # Check for config.json
    config_file = model_path / "config.json"
    if not config_file.exists():
        print(f"config.json not found at: {config_file}")

    import json

    # Create Llama Attn class and load qkvo
    with open(config_file, "r") as f:
        config = json.load(f)
    print("Llama model congig : ")
    print(config)
    torch_rope = LlamaRotaryEmbedding_L31(config=LlamaConfig(**config))
    # Init torch rope

    # Get torch rope out
    print("Computing RoPE outputs...")
    cos, sin = torch_rope(rope_inputs_q_reference, torch.tensor(position_ids, dtype=torch.long).unsqueeze(0))
    query_states, key_states = apply_rotary_pos_emb_L31(rope_inputs_q_reference, rope_inputs_k_reference, cos, sin)
    print("RoPE out shapes    : ", query_states.shape, key_states.shape)
    print("RoPE out ranges(q) : ", torch.min(query_states), torch.max(query_states))
    print("RoPE out ranges(k) : ", torch.min(key_states), torch.max(key_states))
    # Get torch rope out

    # rope_outputs_q_reference = example["outputs/intermediates"]["q_post_rope"]
    # rope_outputs_k_reference = example["outputs/intermediates"]["k_post_rope"]
    print("Casting outputs to : ", str(TORCH_OUT_DTYPE))
    rope_outputs_q_reference = query_states.to(TORCH_OUT_DTYPE)
    rope_outputs_k_reference = key_states.to(TORCH_OUT_DTYPE)
    print("Torch output shapes : ", rope_outputs_q_reference.shape, rope_outputs_k_reference.shape)
    print("Torch output range (q) : ", torch.min(rope_outputs_q_reference), torch.max(rope_outputs_q_reference))
    print("Torch output range (k) : ", torch.min(rope_outputs_k_reference), torch.max(rope_outputs_k_reference))

    # Torch rope setup
    # llama_rope = RotaryPositionalEmbeddings(dim = 128, max_seq_len=4096, base = 50000.0)
    # Torch rope setup

    # TT rope setup
    rope_setup = RotarySetup(
        device=device,
        batch_size=1,
        head_dim=128,
        max_seq_len=2048,
        rope_theta=500000.0,
        scale_factor=8.0,
        orig_context_len=8192,
        datatype=ttnn.float32,
    )
    trans_mats_dict = rope_setup.get_both_trans_mats()
    print("Trans mat dicts [prefill]: ", trans_mats_dict["prefill"].shape, trans_mats_dict["decode"].dtype)

    min_position = min(position_ids)
    max_position = max(position_ids)
    print("Min and max positions : ", min_position, max_position)
    # Pre generate cosine and sines
    tt_rot_mats_prefill = [
        rope_setup.cos_matrix[:, :, min_position : max_position + 1, :],
        rope_setup.sin_matrix[:, :, min_position : max_position + 1, :],
    ]
    # Pick relevant bases | positions. TT -> Torch -> TT
    tt_cosines = tt_rot_mats_prefill[0]
    tt_sines = tt_rot_mats_prefill[1]
    torch_cosines = ttnn.to_torch(ttnn.from_device(tt_cosines))
    torch_sines = ttnn.to_torch(ttnn.from_device(tt_sines))
    print("Shape check(Torch cosines and sines) : ", torch_cosines.shape, torch_sines.shape)
    print("Type  check(Torch cosines and sines) : ", torch_cosines.dtype, torch_sines.dtype)
    idx = torch.tensor(position_ids, dtype=torch.long)
    torch_cosines_full = torch_cosines[:, :, idx - min_position, :]
    torch_sines_full = torch_sines[:, :, idx - min_position, :]
    print("Extended   (Torch cosines and sines) : ", torch_cosines_full.shape, torch_sines_full.shape)
    print("Extended   (Torch cosines and sines) : ", torch_cosines_full.dtype, torch_sines_full.dtype)

    torch_cosines_full = torch_cosines_full.to(torch.bfloat16)
    torch_sines_full = torch_sines_full.to(torch.bfloat16)
    tt_cosines_full = ttnn.from_torch(torch_cosines_full, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_sines_full = ttnn.from_torch(torch_sines_full, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    # print("rot mats prefill shapes  : ", rope_setup.cos_matrix.shape, rope_setup.sin_matrix.shape)
    # print("tt rot mats relevant     : ", tt_rot_mats_prefill[0].shape, tt_rot_mats_prefill[1].shape)

    trans_mats = ttnn.to_torch(ttnn.from_device(trans_mats_dict["prefill"]))
    trans_mats = trans_mats.to(torch.bfloat16)
    trans_mats = ttnn.from_torch(
        trans_mats, device=device, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
    )
    # q_rotated = rope_in_q_tt
    q_rotated = ttnn.experimental.rotary_embedding_llama(
        rope_in_q_tt, cos_cache=tt_cosines_full, sin_cache=tt_sines_full, trans_mat=trans_mats, is_decode_mode=False
    )
    # k_rotated = rope_in_k_tt
    k_rotated = ttnn.experimental.rotary_embedding_llama(
        rope_in_k_tt, tt_cosines_full, tt_sines_full, trans_mat=trans_mats, is_decode_mode=False
    )
    print("Shape post Rope (Q,K) : ", q_rotated.shape, k_rotated.shape)
    # TT rope setup

    q_rotated = ttnn.to_torch(ttnn.from_device(q_rotated)).to(torch.float32)
    k_rotated = ttnn.to_torch(ttnn.from_device(k_rotated)).to(torch.float32)
    errors_q = torch.abs((q_rotated - rope_outputs_q_reference))
    errors_k = torch.abs((k_rotated - rope_outputs_k_reference))
    print(torch.mean(errors_q), torch.max(errors_k), torch.mean(errors_q), torch.mean(errors_k))
