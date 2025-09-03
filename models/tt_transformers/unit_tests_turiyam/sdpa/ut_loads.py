# Started as a ref data script. Base for RoPE testing now.

import json
import math
import os
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from transformers import LlamaConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

import ttnn
from models.tt_transformers.tt.rope import RotarySetup

HEAD_DIM = 128
MAX_PE = 8192
BASE_THETA = 500000.0
N_ATTENTION_HEADS = 32
N_KV_HEADS = 8
GQA_GROUP_SIZE = int(N_ATTENTION_HEADS / N_KV_HEADS)

ATOL = 1e-03
RTOL = 1e-02

SEQ_LEN = 32
RUNPOD_DATA = False


def load_reference_inputs():
    root = os.getcwd()
    ref_data_path = os.path.join(root, "reference_data")
    example = torch.load(os.path.join(ref_data_path, "example_with_intermediates.pt"))
    print("Reference data loaded")
    return example


# models/experimental/llama
class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(
            self.max_seq_len_cached,
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype,
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    # def forward(self, x: torch.Tensor, seq_len: int = None):
    def forward(self, x: torch.Tensor, position_ids: list = []):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        seq_len = len(position_ids)
        min_pos = min(position_ids)
        max_pos = max(position_ids)
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        # return (self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype))
        return (
            self.cos_cached[:, :, min_pos : max_pos + 1, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, min_pos : max_pos + 1, ...].to(dtype=x.dtype),
        )


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# models/experimental/llama


def extend_cos_sin(cos, sin, pos_ids):
    min_pos = min(pos_ids)
    min_pos = torch.tensor(min_pos, dtype=torch.long)
    idx = torch.tensor(pos_ids, dtype=torch.long)
    print("Idx - min_pos : ", idx - min_pos)
    cos_extended = cos[:, :, idx - min_pos, :]
    sin_extended = sin[:, :, idx - min_pos, :]
    print("Extended   (Torch cosines and sines) : ", cos_extended.shape, sin_extended.shape)
    print("Extended   (Torch cosines and sines) : ", cos_extended.dtype, sin_extended.dtype)
    return cos_extended, sin_extended


def load_reference_inputs_runpod():
    root = os.getcwd()
    ref_data_path = os.path.join(root, "reference_data_runpod")
    example = "target_attention_intermediates_372.json"
    with open(os.path.join(ref_data_path, example)) as f:
        example = json.load(f)
        example = example["0"]
    return example


# From common
def apply_scaling(freqs: torch.Tensor, scale_factor: float, orig_context_len: int):
    # FIXME: Llama-3.x specific scaling - we need to support yarn for Qwen2.5 models
    # Values obtained from grid search
    low_freq_factor = 1
    high_freq_factor = 4

    low_freq_wavelen = orig_context_len / low_freq_factor
    high_freq_wavelen = orig_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (orig_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs(dim: int, end: int, theta, scale_factor, orig_context_len):
    """
    Precompute the frequency tensor for sine and cosine values with given dimensions.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 500000.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors containing cosine and sine values.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    if scale_factor is not None:
        freqs = apply_scaling(freqs, scale_factor, orig_context_len)
    freqs = torch.outer(t, freqs).float()
    return torch.cos(freqs), torch.sin(freqs)


def gather_cos_sin(position_ids, cos, sin):
    position_id_expanded = position_ids.unsqueeze(1).expand(-1, cos.shape[-1])
    cos = cos.gather(0, position_id_expanded)
    sin = sin.gather(0, position_id_expanded)
    cos = torch.stack([cos, cos], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    sin = torch.stack([sin, sin], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    return cos, sin


def get_prefill_rot_mat(head_dim, mesh_device, seq_len, theta, scale_factor, orig_context_len, start_pos=0):
    cos, sin = precompute_freqs(
        head_dim, seq_len * 2, theta=theta, scale_factor=scale_factor, orig_context_len=orig_context_len
    )
    cos_gathered, sin_gathered = gather_cos_sin(torch.arange(start_pos, start_pos + seq_len), cos, sin)
    assert cos_gathered.size() == (1, 1, seq_len, head_dim)
    assert sin_gathered.size() == (1, 1, seq_len, head_dim)

    # cos_gathereds = ttnn.from_torch(
    #    cos_gathered,
    #    dtype=ttnn.bfloat16,
    #    layout=ttnn.TILE_LAYOUT,
    #    device=mesh_device,
    #    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    # )
    # sin_gathereds = ttnn.from_torch(
    #    sin_gathered,
    #    dtype=ttnn.bfloat16,
    #    layout=ttnn.TILE_LAYOUT,
    #    device=mesh_device,
    #    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    # )

    rot_mats = [cos_gathered, sin_gathered]
    return rot_mats


def compute_gather_cos_sin(dhead, end, theta, scale_factor, orig_context_len, position_ids):
    cos, sin = precompute_freqs(dhead, end, theta, scale_factor, orig_context_len)
    return gather_cos_sin(position_ids, cos, sin)


def get_rot_transformation_mat(dhead):
    # ROPE op uses a single tile
    dhead = 32
    rot_emb_matrix = torch.zeros(1, 1, dhead, dhead)
    rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1
    rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1
    return rot_emb_matrix


# From common


#
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """
    Apply rotary position embeddings to query and key tensors.

    Args:
        q (torch.Tensor): Query tensor.
        k (torch.Tensor): Key tensor.
        cos (torch.Tensor): Cosine values.
        sin (torch.Tensor): Sine values.
        position_ids (torch.Tensor): Position IDs.

    Returns:
        torch.Tensor: Query and key tensors with rotary position embeddings applied.
    """
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


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


class LlamaRotaryEmbedding_L31(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[LlamaConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
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
            print("CONFIG NOT NONE")
            # BC: "rope_type" was originally "type"
            print("Check : ", config.rope_scaling)
            if config.rope_scaling is not None:
                print("ROPE TYPE NOT DEFAULT")
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


#

if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)

    if RUNPOD_DATA:
        example = load_reference_inputs_runpod()
        for key, item in example.items():
            print("Key      : ", key)
            print("Value    : ", type(item))
        hidden = example["attention_block_in"]
        hidden = torch.tensor(hidden)
        hidden = hidden.to(torch.bfloat16)

        position_ids = example["attention_block_position_ids"][0]
        print("Position IDs : ", position_ids)

        qx = torch.tensor(example["attention_block_query_states"]).to(torch.bfloat16)
        kx = torch.tensor(example["attention_block_key_states"]).to(torch.bfloat16)
        vx = torch.tensor(example["attention_block_value_states"]).to(torch.bfloat16)
        print("QKV pre reshape : ", qx.shape, kx.shape, vx.shape)

        rope_in_q = torch.tensor(example["attention_block_rope_in_query_states"]).to(torch.bfloat16)
        rope_in_k = torch.tensor(example["attention_block_rope_in_key_states"]).to(torch.bfloat16)
        print("QK rope in    : ", rope_in_q.shape, rope_in_k.shape)
        rope_in_qmin = torch.min(rope_in_q)
        rope_in_qmax = torch.max(rope_in_q)
        rope_in_kmin = torch.min(rope_in_k)
        rope_in_kmax = torch.max(rope_in_k)
        print("Rope In q : (min , max) : ", rope_in_qmin, rope_in_qmax)
        print("Rope In k : (min , max) : ", rope_in_kmin, rope_in_kmax)

        rope_out_q = torch.tensor(example["attention_block_rope_out_query_states"]).to(torch.bfloat16)
        rope_out_k = torch.tensor(example["attention_block_rope_out_key_states"]).to(torch.bfloat16)
        print("QK rope_out   : ", rope_out_q.shape, rope_out_k.shape)
        rope_out_qmin = torch.min(rope_out_q)
        rope_out_qmax = torch.max(rope_out_q)
        rope_out_kmin = torch.min(rope_out_k)
        rope_out_kmax = torch.max(rope_out_k)
        print("Rope Out q : (min , max) : ", rope_out_qmin, rope_out_qmax)
        print("Rope Out k : (min , max) : ", rope_out_kmin, rope_out_kmax)
    else:
        example = load_reference_inputs()
        rope_in_q = example["outputs/intermediates"]["q_pre_rope"].to(torch.bfloat16)
        rope_in_k = example["outputs/intermediates"]["k_pre_rope"].to(torch.bfloat16)
        rope_in_qmin = torch.min(rope_in_q)
        rope_in_qmax = torch.max(rope_in_q)
        rope_in_kmin = torch.min(rope_in_k)
        rope_in_kmax = torch.max(rope_in_k)
        print("Rope In q : (min , max) : ", rope_in_qmin, rope_in_qmax)
        print("Rope In k : (min , max) : ", rope_in_kmin, rope_in_kmax)

        rope_out_q = example["outputs/intermediates"]["q_post_rope"].to(torch.bfloat16)
        rope_out_k = example["outputs/intermediates"]["k_post_rope"].to(torch.bfloat16)
        rope_out_qmin = torch.min(rope_out_q)
        rope_out_qmax = torch.max(rope_out_q)
        rope_out_kmin = torch.min(rope_out_k)
        rope_out_kmax = torch.max(rope_out_k)
        print("Rope Out q : (min , max) : ", rope_out_qmin, rope_out_qmax)
        print("Rope Out k : (min , max) : ", rope_out_kmin, rope_out_kmax)

        position_ids = example["args/position_ids"]

    cos_matrix, sin_matrix = compute_gather_cos_sin(
        dhead=HEAD_DIM,
        end=2048 * 2,
        theta=500000.0,
        scale_factor=8.0,
        orig_context_len=MAX_PE,
        position_ids=torch.arange(2048),
    )
    min_pos = torch.min(position_ids).item()
    max_pos = torch.max(position_ids).item()
    print("Min and Max pos : ", min_pos, max_pos)
    vals, counts = torch.unique(position_ids, return_counts=True)
    sorted_counts = counts[vals.argsort()]
    print("Sorted counts : ", sorted_counts)

    cos_relevant = cos_matrix[:, :, min_pos : max_pos + 1, :]
    cos_relevant = torch.repeat_interleave(cos_relevant, sorted_counts, dim=2)
    sin_relevant = sin_matrix[:, :, min_pos : max_pos + 1, :]
    sin_relevant = torch.repeat_interleave(sin_relevant, sorted_counts, dim=2)

    cos_matrix = ttnn.from_torch(
        cos_relevant, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=None
    )
    sin_matrix = ttnn.from_torch(
        sin_relevant, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=None
    )
    print("cos matrix , sin matrix : ", cos_matrix.shape, sin_matrix.shape)

    prefill_trans_mat_torch = get_rot_transformation_mat(dhead=HEAD_DIM)
    transformation_mat_prefill = ttnn.from_torch(
        prefill_trans_mat_torch,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=None,
    )

    rope_in_q_tt = ttnn.from_torch(
        rope_in_q, device=device, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
    )
    rope_in_k_tt = ttnn.from_torch(
        rope_in_k, device=device, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
    )

    rope_out_q_tt = ttnn.experimental.rotary_embedding_llama(
        rope_in_q, cos_matrix, sin_matrix, transformation_mat_prefill, is_decode_mode=False
    )
    rope_out_k_tt = ttnn.experimental.rotary_embedding_llama(
        rope_in_k, cos_matrix, sin_matrix, transformation_mat_prefill, is_decode_mode=False
    )

    EAGLE_FORWARD = False
    if EAGLE_FORWARD:
        # Eagle
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
        print("Llama model config : ")
        print(config)

        rotary_emb = LlamaRotaryEmbedding_L31(config=LlamaConfig.from_json_file(config_file))
        # cos, sin   = rotary_emb(rope_in_q, torch.tensor(position_ids).unsqueeze(0).to(torch.long))
        cos, sin = rotary_emb(rope_in_q, position_ids.to(torch.long))
        cosine_full = cos.unsqueeze(0)
        sine_full = sin.unsqueeze(0)
        query_states, key_states = apply_rotary_pos_emb_L31(rope_in_q, rope_in_k, cos, sin)
        # print(query_states.shape, query_states.dtype)
        # print(key_states.shape, key_states.dtype)
        errors = torch.abs((query_states - rope_out_q))
        print("Max error (q) , Mean error : ", torch.max(errors), torch.mean(errors))
        print("Min and max values     : ", torch.min(query_states), torch.max(query_states))
        print(
            "Is close   : ",
            (torch.isclose(query_states, rope_out_q, atol=ATOL, rtol=RTOL)).float().mean().item() * 100,
        )

        errors = torch.abs((key_states - rope_out_k))
        print("Max error (k) , Mean error : ", torch.max(errors), torch.mean(errors))
        print("Min and max values     : ", torch.min(key_states), torch.max(key_states))
        print(
            "Is close   : ",
            (torch.isclose(key_states, rope_out_k, atol=ATOL, rtol=RTOL)).float().mean().item() * 100,
        )

        # TT RoPE
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

        tt_cosines_full = ttnn.from_torch(
            cosine_full,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        tt_sines_full = ttnn.from_torch(
            sine_full,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        trans_mats = ttnn.to_torch(ttnn.from_device(trans_mats_dict["prefill"]))
        trans_mats = ttnn.from_torch(
            trans_mats, device=device, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )
        print("TT sines and cosines : ", tt_cosines_full.dtype, tt_sines_full.dtype)
        print("TT Transmats         : ", trans_mats.dtype)

        rope_in_q_tt = ttnn.from_torch(
            rope_in_q, device=device, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )
        rope_in_k_tt = ttnn.from_torch(
            rope_in_k, device=device, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )

        q_rotated = ttnn.experimental.rotary_embedding_llama(
            rope_in_q_tt, cos_cache=tt_cosines_full, sin_cache=tt_sines_full, trans_mat=trans_mats, is_decode_mode=False
        )
        k_rotated = ttnn.experimental.rotary_embedding_llama(
            rope_in_k_tt, cos_cache=tt_cosines_full, sin_cache=tt_sines_full, trans_mat=trans_mats, is_decode_mode=False
        )
        print("Shape post Rope (Q,K) : ", q_rotated.shape, k_rotated.shape)
        q_rotated = ttnn.to_torch(ttnn.from_device(q_rotated)).to(torch.bfloat16)
        k_rotated = ttnn.to_torch(ttnn.from_device(k_rotated)).to(torch.bfloat16)
        # TT RoPE

        errors = torch.abs((q_rotated - rope_out_q))
        print("Max error (q) , Mean error : ", torch.max(errors), torch.mean(errors))
        print("Min and max values     : ", torch.min(q_rotated), torch.max(q_rotated))

        errors = torch.abs((k_rotated - rope_out_k))
        print("Max error (k) , Mean error : ", torch.max(errors), torch.mean(errors))
        print("Min and max values     : ", torch.min(q_rotated), torch.max(q_rotated))
        # Eagle

    TT_TEST_CODE = False
    if TT_TEST_CODE:
        rotary_emb = LlamaRotaryEmbedding(HEAD_DIM, max_position_embeddings=MAX_PE, base=BASE_THETA)
        x_random = torch.rand(1, N_ATTENTION_HEADS, len(position_ids), HEAD_DIM)
        cos, sin = rotary_emb(x_random, position_ids)
        cos, sin = extend_cos_sin(cos, sin, position_ids)
        position_ids_access = torch.arange(0, len(position_ids), dtype=torch.long, device=None)
        position_ids_access = position_ids_access.unsqueeze(0).view(-1, len(position_ids))
        # print('Position IDs : ' , position_ids.shape)
        # print('Position IDs : ' , position_ids)
        rope_out_q_tt, rope_out_k_tt = apply_rotary_pos_emb(rope_in_q, rope_in_k, cos, sin, position_ids_access)
        rope_out_q_tt = rope_out_q_tt.to(torch.bfloat16)
        rope_out_k_tt = rope_out_k_tt.to(torch.bfloat16)
        print("RoPE outputs (manual) : ", rope_out_q_tt.shape, rope_out_k_tt.shape)
        print("RoPE outputs (manual) : ", rope_out_q_tt.dtype, rope_out_k_tt.dtype)

        errors = torch.abs((rope_out_q_tt - rope_out_q))
        print("Max error (q) , Mean error : ", torch.max(errors), torch.mean(errors))
        print("Min and max values     : ", torch.min(rope_out_q_tt), torch.max(rope_out_q_tt))
        print(
            "Is close   : ",
            (torch.isclose(rope_out_q_tt, rope_out_q, atol=ATOL, rtol=RTOL)).float().mean().item() * 100,
        )

        errors = torch.abs((rope_out_k_tt - rope_out_k))
        print("Max error (k) , Mean error : ", torch.max(errors), torch.mean(errors))
        print("Min and max values     : ", torch.min(rope_out_k_tt), torch.max(rope_out_k_tt))
        print(
            "Is close   : ",
            (torch.isclose(rope_out_k_tt, rope_out_k, atol=ATOL, rtol=RTOL)).float().mean().item() * 100,
        )

        print("")
        print("Dot product errors")
        rope_out_k_repeated = rope_out_k.repeat(1, GQA_GROUP_SIZE, 1, 1)
        qkt_out_ref = torch.einsum("...d,...d->...", (rope_out_q, rope_out_k_repeated))

        rope_out_k_tt_repeated = rope_out_k_tt.repeat(1, GQA_GROUP_SIZE, 1, 1)
        qkt_out_tt = torch.einsum("...d,...d->...", (rope_out_q_tt, rope_out_k_tt_repeated))
        # dot = torch.sum(rope_out_q_tt * rope_out_k_tt_repeated, dim=-1)
        errors = torch.abs((qkt_out_tt - qkt_out_ref))
        print("Max error(qkt), Mean error : ", torch.max(errors), torch.mean(errors))
        print("Min and max values (tt)    : ", torch.min(qkt_out_tt), torch.max(qkt_out_tt))
        print("Min and max values (ref)   : ", torch.min(qkt_out_ref), torch.max(qkt_out_ref))
        print(
            "Is close   : ",
            (torch.isclose(qkt_out_tt, qkt_out_ref, atol=ATOL, rtol=RTOL)).float().mean().item() * 100,
        )

    ROPE_FROM_COMMON = False
    if ROPE_FROM_COMMON:
        prefill_pre_computed_rot_mats = get_prefill_rot_mat(128, device, 2048, 500000.0, 8.0, 8192, 0)
        precomputed_cosines = prefill_pre_computed_rot_mats[0]
        precomputed_sines = prefill_pre_computed_rot_mats[1]

        print("precomputed cosines shape : ", precomputed_cosines.shape)
        print("precomputed sines   shape : ", precomputed_sines.shape)

        position_ids = position_ids.squeeze()
        position_ids = position_ids.tolist()
        min_pos = min(position_ids)
        max_pos = max(position_ids)
        relevant_cosines = precomputed_cosines[:, :, min_pos : max_pos + 1, :]
        relevant_sines = precomputed_sines[:, :, min_pos : max_pos + 1, :]
        relevant_cosines, relevant_sines = extend_cos_sin(relevant_cosines, relevant_sines, position_ids)
        print("Relevant cosines shape : ", relevant_cosines.shape)

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
        trans_mat = trans_mats_dict["prefill"]
        trans_mat = ttnn.typecast(trans_mat, ttnn.bfloat16)

        tt_cosines_full = ttnn.from_torch(
            relevant_cosines,
            device=device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        tt_sines_full = ttnn.from_torch(
            relevant_sines,
            device=device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        rope_in_q_tt = ttnn.from_torch(
            rope_in_q, device=device, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )
        rope_in_k_tt = ttnn.from_torch(
            rope_in_k, device=device, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )

        q_rotated = ttnn.experimental.rotary_embedding_llama(
            rope_in_q_tt, cos_cache=tt_cosines_full, sin_cache=tt_sines_full, trans_mat=trans_mat, is_decode_mode=False
        )
        k_rotated = ttnn.experimental.rotary_embedding_llama(
            rope_in_k_tt, cos_cache=tt_cosines_full, sin_cache=tt_sines_full, trans_mat=trans_mat, is_decode_mode=False
        )

        q_rotated = ttnn.to_torch(ttnn.from_device(q_rotated)).to(torch.bfloat16)
        k_rotated = ttnn.to_torch(ttnn.from_device(k_rotated)).to(torch.bfloat16)

        errors = torch.abs((q_rotated - rope_out_q))
        print("Max error (q) , Mean error : ", torch.max(errors), torch.mean(errors))
        print("Min and max values     : ", torch.min(q_rotated), torch.max(q_rotated))
        print(
            "Is close   : ",
            (torch.isclose(q_rotated, rope_out_q, atol=ATOL, rtol=RTOL)).float().mean().item() * 100,
        )

        errors = torch.abs((k_rotated - rope_out_k))
        print("Max error (k) , Mean error : ", torch.max(errors), torch.mean(errors))
        print("Min and max values     : ", torch.min(k_rotated), torch.max(k_rotated))
        print(
            "Is close   : ",
            (torch.isclose(k_rotated, rope_out_k, atol=ATOL, rtol=RTOL)).float().mean().item() * 100,
        )

        print("")
        print("Dot product errors")
        k_rotated_repeated = k_rotated.repeat(1, GQA_GROUP_SIZE, 1, 1)
        qkt_out_tt = torch.einsum("...d,...d->...", (q_rotated, k_rotated_repeated))
        errors = torch.abs((qkt_out_tt - qkt_out_ref))
        print("Max error(qkt), Mean error : ", torch.max(errors), torch.mean(errors))
        print("Min and max values (tt)    : ", torch.min(qkt_out_tt), torch.max(qkt_out_tt))
        print("Min and max values (ref)   : ", torch.min(qkt_out_ref), torch.max(qkt_out_ref))
        print(
            "Is close   : ",
            (torch.isclose(qkt_out_tt, qkt_out_ref, atol=ATOL, rtol=RTOL)).float().mean().item() * 100,
        )

    TT_ROPE_MANUAL = False
    if TT_ROPE_MANUAL:
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
        trans_mat = trans_mats_dict["prefill"]
        trans_mat = ttnn.typecast(trans_mat, ttnn.bfloat16)
        tt_cosines_full = ttnn.from_torch(
            cos, device=device, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )
        tt_sines_full = ttnn.from_torch(
            sin, device=device, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )
        rope_in_q_tt = ttnn.from_torch(
            rope_in_q, device=device, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )
        rope_in_k_tt = ttnn.from_torch(
            rope_in_k, device=device, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )

        print("TT cosine , sin : ", tt_cosines_full.shape, tt_sines_full.shape)
        print("in q , k        : ", rope_in_q_tt.shape, rope_in_k_tt.shape)

        q_rotated = ttnn.experimental.rotary_embedding_llama(
            rope_in_q_tt, cos_cache=tt_cosines_full, sin_cache=tt_sines_full, trans_mat=trans_mat, is_decode_mode=False
        )
        k_rotated = ttnn.experimental.rotary_embedding_llama(
            rope_in_k_tt, cos_cache=tt_cosines_full, sin_cache=tt_sines_full, trans_mat=trans_mat, is_decode_mode=False
        )

        q_rotated = ttnn.to_torch(ttnn.from_device(q_rotated)).to(torch.bfloat16)
        k_rotated = ttnn.to_torch(ttnn.from_device(k_rotated)).to(torch.bfloat16)

        errors = torch.abs((q_rotated - rope_out_q))
        print("Max error (q) , Mean error : ", torch.max(errors), torch.mean(errors))
        print("Min and max values     : ", torch.min(q_rotated), torch.max(q_rotated))
        print(
            "Is close   : ",
            (torch.isclose(q_rotated, rope_out_q, atol=ATOL, rtol=RTOL)).float().mean().item() * 100,
        )

        errors = torch.abs((k_rotated - rope_out_k))
        print("Max error (k) , Mean error : ", torch.max(errors), torch.mean(errors))
        print("Min and max values     : ", torch.min(k_rotated), torch.max(k_rotated))
        print(
            "Is close   : ",
            (torch.isclose(k_rotated, rope_out_k, atol=ATOL, rtol=RTOL)).float().mean().item() * 100,
        )

        print("")
        print("Dot product errors")
        # rope_out_k_repeated     = rope_out_k.repeat(1,GQA_GROUP_SIZE,1,1)
        # qkt_out_ref             = torch.einsum('...d,...d->...',(rope_out_q, rope_out_k_repeated))

        k_rotated_repeated = k_rotated.repeat(1, GQA_GROUP_SIZE, 1, 1)
        qkt_out_tt = torch.einsum("...d,...d->...", (q_rotated, k_rotated_repeated))
        # dot = torch.sum(rope_out_q_tt * rope_out_k_tt_repeated, dim=-1)
        errors = torch.abs((qkt_out_tt - qkt_out_ref))
        print("Max error(qkt), Mean error : ", torch.max(errors), torch.mean(errors))
        print("Min and max values (tt)    : ", torch.min(qkt_out_tt), torch.max(qkt_out_tt))
        print("Min and max values (ref)   : ", torch.min(qkt_out_ref), torch.max(qkt_out_ref))
        print(
            "Is close   : ",
            (torch.isclose(qkt_out_tt, qkt_out_ref, atol=ATOL, rtol=RTOL)).float().mean().item() * 100,
        )

    TT_ROPE_DEF = False
    if TT_ROPE_DEF:
        # TT RoPE
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

        # Pre generate cosine and sines -> collect relevant
        min_position = min(position_ids)
        max_position = max(position_ids)
        n_seq = len(position_ids)
        print("Min and max positions : ", min_position, max_position)
        tt_rot_mats_prefill = [
            rope_setup.cos_matrix[:, :, min_position : max_position + 1, :],
            rope_setup.sin_matrix[:, :, min_position : max_position + 1, :],
        ]
        print("Rotation mats shape : ", tt_rot_mats_prefill[0].shape, tt_rot_mats_prefill[1].shape)
        # Pre generate cosine and sines -> collect relevant

        # Pick relevant bases | positions. TT -> Torch -> TT
        torch_cosines = ttnn.to_torch(ttnn.from_device(tt_rot_mats_prefill[0]))
        torch_sines = ttnn.to_torch(ttnn.from_device(tt_rot_mats_prefill[1]))
        print("Shape check(Torch cosines and sines) : ", torch_cosines.shape, torch_sines.shape)
        print("Type  check(Torch cosines and sines) : ", torch_cosines.dtype, torch_sines.dtype)
        idx = torch.tensor(position_ids, dtype=torch.long)
        print("Idx - min_pos : ", idx - min_position)
        torch_cosines_full = torch_cosines[:, :, idx - min_position, :]
        torch_sines_full = torch_sines[:, :, idx - min_position, :]
        print("Extended   (Torch cosines and sines) : ", torch_cosines_full.shape, torch_sines_full.shape)
        print("Extended   (Torch cosines and sines) : ", torch_cosines_full.dtype, torch_sines_full.dtype)

        tt_cosines_full = ttnn.from_torch(
            torch_cosines_full,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        tt_sines_full = ttnn.from_torch(
            torch_sines_full,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        trans_mats = ttnn.to_torch(ttnn.from_device(trans_mats_dict["prefill"]))
        trans_mats = ttnn.from_torch(
            trans_mats, device=device, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )
        print("TT sines and cosines : ", tt_cosines_full.dtype, tt_sines_full.dtype)
        print("TT Transmats         : ", trans_mats.dtype)

        rope_in_q_tt = ttnn.from_torch(
            rope_in_q, device=device, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )
        rope_in_k_tt = ttnn.from_torch(
            rope_in_k, device=device, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )

        q_rotated = ttnn.experimental.rotary_embedding_llama(
            rope_in_q_tt, cos_cache=tt_cosines_full, sin_cache=tt_sines_full, trans_mat=trans_mats, is_decode_mode=False
        )
        k_rotated = ttnn.experimental.rotary_embedding_llama(
            rope_in_k_tt, cos_cache=tt_cosines_full, sin_cache=tt_sines_full, trans_mat=trans_mats, is_decode_mode=False
        )
        print("Shape post Rope (Q,K) : ", q_rotated.shape, k_rotated.shape)
        q_rotated = ttnn.to_torch(ttnn.from_device(q_rotated)).to(torch.bfloat16)
        k_rotated = ttnn.to_torch(ttnn.from_device(k_rotated)).to(torch.bfloat16)
        # TT RoPE

        errors = torch.abs((q_rotated - rope_out_q))
        print("Max error (q) , Mean error : ", torch.max(errors), torch.mean(errors))
        print("Min and max values     : ", torch.min(q_rotated), torch.max(q_rotated))

        errors = torch.abs((k_rotated - rope_out_k))
        print("Max error (k) , Mean error : ", torch.max(errors), torch.mean(errors))
        print("Min and max values     : ", torch.min(q_rotated), torch.max(q_rotated))

    ttnn.close_device(device)
