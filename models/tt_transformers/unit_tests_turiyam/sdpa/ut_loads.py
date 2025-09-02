import json
import os

import torch

import ttnn
from models.tt_transformers.tt.rope import RotarySetup


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

    def forward(self, x: torch.Tensor, seq_len: int = None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
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


def load_reference_inputs():
    root = os.getcwd()
    ref_data_path = os.path.join(root, "reference_data_runpod")
    example = "target_attention_intermediates_372.json"
    with open(os.path.join(ref_data_path, example)) as f:
        example = json.load(f)
        example = example["0"]
    return example


if __name__ == "__main__":
    example = load_reference_inputs()
    # for key,item in example.items():
    #    print('Key      : ' , key)
    #    print('Value    : ' , type(item))

    hidden = example["attention_block_in"]
    hidden = torch.tensor(hidden)
    hidden = hidden.to(torch.bfloat16)

    position_ids = example["attention_block_position_ids"][0]
    # position_ids = position_ids.to(torch.long)
    print("Position IDs : ", position_ids)

    qx = torch.tensor(example["attention_block_query_states"]).to(torch.bfloat16)
    kx = torch.tensor(example["attention_block_key_states"]).to(torch.bfloat16)
    vx = torch.tensor(example["attention_block_value_states"]).to(torch.bfloat16)
    print("QKV pre reshape : ", qx.shape, kx.shape, vx.shape)

    rope_in_q = torch.tensor(example["attention_block_rope_in_query_states"]).to(torch.bfloat16)
    rope_in_k = torch.tensor(example["attention_block_rope_in_key_states"]).to(torch.bfloat16)
    print("QKV post reshape : ", rope_in_q.shape, rope_in_k.shape)
    rope_in_qmin = torch.min(rope_in_q)
    rope_in_qmax = torch.max(rope_in_q)
    rope_in_kmin = torch.min(rope_in_k)
    rope_in_kmax = torch.max(rope_in_k)
    print("Rope In q : (min , max) : ", rope_in_qmin, rope_in_qmax)
    print("Rope In k : (min , max) : ", rope_in_kmin, rope_in_kmax)

    rope_out_q = torch.tensor(example["attention_block_rope_out_query_states"]).to(torch.bfloat16)
    rope_out_k = torch.tensor(example["attention_block_rope_out_key_states"]).to(torch.bfloat16)
    print("QKV post RoPE   : ", rope_out_q.shape, rope_out_k.shape)
    rope_out_qmin = torch.min(rope_out_q)
    rope_out_qmax = torch.max(rope_out_q)
    rope_out_kmin = torch.min(rope_out_k)
    rope_out_kmax = torch.max(rope_out_k)
    print("Rope Out q : (min , max) : ", rope_out_qmin, rope_out_qmax)
    print("Rope Out k : (min , max) : ", rope_out_kmin, rope_out_kmax)

    device = ttnn.open_device(device_id=0)
    rope_setup = RotarySetup(
        device=device,
        batch_size=1,
        head_dim=128,
        max_seq_len=2048,
        rope_theta=500000.0,
        scale_factor=None,
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

    errors = torch.abs((q_rotated - rope_out_q))
    print("Max error (q) , Mean error : ", torch.max(errors), torch.mean(errors))
    print("Min and max values     : ", torch.min(q_rotated), torch.max(q_rotated))

    errors = torch.abs((k_rotated - rope_out_k))
    print("Max error (k) , Mean error : ", torch.max(errors), torch.mean(errors))
    print("Min and max values     : ", torch.min(q_rotated), torch.max(q_rotated))
