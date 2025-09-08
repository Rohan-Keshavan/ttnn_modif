import os
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM

import ttnn

dtypes_config = {
    "parameters_torch": torch.bfloat16,
    "parameters_tt": ttnn.bfloat16,
    "inputs_torch": torch.bfloat16,
    "inputs_tt": ttnn.bfloat16,
    "outputs_torch": torch.bfloat16,
    "outputs_tt": ttnn.bfloat16,
    "intermediates_tt": ttnn.float32,
}

dtypes_torch_tt = {"torch.bfloat16": ttnn.bfloat16, "torch.float32": ttnn.float32}
closeness_config = {"ATOL": 1e-03, "RTOL": 1e-02}
compute_kernel_config_hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
    # False is an order of mangnitude worse. Why?
)

TILE_SIZE = 32


def load_attn_weights_llama_3(layer_idx=0):
    model_path = os.getenv("HF_MODEL")
    model_path = Path(model_path)
    print(f"Loading model from local weights: {model_path}")

    # Define and load model. 8B is small enough. Explicit load is safer.
    # Loudbox cpu load
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        local_files_only=True,
    )

    print("Model loaded from local weights.")

    # Extract attention weights from specified layer
    attention_weights = {}
    weight_keys = {
        "q_proj": f"model.layers.{layer_idx}.self_attn.q_proj.weight",
        "k_proj": f"model.layers.{layer_idx}.self_attn.k_proj.weight",
        "v_proj": f"model.layers.{layer_idx}.self_attn.v_proj.weight",
        "o_proj": f"model.layers.{layer_idx}.self_attn.o_proj.weight",
    }

    for name, key in weight_keys.items():
        keys = key.split(".")
        current = model
        for k in keys:
            current = getattr(current, k)
        if hasattr(current, "weight"):
            weight = current.weight.data
            print(f"Loaded {name} weight: {weight.shape}")
            attention_weights[name] = weight
        else:
            if hasattr(current, "data"):
                weight = current.data
                print(f"Loaded {name} weight (direct data): {weight.shape}")
                attention_weights[name] = weight
            else:
                raise AttributeError(f"Could not find weight for {name} at path {key}")
    return attention_weights


def generate_random_qkv_inputs_torch(
    N=1, low=-0.5, high=0.5, seq_len=32, n_heads=32, head_dim=128, dtype=torch.bfloat16
):
    random_tensors = []
    model_hidden = int(n_heads * head_dim)
    for _ in range(min(int(N), 1)):
        random_tensor = low + (high - low) * torch.rand(1, int(seq_len), model_hidden)
        random_tensor.to(dtype)
        random_tensors.append(random_tensor)
    return random_tensors


def get_tensor_stats(x):
    # check if torch tensor
    return {"min": torch.min(x), "max": torch.max(x), "l2": torch.norm(x)}


def get_torch_linear_out_llama_3(x, A):
    # check for types first
    # No concept of intermediates here
    x = x.to(dtype=dtypes_config["inputs_torch"])
    x_stats = get_tensor_stats(x)
    A = A.to(dtype=dtypes_config["parameters_torch"])
    in_dim = A.shape[1]
    out_dim = A.shape[0]
    torch_layer = nn.Linear(in_dim, out_dim, bias=False)  # No biases in Llama 3 (or 4)
    torch_layer.load_state_dict({"weight": A})
    torch_layer.weight.data = torch_layer.weight.data.to(dtypes_config["parameters_torch"])
    print("x and A type : ", x.dtype, torch_layer.weight.data.dtype)
    Ax = torch_layer(x).to(dtype=dtypes_config["outputs_torch"])
    Ax_stats = get_tensor_stats(Ax)
    return [Ax, x_stats, Ax_stats]


def get_tt_linear_out_llama_3(x, A, device=None):
    x = x.to(dtype=dtypes_config["inputs_torch"])
    x = ttnn.from_torch(
        x,
        device=device,
        dtype=dtypes_config["inputs_tt"],
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    A = A.to(dtype=dtypes_config["inputs_torch"])
    A = ttnn.from_torch(
        A,
        device=device,
        dtype=dtypes_config["parameters_tt"],
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    Ax = ttnn.linear(
        x,
        A,
        transpose_b=True,
        dtype=dtypes_config["outputs_tt"],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config_hifi4,
    )
    Ax = tt_to_torch(Ax, on_device=True)
    Ax = Ax.to(dtype=dtypes_config["outputs_torch"])
    Ax_stats = get_tensor_stats(Ax)
    return [Ax, Ax_stats]


def tt_attention_llama_3(k_past, v_past, q_rotated, k_current, v_current, scale, attention_mask=None):
    """Compute attention using KV cache + current tokens."""
    # k and v past are the right slices needed for the attention computation
    # Attention mask is only for the current sequence (Draft tree)
    intermediates = {}
    head_dim = k_past.shape[-1]
    attn_scale = scale

    # Get the relevant KV slice, Expand K and V
    n_qheads = q_rotated.shape[1]
    n_kv_heads = k_past.shape[1]
    gqa_group_size = int(n_qheads / n_kv_heads)
    k_past = ttnn.repeat_interleave(k_past, repeats=gqa_group_size, dim=1)
    v_past = ttnn.repeat_interleave(v_past, repeats=gqa_group_size, dim=1)

    # Compute attention scores past: Q @ K_past^T
    qkt_past = ttnn.linear(q_rotated, k_past, transpose_b=True, compute_kernel_config=compute_kernel_config_hifi4)
    qkt_past = qkt_past * attn_scale  # No mask needed
    intermediates["qkt_past"] = qkt_past
    qkt_past = ttnn.typecast(qkt_past, dtype=dtypes_config["intermediates_tt"])

    # repeat k, compute attention scores current : Q @ K^T
    k_current = ttnn.repeat_interleave(k_current, repeats=gqa_group_size, dim=1)
    qkt_current = ttnn.linear(q_rotated, k_current, transpose_b=True, compute_kernel_config=compute_kernel_config_hifi4)
    qkt_current = qkt_current * attn_scale
    intermediates["qkt_current"] = qkt_current
    qkt_current = ttnn.typecast(qkt_current, dtype=dtypes_config["intermediates_tt"])

    # Apply attention mask if provided
    if attention_mask is not None:
        qkt_current = ttnn.add(qkt_current, attention_mask)

    # Concat prefix and current qkts
    qkt = ttnn.concat([qkt_past, qkt_current], dim=-1)
    intermediates["qkt_masked"] = qkt

    # Softmax and attention output
    qkt = ttnn.softmax(qkt, dim=-1)
    intermediates["qkt_softmaxed"] = qkt

    qkt = ttnn.typecast(qkt, dtype=dtypes_config["outputs_tt"])
    intermediates["qkt_softmaxed_down_casted"] = qkt

    # Softmax and attention output
    v_current = ttnn.repeat_interleave(v_current, repeats=gqa_group_size, dim=1)
    v_complete = ttnn.concat([v_past, v_current], dim=2)
    qktv = ttnn.linear(qkt, v_complete, compute_kernel_config=compute_kernel_config_hifi4)
    intermediates["qktv"] = qktv

    return qktv, intermediates


def get_tt_attn_out_llama_3(kv_k, kv_v, q_rotated, k_rotated, v_current, attention_mask, device=None):
    bsz, nkv_heads, _, head_dim = kv_k.shape
    attn_scale = 1 / (head_dim**0.5)
    _, _, n_draft, cur_seq_len = attention_mask.shape
    past_kv_len = int(cur_seq_len - n_draft)
    kv_k = kv_k[:, :, 0:past_kv_len, :]
    kv_v = kv_v[:, :, 0:past_kv_len, :]
    attention_mask = attention_mask[:, :, :, past_kv_len:]
    attention_mask = torch_to_tt_default(attention_mask, device=device)
    kv_k = torch_to_tt_default(kv_k, device=device)
    kv_v = torch_to_tt_default(kv_v, device=device)
    q_rotated = torch_to_tt_default(q_rotated, device=device)
    k_rotated = torch_to_tt_default(k_rotated, device=device)
    v_current = torch_to_tt_default(v_current, device=device)
    _, tt_intermediates = tt_attention_llama_3(kv_k, kv_v, q_rotated, k_rotated, v_current, attn_scale, attention_mask)
    return tt_intermediates


def tt_rope_llama3(position_ids, q, k, device=None):
    rot_mat = get_rot_transformation_mat()
    cos, sin = get_torch_sin_cos_tt(position_ids)
    cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    cos = ttnn.from_torch(
        cos,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtypes_config["inputs_tt"],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    sin = ttnn.from_torch(
        sin,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtypes_config["inputs_tt"],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    rot_mat = ttnn.from_torch(
        rot_mat,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtypes_config["inputs_tt"],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    q = ttnn.from_torch(
        q,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtypes_config["inputs_tt"],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    k = ttnn.from_torch(
        k,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtypes_config["inputs_tt"],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    print("Shapes in to rope api :: ", q.shape, cos.shape, sin.shape, rot_mat.shape)
    q_rotated = ttnn.experimental.rotary_embedding_llama(q, cos, sin, rot_mat, is_decode_mode=False)
    k_rotated = ttnn.experimental.rotary_embedding_llama(k, cos, sin, rot_mat, is_decode_mode=False)
    q_rotated = tt_to_torch(q_rotated, on_device=True).to(dtypes_config["outputs_torch"])
    k_rotated = tt_to_torch(k_rotated, on_device=True).to(dtypes_config["outputs_torch"])
    return q_rotated, k_rotated


def get_torch_sin_cos_tt(position_ids):
    # computation of the basis vectors happen in f32 in Eagle.
    cos, sin = compute_gather_cos_sin(
        dhead=128,
        end=2048,
        theta=500000.0,
        scale_factor=8.0,
        orig_context_len=8192,
        position_ids=position_ids,  # if dynamic, why pre-compute?
    )
    return cos.to(torch.bfloat16), sin.to(torch.bfloat16)


def torch_to_tt_default(x, device):
    return ttnn.from_torch(
        x,
        device=device,
        dtype=dtypes_torch_tt[str(x.dtype)],
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def tt_to_torch(x, on_device=True):
    return ttnn.to_torch(ttnn.from_device(x)) if on_device else ttnn.to_torch(x)


def pcc(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # x = x.view(-1).float()
    # y = y.view(-1).float()
    x = x.flatten()
    y = y.flatten()
    vx = x - x.mean()
    vy = y - y.mean()
    corr = torch.sum(vx * vy) / torch.sqrt(torch.sum(vx**2) * torch.sum(vy**2))
    return corr


def compare(x: torch.Tensor, y: torch.Tensor, atol=closeness_config["ATOL"], rtol=closeness_config["RTOL"]):
    # pcc , all close, is close ( fail %s )
    # expect x and y to be torch tensors : y is reference, x is input
    # min/max
    errors = torch.abs((y - x))
    mean_error = torch.mean(errors)
    max_error = torch.max(errors)
    print(x.dtype, y.dtype)
    print("torch allclose           : ", torch.allclose(x, y, atol=atol, rtol=rtol))
    print("Mean and Max L1 errors   : ", mean_error, max_error)
    mask = torch.abs(y - x) > (atol + rtol * torch.abs(y))
    fail_fraction = mask.float().mean().item()
    fail_percent = fail_fraction * 100
    print("Fail percent             : ", fail_percent)
    print("Pcc                      : ", pcc(x, y))
    return


# RoPE utils TT
def compute_gather_cos_sin(dhead, end, theta, scale_factor, orig_context_len, position_ids):
    cos, sin = precompute_freqs(dhead, end, theta, scale_factor, orig_context_len)
    return gather_cos_sin(position_ids, cos, sin)


def precompute_freqs(dim: int, end: int, theta, scale_factor, orig_context_len):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    if scale_factor is not None:
        freqs = apply_scaling(freqs, scale_factor, orig_context_len)
    freqs = torch.outer(t, freqs).float()
    return torch.cos(freqs), torch.sin(freqs)


def apply_scaling(freqs: torch.Tensor, scale_factor: float, orig_context_len: int):
    # FIXME: Llama-3.x specific scaling - we need to support yarn for Qwen2.5 models
    # Values obtained from grid search
    import math

    low_freq_factor = 1.0
    high_freq_factor = 4.0

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


# NOTE: modified stack doesn't stack right ? repeat establishes equivalence
def gather_cos_sin(position_ids, cos, sin):
    position_id_expanded = position_ids.unsqueeze(1).expand(-1, cos.shape[-1])
    cos = cos.gather(0, position_id_expanded)
    sin = sin.gather(0, position_id_expanded)
    # cos                     = torch.stack([cos, cos], dim=-1).flatten(-2).unsqueeze(0)
    # sin                     = torch.stack([sin, sin], dim=-1).flatten(-2).unsqueeze(0)
    cos = cos.repeat(1, 2).unsqueeze(0)
    sin = sin.repeat(1, 2).unsqueeze(0)
    return cos, sin


def get_rot_transformation_mat():
    # ROPE op uses a single tile
    dhead = TILE_SIZE
    rot_emb_matrix = torch.zeros(1, 1, dhead, dhead)
    rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1
    rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1
    return rot_emb_matrix


# RoPE utils TT
