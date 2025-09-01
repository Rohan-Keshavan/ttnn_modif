import math
import os
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from transformers import AutoModelForCausalLM

import ttnn

TORCH_IN_DTYPE = torch.bfloat16
TORCH_OUT_DTYPE = torch.bfloat16
TT_OUT_DTYPE = ttnn.bfloat16

TILE_SIZE = 32
MODEL_HIDDEN = 4096
N_ATTENTION_HEADS = 32
GQA_GROUP_SIZE = 4
OBSERVE_SEQ_LEN = 60

inputs_from = "random"


def load_attn_weights(layer_idx=0):
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


def load_reference_inputs():
    root = os.getcwd()
    ref_data_path = os.path.join(root, "reference_data")
    example = torch.load(os.path.join(ref_data_path, "example_with_intermediates.pt"))
    ref_inputs = example["args/hidden_states"]
    print("Reference data loaded")
    return example, ref_inputs


def pcc(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.view(-1).float()
    y = y.view(-1).float()

    vx = x - x.mean()
    vy = y - y.mean()

    corr = torch.sum(vx * vy) / torch.sqrt(torch.sum(vx**2) * torch.sum(vy**2))
    return corr


def generate_random_qkv_inputs_torch(N=1, low=-0.5, high=0.5, seq_len=32, dtype=torch.bfloat16):
    random_tensors = []
    for _ in range(min(int(N), 1)):
        random_tensor = low + (high - low) * torch.rand(1, int(seq_len), MODEL_HIDDEN)
        random_tensor.to(dtype)
        random_tensors.append(random_tensor)
    print("Random torch tensors created...")
    return random_tensors


# For program configs. Sharded tensors. TT default. No reason to stick to this?
tile_padded_batch_rows = TILE_SIZE


def find_largest_divisor(n, max_divisor=8):
    for i in range(max_divisor, 0, -1):
        if n % i == 0:
            return i
    return 1


def dram_matmul_config(m: int, k: int, n: int, num_cores=None):
    if num_cores is None:
        num_cores = dram_shard_core_grid_for_k_and_n(k, n).num_cores
        assert (
            k % (TILE_SIZE * num_cores) == 0
        ), f"k must be divisible by tile_size * num_cores: {k} % {TILE_SIZE * num_cores} != 0"
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=find_largest_divisor(k // (TILE_SIZE * num_cores)),
        per_core_M=math.ceil(m / TILE_SIZE),
        per_core_N=math.ceil(n / (TILE_SIZE * num_cores)),
        fused_activation=None,
    )


def find_grid_k_n(K, N):
    """
    Find the number of rows and columns for a grid of cores such that
    the total number of tiles N can be evenly divided among the cores.
    Each core will have the same integer number of tiles.

    Parameters:
        N (int): Total number of tiles to be distributed.

    Returns:
        tuple: A tuple (rows, cols) representing the grid dimensions.

    Raises:
        AssertionError: If it's not possible to find such a grid configuration.
    """
    max_rows = 8
    max_cols = 8  # Maximum number of rows or columns
    max_cores = max_rows * max_cols  # Maximum number of cores

    # Find all possible numbers of cores that divide N and are less than or equal to max_cores
    possible_cores = [c for c in range(1, max_cores + 1) if K % c == 0 and N % c == 0]
    possible_cores.sort(reverse=True)  # Start checking from the largest number of cores

    for cores in possible_cores:
        # Try to find a grid configuration with the current number of cores
        for rows in range(1, max_rows + 1):
            if cores % rows == 0:
                cols = cores // rows
                if cols <= max_cols:
                    return rows, cols

    # If no configuration is found, assert an error
    raise AssertionError(
        f"Cannot find a grid configuration such that both {K} and {N} tiles evenly divide into cores of max size {max_rows}x{max_cols}."
    )


def dram_shard_core_grid_for_k_and_n(k: int, n: int) -> Tuple[int, int]:
    rows, cols = find_grid_k_n(k // TILE_SIZE, n // TILE_SIZE)
    return ttnn.CoreGrid(x=cols, y=rows)


mlp_core_grid_qo = dram_shard_core_grid_for_k_and_n(MODEL_HIDDEN, MODEL_HIDDEN)
matmul_prog_config_qo = dram_matmul_config(
    m=tile_padded_batch_rows,
    k=MODEL_HIDDEN,
    n=MODEL_HIDDEN,
    num_cores=mlp_core_grid_qo.num_cores,
)

mlp_core_grid_kv = dram_shard_core_grid_for_k_and_n(MODEL_HIDDEN, int(MODEL_HIDDEN / GQA_GROUP_SIZE))
matmul_prog_config_qo = dram_matmul_config(
    m=tile_padded_batch_rows,
    k=MODEL_HIDDEN,
    n=int(MODEL_HIDDEN / GQA_GROUP_SIZE),
    num_cores=mlp_core_grid_qo.num_cores,
)
# For program configs. Sharded tensors. TT default. No reason to stick to this?

# Compute kernel definitions
compute_kernel_config_hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
    # False is an order of mangnitude worse. Why?
)
# Compute kernel definitions

if __name__ == "__main__":
    print("")

    attention_weights_torch = load_attn_weights(layer_idx=0)

    # QKV load + cast to bfloat16
    q_proj_layer = nn.Linear(MODEL_HIDDEN, MODEL_HIDDEN, bias=False)
    q_proj_layer.load_state_dict({"weight": attention_weights_torch["q_proj"]})
    q_proj_layer.weight.data = q_proj_layer.weight.data.to(torch.bfloat16)

    k_proj_layer = nn.Linear(MODEL_HIDDEN, int(MODEL_HIDDEN / GQA_GROUP_SIZE), bias=False)
    k_proj_layer.load_state_dict({"weight": attention_weights_torch["k_proj"]})
    k_proj_layer.weight.data = k_proj_layer.weight.data.to(torch.bfloat16)

    v_proj_layer = nn.Linear(MODEL_HIDDEN, int(MODEL_HIDDEN / GQA_GROUP_SIZE), bias=False)
    v_proj_layer.load_state_dict({"weight": attention_weights_torch["v_proj"]})
    v_proj_layer.weight.data = v_proj_layer.weight.data.to(torch.bfloat16)
    # QKV load + cast to bfloat16

    if inputs_from == "random":
        reference_inputs = generate_random_qkv_inputs_torch()
        reference_inputs = reference_inputs[0]
        # sweep later
    else:
        _, reference_inputs = load_reference_inputs()
    # Set sequence lengths
    seq_len = min([int(reference_inputs.shape[1]), OBSERVE_SEQ_LEN])
    reference_inputs = reference_inputs[:, 0:seq_len, :]
    print("Inputs           : ", reference_inputs.shape, reference_inputs.dtype)
    # Set sequence lengths

    reference_inputs = reference_inputs.to(TORCH_IN_DTYPE)
    torch_output_q = q_proj_layer(reference_inputs).to(TORCH_OUT_DTYPE)
    torch_output_k = k_proj_layer(reference_inputs).to(TORCH_OUT_DTYPE)
    torch_output_v = v_proj_layer(reference_inputs).to(TORCH_OUT_DTYPE)

    print("")
    print("Input and parameter ranges..")
    print("Min and max (h)   : ", torch.min(reference_inputs), torch.max(reference_inputs))
    print(
        "Min and max (Q)   : ",
        torch.min(attention_weights_torch["q_proj"]),
        torch.max(attention_weights_torch["q_proj"]),
    )
    print(
        "Min and max (K)   : ",
        torch.min(attention_weights_torch["k_proj"]),
        torch.max(attention_weights_torch["k_proj"]),
    )
    print(
        "Min and max (V)   : ",
        torch.min(attention_weights_torch["v_proj"]),
        torch.max(attention_weights_torch["v_proj"]),
    )

    print("Torch outputs (q) : ", torch_output_q.shape, torch_output_q.dtype)
    print("Torch outputs (k) : ", torch_output_k.shape, torch_output_k.dtype)
    print("Torch outputs (v) : ", torch_output_v.shape, torch_output_v.dtype)
    print("")

    print("Opening device..")
    device = ttnn.open_device(device_id=0)

    reference_inputs = ttnn.from_torch(
        reference_inputs,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    print("Reference inputs pushed to device DRAM... Shape : ", reference_inputs.shape)

    Q_tt = ttnn.from_torch(
        attention_weights_torch["q_proj"],
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    print("q projection matrix pushed to device. Shape : ", Q_tt.shape)

    K_tt = ttnn.from_torch(
        attention_weights_torch["k_proj"],
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    print("k projection matrix pushed to device. Shape : ", K_tt.shape)

    V_tt = ttnn.from_torch(
        attention_weights_torch["v_proj"],
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    print("v projection matrix pushed to device. Shape : ", V_tt.shape)

    tt_output_q = ttnn.linear(
        reference_inputs,
        Q_tt,
        transpose_b=True,
        dtype=TT_OUT_DTYPE,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config_hifi4,
    )
    tt_output_q_torch = ttnn.to_torch(ttnn.from_device(tt_output_q))
    print("TT q output computed. Copied to cpu as torch.")

    tt_output_k = ttnn.linear(
        reference_inputs,
        K_tt,
        transpose_b=True,
        dtype=TT_OUT_DTYPE,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config_hifi4,
    )
    tt_output_k_torch = ttnn.to_torch(ttnn.from_device(tt_output_k))
    print("TT k output computed. Copied to cpu as torch.")

    tt_output_v = ttnn.linear(
        reference_inputs,
        V_tt,
        transpose_b=True,
        dtype=TT_OUT_DTYPE,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config_hifi4,
    )
    tt_output_v_torch = ttnn.to_torch(ttnn.from_device(tt_output_v))
    print("TT v output computed. Copied to cpu as torch.")

    atol = 1e-03
    rtol = 1e-02  # sweep to get zero error threshold

    error_pointwise = torch.abs((torch_output_q - tt_output_q_torch))
    mean_error = torch.mean(error_pointwise)
    max_error = torch.max(error_pointwise)
    print("Mean error , Max error (q) : ", mean_error, max_error)
    print("Allclose   : ", torch.allclose(tt_output_q_torch, torch_output_q, atol=atol, rtol=rtol))
    print(
        "Is close   : ",
        (torch.isclose(tt_output_q_torch, torch_output_q, atol=atol, rtol=rtol)).float().mean().item() * 100,
    )

    error_pointwise = torch.abs((torch_output_k - tt_output_k_torch))
    mean_error = torch.mean(error_pointwise)
    max_error = torch.max(error_pointwise)
    print("Mean error , Max error (k) : ", mean_error, max_error)
    print("Allclose   : ", torch.allclose(tt_output_k_torch, torch_output_k, atol=atol, rtol=rtol))
    print(
        "Is close   : ",
        (torch.isclose(tt_output_k_torch, torch_output_k, atol=atol, rtol=rtol)).float().mean().item() * 100,
    )

    error_pointwise = torch.abs((torch_output_v - tt_output_v_torch))
    mean_error = torch.mean(error_pointwise)
    max_error = torch.max(error_pointwise)
    print("Mean error , Max error (v) : ", mean_error, max_error)
    print("Allclose   : ", torch.allclose(tt_output_v_torch, torch_output_v, atol=atol, rtol=rtol))
    print(
        "Is close   : ",
        (torch.isclose(tt_output_v_torch, torch_output_v, atol=atol, rtol=rtol)).float().mean().item() * 100,
    )

    ttnn.close_device(device)
