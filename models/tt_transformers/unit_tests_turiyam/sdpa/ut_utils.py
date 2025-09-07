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

closeness_config = {"ATOL": 1e-03, "RTOL": 1e-02}

compute_kernel_config_hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
    # False is an order of mangnitude worse. Why?
)


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


# 4096 * 4096
# 1024 * 4096
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


def torch_to_tt(x, device):
    return


def tt_to_torch(x, on_device=True):
    return ttnn.to_torch(ttnn.from_device(x)) if on_device else ttnn.to_torch(x)


def pcc(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.view(-1).float()
    y = y.view(-1).float()
    vx = x - x.mean()
    vy = y - y.mean()
    corr = torch.sum(vx * vy) / torch.sqrt(torch.sum(vx**2) * torch.sum(vy**2))
    return corr


def compare(x: torch.Tensor, y: torch.Tensor, atol=closeness_config["ATOL"], rtol=closeness_config["RTOL"]):
    # pcc , all close, is close ( fail %s )
    # expect x and y to be torch tensors : y is reference, x is input
    errors = torch.abs((y - x))
    mean_error = torch.mean(errors)
    max_error = torch.max(errors)
    print("torch allclose           : ", torch.allclose(y, x, atol=atol, rtol=rtol))
    print("Mean and Max L1 errors   : ", mean_error, max_error)
    mask = torch.abs(y - x) > (atol + rtol * torch.abs(y))
    fail_fraction = mask.float().mean().item()
    fail_percent = fail_fraction * 100
    print("Fail percent             : ", fail_percent)
    print("Pcc                      : ", pcc(y, x))
    return
