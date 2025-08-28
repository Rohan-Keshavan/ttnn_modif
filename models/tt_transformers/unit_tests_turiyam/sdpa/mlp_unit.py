import os
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM

import ttnn


def load_attn_weights(layer_idx=0):
    model_path = os.getenv("HF_MODEL")
    model_path = Path(model_path)
    print(f"Loading model from local weights: {model_path}")

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        local_files_only=True,
    )

    print("Model loaded successfully from local weights.")

    # Extract attention weights
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
    """
    interest        = "decoder_0.pt"
    root            = os.getcwd()
    ref_data_path   = os.path.join(root, "reference_data")
    ref_data_file   = os.path.join(ref_data_path, interest)
    ref_data        = torch.load(ref_data_file)
    example         = ref_data[4]
    ref_inputs      = example['args/hidden_states']
    """
    root = os.getcwd()
    ref_data_path = os.path.join(root, "reference_data")
    example = torch.load(os.path.join(ref_data_path, "example_with_intermediates.pt"))
    ref_inputs = example["args/hidden_states_post_norm"]
    print("Reference data loaded")
    return example, ref_inputs


if __name__ == "__main__":
    print("")
    attention_weights_torch = load_attn_weights()

    device = ttnn.open_device(device_id=0)

    q_proj_layer = nn.Linear(4096, 32 * 128, bias=False)
    q_proj_layer.load_state_dict({"weight": attention_weights_torch["q_proj"]})

    k_proj_layer = nn.Linear(4096, 8 * 128, bias=False)
    k_proj_layer.load_state_dict({"weight": attention_weights_torch["k_proj"]})

    v_proj_layer = nn.Linear(4096, 8 * 128, bias=False)
    v_proj_layer.load_state_dict({"weight": attention_weights_torch["v_proj"]})

    reference_example, reference_inputs = load_reference_inputs()
    seq_len = min([int(reference_inputs.shape[1]), 60])
    reference_inputs = reference_inputs[:, 0:seq_len, :]
    print("Obtained inputs : ", reference_inputs.shape, reference_inputs.dtype)

    reference_inputs = reference_inputs.to(torch.float)
    torch_output_q = q_proj_layer(reference_inputs)  # Needs f32 in
    torch_output_k = k_proj_layer(reference_inputs)  # Needs f32 in
    torch_output_v = v_proj_layer(reference_inputs)  # Needs f32 in

    print("Torch outputs (q) : ", torch_output_q.shape, torch_output_q.dtype)
    print("Torch outputs (k) : ", torch_output_k.shape, torch_output_k.dtype)
    print("Torch outputs (v) : ", torch_output_v.shape, torch_output_v.dtype)

    print("")
    reference_example, reference_inputs = load_reference_inputs()
    reference_inputs = reference_inputs[:, 0:seq_len, :]
    print("Obtained inputs : ", reference_inputs.shape, reference_inputs.dtype)

    print("Opening device...")
    compute_kernel_config_hifi4 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    out_data_dype = ttnn.bfloat16

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
        dtype=out_data_dype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config_hifi4,
    )
    tt_output_q_torch = ttnn.to_torch(ttnn.from_device(tt_output_q))
    tt_output_q_torch = tt_output_q_torch.to(dtype=torch.float)
    print("TT output computed. Copied to cpu as torch.")

    tt_output_k = ttnn.linear(
        reference_inputs,
        K_tt,
        transpose_b=True,
        dtype=out_data_dype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config_hifi4,
    )
    tt_output_k_torch = ttnn.to_torch(ttnn.from_device(tt_output_k))
    tt_output_k_torch = tt_output_k_torch.to(dtype=torch.float)
    print("TT output computed. Copied to cpu as torch.")

    tt_output_v = ttnn.linear(
        reference_inputs,
        V_tt,
        transpose_b=True,
        dtype=out_data_dype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config_hifi4,
    )
    tt_output_v_torch = ttnn.to_torch(ttnn.from_device(tt_output_v))
    tt_output_v_torch = tt_output_v_torch.to(dtype=torch.float)
    print("TT output computed. Copied to cpu as torch.")

    atol = 1e-04
    rtol = 1e-02

    error_pointwise = torch.abs((torch_output_q - tt_output_q_torch))
    mean_error = torch.mean(error_pointwise)
    max_error = torch.max(error_pointwise)
    print("Mean error , Max error (q) : ", mean_error, max_error)
    print("Allclose   : ", torch.allclose(tt_output_q_torch, torch_output_q, atol=atol, rtol=rtol))
    # elementwise check (like allclose does internally)
    mask = torch.abs(torch_output_q - tt_output_q_torch) > (atol + rtol * torch.abs(tt_output_q_torch))
    # fraction (or %) of elements failing the criterion
    fail_fraction = mask.float().mean().item()
    fail_percent = fail_fraction * 100
    print("Fail percent : ", fail_percent)

    error_pointwise = torch.abs((torch_output_k - tt_output_k_torch))
    mean_error = torch.mean(error_pointwise)
    max_error = torch.max(error_pointwise)
    print("Mean error , Max error (k) : ", mean_error, max_error)
    print("Allclose   : ", torch.allclose(tt_output_k_torch, torch_output_k, atol=atol, rtol=rtol))
    # elementwise check (like allclose does internally)
    mask = torch.abs(torch_output_k - tt_output_k_torch) > (atol + rtol * torch.abs(tt_output_k_torch))
    # fraction (or %) of elements failing the criterion
    fail_fraction = mask.float().mean().item()
    fail_percent = fail_fraction * 100
    print("Fail percent : ", fail_percent)

    error_pointwise = torch.abs((torch_output_v - tt_output_v_torch))
    mean_error = torch.mean(error_pointwise)
    max_error = torch.max(error_pointwise)
    print("Mean error , Max error (v) : ", mean_error, max_error)
    print("Allclose   : ", torch.allclose(tt_output_v_torch, torch_output_v, atol=atol, rtol=rtol))
    # elementwise check (like allclose does internally)
    mask = torch.abs(torch_output_v - tt_output_v_torch) > (atol + rtol * torch.abs(tt_output_v_torch))
    # fraction (or %) of elements failing the criterion
    fail_fraction = mask.float().mean().item()
    fail_percent = fail_fraction * 100
    print("Fail percent : ", fail_percent)

    ttnn.close_device(device)
