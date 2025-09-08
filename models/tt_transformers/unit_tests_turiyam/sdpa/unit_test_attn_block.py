import gzip
import os

import torch
from ut_utils import *


def load_pt_gz(file_path):
    with gzip.open(file_path, "rb") as f:
        data = torch.load(f, map_location="cpu")
        print("")
        print("Loaded data type : ", type(data))
        if isinstance(data, dict):
            print("Load loaded a dict.")
    return data


def get_qkvo_reference(data):
    qkvo_reference = {}
    qkvo_reference["qkv_in"] = data["layer_in"]
    qkvo_reference["o_in"] = data["attn_out_vproj"]
    bsz, nheads, seq_len, head_dim = qkvo_reference["o_in"].shape
    qkvo_reference["o_in"] = qkvo_reference["o_in"].permute(0, 2, 1, 3).reshape(bsz, seq_len, nheads * head_dim)
    qkvo_reference["qkv_q_out"] = data["qkv_q"]
    qkvo_reference["qkv_k_out"] = data["qkv_k"]
    qkvo_reference["qkv_v_out"] = data["qkv_v"]
    qkvo_reference["o_out"] = data["layer_out"]
    return qkvo_reference


def get_rope_reference(data):
    rope_reference = {}
    rope_reference["pos_ids"] = data["position_ids"]
    rope_reference["q_in"] = data["qkv_q"]
    rope_reference["k_in"] = data["qkv_k"]
    rope_reference["cos"] = data["rope_cos"]
    rope_reference["sin"] = data["rope_sin"]
    rope_reference["q_out"] = data["roped_q"]
    rope_reference["k_out"] = data["roped_k"]
    return rope_reference


def get_attn_reference(data):
    attn_reference = {}
    attn_reference["k_cache"] = data["past_kv_k"]  # bf16
    attn_reference["v_cache"] = data["past_kv_v"]  # bf16
    attn_reference["q_rotated"] = data["roped_q"]  # bf16
    attn_reference["k_rotated"] = data["roped_k"]  # bf16
    attn_reference["v"] = data["qkv_v"]  # bf16
    attn_reference["mask"] = data["attention_mask"]  # f32
    attn_reference["qkt_scaled"] = data["attn_weights"]  # bf16 -> f32
    attn_reference["qkt_scaled_masked"] = data["attn_weights_masked"]  # f32
    attn_reference["qkt_scaled_masked_softmaxed"] = data["attn_weights_softmaxed"]  # f32
    attn_reference["qkt_scaled_masked_softmaxed_downcasted"] = data["attn_weights_softmaxed_down_casted"]  # bf16
    attn_reference["qkt_v_proj"] = data["attn_out_vproj"]  # bf16

    return attn_reference


def get_attn_block_reference(data):
    attn_block_reference = {}
    attn_block_reference["inputs"] = data["layer_in"]
    attn_reference["k_cache"] = data["past_kv_k"]
    attn_reference["v_cache"] = data["past_kv_v"]
    attn_block_reference["outputs"] = data["layer_out"]
    return attn_block_reference


nvidia_data_path = "reference_data_nvidia"
sub_path = "bfloat16_ev3_validation"

if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)

    root = os.getcwd()
    target_folder = os.path.join(root, nvidia_data_path, sub_path)
    gz_path = "layer_0_target_eagle_example_640.pt.gz"
    x = load_pt_gz(os.path.join(target_folder, gz_path))

    layer_0_weights = load_attn_weights_llama_3()
    qkvo_reference = get_qkvo_reference(x)
    attn_reference = get_attn_reference(x)
    rope_reference = get_rope_reference(x)
    # Qx (Isolated)
    print("")
    print("Q projection")
    Qx, x_stats, Qx_stats = get_torch_linear_out_llama_3(qkvo_reference["qkv_in"], layer_0_weights["q_proj"])
    Qx_tt, Qx_tt_stats = get_tt_linear_out_llama_3(qkvo_reference["qkv_in"], layer_0_weights["q_proj"], device=device)
    compare(Qx_tt, Qx)

    # Kx (Isolated)
    print("")
    print("K projection")
    Kx, x_stats, Kx_stats = get_torch_linear_out_llama_3(qkvo_reference["qkv_in"], layer_0_weights["k_proj"])
    Kx_tt, Kx_tt_stats = get_tt_linear_out_llama_3(qkvo_reference["qkv_in"], layer_0_weights["k_proj"], device=device)
    compare(Kx_tt, Kx)

    # Vx (Isolated)
    print("")
    print("V projection")
    Vx, x_stats, Vx_stats = get_torch_linear_out_llama_3(qkvo_reference["qkv_in"], layer_0_weights["v_proj"])
    Vx_tt, Vx_tt_stats = get_tt_linear_out_llama_3(qkvo_reference["qkv_in"], layer_0_weights["v_proj"], device=device)
    compare(Vx_tt, Vx)

    # Oattn (Isolated)
    print("")
    print("O projection")
    Ox, x_stats, Ox_stats = get_torch_linear_out_llama_3(qkvo_reference["o_in"], layer_0_weights["o_proj"])
    Ox_tt, Ox_tt_stats = get_tt_linear_out_llama_3(qkvo_reference["o_in"], layer_0_weights["o_proj"], device=device)
    compare(Ox_tt, Ox)

    # attn  :Errors are cascaded (intermediates)
    print("")
    print("Attention")
    print("")
    print("At Attn weights")
    tt_intermediates = get_tt_attn_out_llama_3(
        attn_reference["k_cache"],
        attn_reference["v_cache"],
        attn_reference["q_rotated"],
        attn_reference["k_rotated"],
        attn_reference["v"],
        attn_reference["mask"],
        device,
    )
    tt_qkt_past = (tt_to_torch(tt_intermediates["qkt_past"], on_device=True)).to(dtypes_config["outputs_torch"])
    tt_qkt_current = (tt_to_torch(tt_intermediates["qkt_current"], on_device=True)).to(dtypes_config["outputs_torch"])
    tt_attn_weights = torch.cat([tt_qkt_past, tt_qkt_current], dim=-1)
    compare(tt_attn_weights, attn_reference["qkt_scaled"])
    print("")
    print("At Attn scaling  + softmax (fp32)")
    tt_qkt_softmaxed = tt_to_torch(tt_intermediates["qkt_softmaxed"], on_device=True)
    compare(tt_qkt_softmaxed, attn_reference["qkt_scaled_masked_softmaxed"])
    print("")
    print("At qktv")
    tt_qktv = tt_to_torch(tt_intermediates["qktv"], on_device=True).to(dtypes_config["outputs_torch"])
    compare(tt_qktv, attn_reference["qkt_v_proj"])

    """
    # RoPE
    print('')
    print('RoPE')
    print('cos and sines')
    cos_tt , sin_tt         = get_torch_sin_cos_tt(rope_reference['pos_ids'].squeeze(0))
    cos_tt , sin_tt         = cos_tt.squeeze(0) , sin_tt.squeeze(0)
    #Implementation can be different. Check final output (q_rotated and k_rotated)
    compare(cos_tt, rope_reference['cos'])
    compare(sin_tt, rope_reference['sin'])
    print('RoPE q and k')
    qr_tt , kr_tt = tt_rope_llama3(rope_reference['pos_ids'].squeeze(0),rope_reference['q_in'],rope_reference['k_in'],device=device)
    compare(qr_tt,rope_reference['q_out'])
    compare(kr_tt,rope_reference['k_out'])
    """
