import gzip
import os

import torch


def load_pt_gz(file_path):
    with gzip.open(file_path, "rb") as f:
        data = torch.load(f, map_location="cpu")
        print("")
        print("Loaded data type : ", type(data))
        if isinstance(data, dict):
            print("Load loaded a dict. Keys -> Shapes -> Types")
            for key, value in data.items():
                print(key, value.shape, value.dtype)
    return data


def get_qkvo_reference(data):
    qkvo_reference = {}
    qkvo_reference["qkv_in"] = data["layer_in"]
    qkvo_reference["o_in"] = data["attn_out_vproj"]
    qkvo_reference["qkv_q_out"] = data["qkv_q"]
    qkvo_reference["qkv_k_out"] = data["qkv_k"]
    qkvo_reference["qkv_v_out"] = data["qkv_v"]
    qkvo_reference["o_out"] = data["layer_out"]
    return qkvo_reference


def get_rope_reference(data):
    rope_reference = {}
    rope_reference["q_in"] = data["qkv_q"]
    rope_reference["k_in"] = data["qkv_k"]
    rope_reference["cos"] = data["cos"]
    rope_reference["sin"] = data["sin"]
    rope_reference["q_out"] = data["q_rotated"]
    rope_reference["k_out"] = data["k_rotated"]
    return rope_reference


def get_attn_reference(data):
    attn_reference = {}
    attn_reference["q_rotated"] = data["q_rotated"]
    attn_reference["k_rotated"] = data["k_rotated"]
    attn_reference["v"] = data["qkv_v"]
    attn_reference["mask"] = data["attention_mask"]

    attn_reference["qkt_scaled"] = data["attn_weights"]
    attn_reference["qkt_scaled_masked"] = data["attn_weights_masked"]
    attn_reference["qkt_scaled_masked_softmaxed"] = data["attn_weights_softmaxed"]
    attn_reference["qkt_scaled_masked_softmaxed_downcasted"] = data["attn_weights_softmaxed_downcasted"]
    attn_reference["`qkt_v_proj"] = data["attn_out_vproj"]

    return attn_reference


def attn_block_reference(data):
    attn_block_reference = {}
    attn_block_reference["inputs"] = data["layer_in"]
    attn_block_reference["outputs"] = data["layer_out"]
    return attn_block_reference


nvidia_data_path = "reference_data_nvidia"
sub_path = "bfloat16_ev3_validation"
if __name__ == "__main__":
    root = os.getcwd()
    target_folder = os.path.join(root, nvidia_data_path, sub_path)
    gz_path = "layer_0_target_eagle_example_640.pt.gz"
    load_pt_gz(os.path.join(target_folder, gz_path))
