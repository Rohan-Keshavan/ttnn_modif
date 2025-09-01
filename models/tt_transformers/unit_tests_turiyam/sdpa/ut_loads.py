import json
import os

import torch

import ttnn
from models.tt_transformers.tt.rope import RotarySetup


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
    # print("Trans mat dicts [prefill]: ", trans_mats_dict["prefill"].shape, trans_mats_dict["decode"].shape)
    # prefill_trans_mat_dict = torch.squeeze(ttnn.to_torch(ttnn.from_device(trans_mats_dict["prefill"])))
    # print('Trans mat dict - prefill')
    # for j in range(0,32):
    #    print(prefill_trans_mat_dict[j,:])
    min_position = min(position_ids)
    max_position = max(position_ids)
    n_seq = len(position_ids)
    print("Min and max positions : ", min_position, max_position)
    # Pre generate cosine and sines
    tt_rot_mats_prefill = [
        rope_setup.cos_matrix[:, :, min_position : max_position + 1, :],
        rope_setup.sin_matrix[:, :, min_position : max_position + 1, :],
    ]
    print("Rotation mats shape : ", tt_rot_mats_prefill[0].shape, tt_rot_mats_prefill[1].shape)

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

    # torch_cosines_full  = torch_cosines_full.to(torch.bfloat16)
    # torch_sines_full    = torch_sines_full.to(torch.bfloat16)
    tt_cosines_full = ttnn.from_torch(torch_cosines_full, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32)
    tt_sines_full = ttnn.from_torch(torch_sines_full, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32)
