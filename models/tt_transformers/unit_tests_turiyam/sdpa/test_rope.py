import os

import torch

import ttnn
from models.tt_transformers.tt.rope import RotarySetup

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
    print("Torch input shapes : ", rope_inputs_q_reference.shape, rope_inputs_k_reference.shape)

    rope_in_q_tt = ttnn.from_torch(
        rope_inputs_q_reference,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    rope_in_k_tt = ttnn.from_torch(
        rope_inputs_k_reference,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    rope_outputs_q_reference = example["outputs/intermediates"]["q_post_rope"]
    rope_outputs_k_reference = example["outputs/intermediates"]["k_post_rope"]
    print("Torch output shapes : ", rope_outputs_q_reference.shape, rope_outputs_k_reference.shape)

    rope_setup = RotarySetup(device, 1, 128, 8192, 500000.0, None, 8192)
    trans_mats_dict = rope_setup.get_both_trans_mats()
    print("Trans mat dict keys      : ", trans_mats_dict.keys())
    print("Trans mat dicts [prefill]: ", trans_mats_dict["prefill"].shape)

    min_position = min(position_ids)
    max_position = max(position_ids)
    print("Min and max positions : ", min_position, max_position)
    tt_rot_mats_prefill = [
        rope_setup.cos_matrix[:, :, min_position:max_position, :],
        rope_setup.sin_matrix[:, :, min_position:max_position, :],
    ]

    tt_cosines = rope_setup.cos_matrix[:, :, min_position : max_position + 1, :]
    tt_sines = rope_setup.sin_matrix[:, :, min_position : max_position + 1, :]
    torch_cosines = ttnn.to_torch(ttnn.from_device(tt_cosines))
    torch_sines = ttnn.to_torch(ttnn.from_device(tt_sines))
    print("Shape check : ", torch_cosines.shape, torch_sines.shape)

    idx = torch.tensor(position_ids, dtype=torch.long)
    print(idx)
    torch_cosines_full = torch_cosines[:, :, idx - min_position, :]
    torch_sines_full = torch_sines[:, :, idx - min_position, :]
    print("Full shapes : ", torch_cosines_full.shape, torch_sines_full.shape)

    tt_cosines_full = ttnn.from_torch(torch_cosines_full, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_sines_full = ttnn.from_torch(torch_sines_full, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    print("rot mats prefill shapes  : ", rope_setup.cos_matrix.shape, rope_setup.sin_matrix.shape)
    print("tt rot mats relevant     : ", tt_rot_mats_prefill[0].shape, tt_rot_mats_prefill[1].shape)
    q_rotated = ttnn.experimental.rotary_embedding_llama(
        rope_in_q_tt,
        cos_cache=tt_cosines_full,
        sin_cache=tt_sines_full,
        trans_mat=trans_mats_dict["prefill"],
        is_decode_mode=False,
    )

    k_rotated = ttnn.experimental.rotary_embedding_llama(
        rope_in_k_tt,
        tt_cosines_full,
        tt_sines_full,
        trans_mats_dict["prefill"],
        is_decode_mode=False,
    )
    print("Shape post Rope (Q,K) : ", q_rotated.shape, k_rotated.shape)

    q_rotated = ttnn.to_torch(ttnn.from_device(q_rotated)).to(torch.float32)
    k_rotated = ttnn.to_torch(ttnn.from_device(k_rotated)).to(torch.float32)
    errors_q = torch.abs((q_rotated - rope_outputs_q_reference))
    errors_k = torch.abs((k_rotated - rope_outputs_k_reference))
    print(torch.mean(errors_q), torch.max(errors_k), torch.mean(errors_q), torch.mean(errors_k))
