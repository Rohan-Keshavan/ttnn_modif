import torch

import ttnn

if __name__ == "__main__":
    q = torch.rand(1, 1, 32, 128)
    k = torch.rand(1, 1, 32, 128)
    v = torch.rand(1, 1, 32, 128)  # user , n_heads , seq len , head_dim
    attn_mask = torch.unsqueeze(torch.unsqueeze(torch.eye(32), dim=0), dim=0)

    device = ttnn.open_device(device_id=0)
    q_tt = ttnn.as_tensor(q, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    k_tt = ttnn.as_tensor(
        k, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    v_tt = ttnn.as_tensor(
        v, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    attn_mask_tt = ttnn.as_tensor(attn_mask, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.int32)

    print(q_tt.shape, k_tt.shape, v_tt.shape, attn_mask_tt.shape)
    print(q_tt.dtype, k_tt.dtype, v_tt.dtype, attn_mask_tt.dtype)

    attn_output = ttnn.transformer.scaled_dot_product_attention_decode(q_tt, k_tt, v_tt, is_causal=True)
    print(attn_output)

    attn_output = ttnn.transformer.scaled_dot_product_attention_decode(q_tt, k_tt, v_tt, is_causal=False)
    print(attn_output)
