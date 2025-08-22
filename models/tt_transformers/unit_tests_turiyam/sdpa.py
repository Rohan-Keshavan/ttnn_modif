import torch

import ttnn

MODEL_HIDDEN = 8192

N_QHEADS = 64
N_KVHEADS = 64  # 8 later
HEAD_DIM = 128

MAX_SEQ_LEN = 1024

TILE_SIZE = 32
BATCH_SIZE = 1
START_KV_LEN = 26
ATTN_SCALE = 1 / (HEAD_DIM**0.5)


# Reshaped
def QKVO_generate_and_push_random(device):
    Q_random = torch.rand(1, 1, HEAD_DIM * N_QHEADS, MODEL_HIDDEN)
    K_random = torch.rand(1, 1, HEAD_DIM * N_KVHEADS, MODEL_HIDDEN)
    V_random = torch.rand(1, 1, HEAD_DIM * N_KVHEADS, MODEL_HIDDEN)
    O_random = torch.rand(1, 1, MODEL_HIDDEN, MODEL_HIDDEN)

    Q_tt = ttnn.as_tensor(
        Q_random, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    K_tt = ttnn.as_tensor(
        K_random, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    V_tt = ttnn.as_tensor(
        V_random, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    O_tt = ttnn.as_tensor(
        O_random, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )

    return [Q_tt, K_tt, V_tt, O_tt]


def init_kv_cache(device):
    # (seq_len,n_heads,head_dim) , #Need a current_kv_len
    # kv in row major layout
    k_cache = torch.zeros(START_KV_LEN, 1, N_KVHEADS, HEAD_DIM)
    k_tt = ttnn.as_tensor(
        k_cache, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )

    v_cache = torch.zeros(START_KV_LEN, 1, N_KVHEADS, HEAD_DIM)
    v_tt = ttnn.as_tensor(
        v_cache, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    current_kv_length = 0
    return [k_tt, v_tt, current_kv_length]


def update_kv_cache(kv_cache, kv_new, current_kv_len):
    return


def tt_llama_sdpa(x: ttnn._ttnn.tensor.Tensor, mask: ttnn._ttnn.tensor.Tensor, kv: ttnn._ttnn.tensor.Tensor):
    # Load Q, K ,V weights : Convert to device tensors

    # q,k,v compute
    # k extend, v extend (Pre or post RoPE ?)
    # RoPE (q,k) | draft tree
    # attn | kv
    # mask apply
    # output projection
    # post verify : kv.update (kv.cat , kv.remove)
    # paged vs non paged
    return


def generate_qkv_sequence_and_mask(in_seq_len, device):
    q = torch.rand(BATCH_SIZE, 1, in_seq_len, HEAD_DIM)
    k = torch.rand(BATCH_SIZE, 1, in_seq_len, HEAD_DIM)
    v = torch.rand(BATCH_SIZE, 1, in_seq_len, HEAD_DIM)  # user , n_heads , seq len , head_dim
    attn_mask = torch.unsqueeze(torch.unsqueeze(torch.eye(in_seq_len), dim=0), dim=0)  # Implicit head broadcasting
    # And keep same for all 'users'

    q_tt = ttnn.as_tensor(
        q, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    k_tt = ttnn.as_tensor(
        k, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    v_tt = ttnn.as_tensor(
        v, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    attn_mask_tt = ttnn.as_tensor(attn_mask, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.float32)

    return [q_tt, k_tt, v_tt, attn_mask_tt]


def generate_random_embeddings_and_mask(in_seq_len, device):
    attn_input = torch.rand(BATCH_SIZE, in_seq_len, MODEL_HIDDEN)
    attn_mask = torch.rand(BATCH_SIZE, 1, in_seq_len, in_seq_len)
    attn_inputs_tt = ttnn.as_tensor(
        attn_input, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    # attn_mask_tt = ttnn.as_tensor(
    #    attn_mask, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    # )
    return [attn_inputs_tt, attn_mask]


if __name__ == "__main__":
    print("Opening device")
    device = ttnn.open_device(device_id=0)

    [q_tt, k_tt, v_tt, attn_mask_tt] = generate_qkv_sequence_and_mask(28, device=device)
    print("")
    print("Shapes : (q,k,v,mask)", q_tt.shape, k_tt.shape, v_tt.shape, attn_mask_tt.shape)
    print("DTypes : (q,k,v,mask)", q_tt.dtype, k_tt.dtype, v_tt.dtype, attn_mask_tt.dtype)
    print("PTypes : (q,k,v,mask)", type(q_tt), type(k_tt), type(v_tt), type(attn_mask_tt))
    attn_output = ttnn.transformer.scaled_dot_product_attention(q_tt, k_tt, v_tt, is_causal=True)
    print("SDPA Output : ", attn_output.shape)

    q_tt.deallocate(True)
    k_tt.deallocate(True)
    v_tt.deallocate(True)
    attn_mask_tt.deallocate(True)

    print("")
    print("Generating and pushing random QKVO matrices")
    [Q, K, V, O] = QKVO_generate_and_push_random(device=device)
    print("Shapes : (q,k,v,o) : ", Q.shape, K.shape, V.shape, O.shape)

    print("")
    print("KV Init")
    [K_past, V_past, kv_len] = init_kv_cache(device=device)
    # print("Shape  : (k cache, v cache) : ", K_past.shape, V_past.shape)

    print("")
    print("Generating random embeddings and mask")
    in_seq_len = 16
    E, M = generate_random_embeddings_and_mask(in_seq_len=in_seq_len, device=device)
    print("E , M  : ", E.shape, M.shape)
    current_sequence_length = K_past.shape[0]
    draft_sequence_length = E.shape[-2]
    kv_acces_indices = ttnn.arange(start=0, end=current_sequence_length + draft_sequence_length, dtype=ttnn.int32)

    print("Multiplying E , Q : ", E.shape, Q.shape)
    # q_proj = ttnn.matmul(E,Q)
    q_proj = ttnn.linear(E, Q)
    print("Multiplying E , K : ", E.shape, K.shape)
    # k_proj = ttnn.matmul(E,K)
    k_proj = ttnn.linear(E, K)
    print("Multiplying E , V : ", E.shape, V.shape)
    # v_proj = ttnn.matmul(E,V)
    v_proj = ttnn.linear(E, V)
    print("Q proj , K proj , V proj : ", q_proj.shape, k_proj.shape, v_proj.shape)

    E.deallocate(True)
    q_proj_reshaped = ttnn.reshape(q_proj, (BATCH_SIZE, in_seq_len, N_QHEADS, HEAD_DIM))
    q_proj_reshaped = ttnn.transpose(q_proj_reshaped, 0, 1)
    q_proj.deallocate(True)

    k_proj_reshaped = ttnn.reshape(k_proj, (BATCH_SIZE, in_seq_len, N_QHEADS, HEAD_DIM))
    k_proj_reshaped = ttnn.transpose(k_proj_reshaped, 0, 1)
    k_proj.deallocate(True)

    v_proj_reshaped = ttnn.reshape(v_proj, (BATCH_SIZE, in_seq_len, N_QHEADS, HEAD_DIM))
    v_proj_reshaped = ttnn.transpose(v_proj_reshaped, 0, 1)
    v_proj.deallocate(True)
    print("Re heading")
    print("Reshaped (q , k, v): ", q_proj_reshaped.shape, k_proj_reshaped.shape, v_proj_reshaped.shape)

    print("")
    kv_len += draft_sequence_length

    # RoPE q and k
    # RoPE q and k

    # Add to cache, k and v
    print("K past shape before update : ", K_past.shape)
    K_past = ttnn.concat([K_past, k_proj_reshaped], dim=0)
    V_past = ttnn.concat([V_past, v_proj_reshaped], dim=0)
    print("K past shape after update : ", K_past.shape)
    v_proj_reshaped.deallocate(True)
    k_proj_reshaped.deallocate(True)
    # Add to cache, k and v

    # Reshape and fill attn mask
    print("")
    print("Making rectangular mask")
    M_prefix = torch.ones(BATCH_SIZE, 1, draft_sequence_length, current_sequence_length)
    M_full = torch.cat([M_prefix, M], dim=-1)
    M_full = torch.clamp(torch.log(M_full), min=-1e06)
    M_full = M_full.repeat(1, N_QHEADS, 1, 1)
    M_full_tt = ttnn.as_tensor(
        M_full, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    print("Rectangular mask shape : ", M_full_tt.shape)
    # print('Rectangular mask       : ', M_full_tt)
    # Reshape and fill attn mask

    # Use cache and do attention
    print("Attn computation")
    print("QKT current in (q,k)  : ", q_proj_reshaped.shape, K_past.shape)
    q_proj_reshaped = ttnn.permute(q_proj_reshaped, (1, 2, 0, 3))
    K_past = ttnn.permute(K_past, (1, 2, 3, 0))
    print("QKT in reshaped (q,k) : ", q_proj_reshaped.shape, K_past.shape)
    QKT = ttnn.matmul(q_proj_reshaped, K_past)
    QKT = QKT * ATTN_SCALE
    QKT += M_full_tt
    QKT = ttnn.softmax(QKT, dim=-1)
    print("QKT out (QKT) : ", QKT.shape)
    q_proj_reshaped.deallocate(True)

    print("")
    V_past = ttnn.permute(V_past, (1, 2, 0, 3))
    print("V past shape   : ", V_past.shape)
    attn_out = ttnn.matmul(QKT, V_past)
    print("Attn out shape : ", attn_out.shape)
    # Use cache and do attention

    print("")
    attn_out = ttnn.to_layout(attn_out, layout=ttnn.ROW_MAJOR_LAYOUT)
    attn_out = ttnn.reshape(attn_out, (BATCH_SIZE, draft_sequence_length, MODEL_HIDDEN))
    attn_out = ttnn.to_layout(attn_out, layout=ttnn.TILE_LAYOUT)
    print("Attn out reshaped : ", attn_out.shape)
    layer_out = ttnn.linear(attn_out, O)
    print("Final out : ", layer_out.shape)
    # Reshape K and Q
    ttnn.close_device(device)
