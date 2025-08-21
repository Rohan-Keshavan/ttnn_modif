import torch

import ttnn

MODEL_HIDDEN = 8192
N_QHEADS = 64
N_KVHEADS = 64  # 8 later
HEAD_DIM = 128
MAX_SEQ_LEN = 2048
TILE_SIZE = 32
BATCH_SIZE = 32


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
    k_cache = torch.zeros(MAX_SEQ_LEN, 1, N_KVHEADS, HEAD_DIM)
    k_tt = ttnn.as_tensor(
        k_cache, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )

    v_cache = torch.zeros(MAX_SEQ_LEN, 1, N_KVHEADS, HEAD_DIM)
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
    attn_mask_tt = ttnn.as_tensor(attn_mask, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.int32)

    return [q_tt, k_tt, v_tt, attn_mask_tt]


def generate_random_embeddings_and_mask(in_seq_len, device):
    attn_input = torch.rand(BATCH_SIZE, in_seq_len, MODEL_HIDDEN)
    attn_mask = torch.rand(BATCH_SIZE, 1, in_seq_len, in_seq_len)
    attn_inputs_tt = ttnn.as_tensor(
        attn_input, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    attn_mask_tt = ttnn.as_tensor(
        attn_mask, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    return [attn_inputs_tt, attn_mask_tt]


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

    print("")
    print("Generating and pushing random QKVO matrices")
    [Q, K, V, O] = QKVO_generate_and_push_random(device=device)
    print("Shapes : (q,k,v,o) : ", Q.shape, K.shape, V.shape, O.shape)

    print("")
    print("KV Init")
    [K_past, V_past, kv_len] = init_kv_cache(device=device)
    print("Shape  : (k cache, v cache) : ", K_past.shape, V_past.shape)

    print("")
    print("Generating random embeddings and mask")
    in_seq_len = 16
    E, M = generate_random_embeddings_and_mask(in_seq_len=in_seq_len, device=device)
    print("E , M  : ", E.shape, M.shape)

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

    # q_proj = ttnn.reshape_on_device(q_proj, BATCH_SIZE, in_seq_len , N_QHEADS, HEAD_DIM)
    # q_proj = ttnn.transpose(q_proj, 1, -2)
    # print('Q reshaped, headed : ', q_proj.shape)

    ttnn.close_device(device)
