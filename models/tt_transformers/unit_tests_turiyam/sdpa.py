import torch

import ttnn
from models.tt_transformers.tt.common import gather_cos_sin, get_rot_transformation_mat, precompute_freqs
from models.tt_transformers.tt.rope import RotarySetup

# get from model config file. override this
MODEL_HIDDEN = 8192
N_QHEADS = 64
N_KVHEADS = 64  # 8 later
HEAD_DIM = 128

# override config file for spec decode
MAX_SEQ_LEN = 1024

# ttnn
TILE_SIZE = 32
BATCH_SIZE = 1
MAX_BATCH_SIZE = 1
START_KV_LEN = 26
ATTN_SCALE = 1 / (HEAD_DIM**0.5)

# load model config file
hf_model_config = {"rope_theta": 500000.0, "original_max_position_embeddings": 8192}
# load model config file

# LLaMA 3.3 70B RoPE configuration
# LLaMA 3.3 uses RoPE scaling with factor=8 and original context length=8192
hf_model_config = {
    "rope_theta": 500000.0,  # Base frequency for RoPE
    "original_max_position_embeddings": 8192,  # Original context length before scaling
    "rope_scaling_factor": 8,  # Scaling factor for extended context
    "max_position_embeddings": 128 * 1024,  # Extended context length (128K)
}


# Generate random weights : QKVO
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


# Generate random weights : QKVO


# Init KV cache : Pre allocate max model len, push to device DRAM
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


# Init KV cache : Pre allocate max model len, push to device DRAM


# Fill in : .cat and .remove , .head_position
def update_kv_cache(kv_cache, kv_new, current_kv_len):
    return


# Fill in : .cat and .remove , .head_position


# Model forward. Fill in
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


# Model forward. Fill in


# Generate random attention inputs: qkvmask
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


# Generate random attention inputs: qkvmask


# Generate random inputs, attention mask. Push inputs to device DRAM
def generate_random_embeddings_and_mask(in_seq_len, device):
    attn_input = torch.rand(BATCH_SIZE, in_seq_len, MODEL_HIDDEN)
    attn_mask = torch.rand(BATCH_SIZE, 1, in_seq_len, in_seq_len)
    attn_inputs_tt = ttnn.as_tensor(
        attn_input, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    return [attn_inputs_tt, attn_mask]


# Generate random inputs, attention mask. Push inputs to device DRAM


# Look in the data/models folder for the config file. Or just pick up from the env variable
def find_llama_models():
    """
    Utility function to find LLaMA models in common locations

    Returns:
        list: List of found LLaMA model paths
    """
    import os
    from pathlib import Path

    found_models = []

    # Check HuggingFace cache directory
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache.exists():
        for model_dir in hf_cache.iterdir():
            if model_dir.is_dir() and "llama" in model_dir.name.lower():
                config_file = model_dir / "config.json"
                if config_file.exists():
                    found_models.append(str(model_dir))

    # Check current directory
    current_dir = Path.cwd()
    for item in current_dir.iterdir():
        if item.is_dir() and "llama" in item.name.lower():
            config_file = item / "config.json"
            if config_file.exists():
                found_models.append(str(item))

    # Check common model directories
    common_paths = ["/mnt/MLPerf/tt_dnn-models/llama", "/home/llama-data", "/proj_sw/user_dev/llama"]

    for path in common_paths:
        if os.path.exists(path):
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path) and "llama" in item.lower():
                    config_file = os.path.join(item_path, "config.json")
                    if os.path.exists(config_file):
                        found_models.append(item_path)

    return found_models


# Look in the data/models folder for the config file. Or just pick up from the env variable


# Check llama path
def check_llama_availability(model_name=None):
    """
    Check if LLaMA model weights exist locally using HF_MODEL environment variable

    Args:
        model_name: Optional model name override (defaults to HF_MODEL env var)

    Returns:
        bool: True if model weights exist locally, False otherwise
    """
    import os
    from pathlib import Path

    # Get model path from HF_MODEL environment variable
    if model_name is None:
        model_name = os.getenv("HF_MODEL")
        if not model_name:
            print("❌ HF_MODEL environment variable not set")
            print("   Please set: export HF_MODEL='/path/to/your/llama/model'")
            return False

    model_path = Path(model_name)

    print(f"🔍 Checking model path: {model_path}")

    # Check if the directory exists
    if not model_path.exists():
        print(f"❌ Model directory does not exist: {model_path}")
        return False

    # Check for config.json
    config_file = model_path / "config.json"
    if not config_file.exists():
        print(f"❌ config.json not found at: {config_file}")
        return False

    # Check for safetensors files
    safetensor_files = list(model_path.glob("*.safetensors"))
    if not safetensor_files:
        # Also check for model.safetensors.index.json
        index_file = model_path / "model.safetensors.index.json"
        if not index_file.exists():
            print(f"❌ No safetensors files found at: {model_path}")
            print("   Expected: *.safetensors files or model.safetensors.index.json")
            return False
        else:
            print(f"✅ Found safetensors index file: {index_file}")

    print(f"✅ Found local model weights at: {model_path}")
    print(f"   - Config file: {config_file}")
    if safetensor_files:
        print(f"   - Safetensors files: {len(safetensor_files)} found")
        for sf in safetensor_files[:3]:  # Show first 3
            print(f"     - {sf.name}")
        if len(safetensor_files) > 3:
            print(f"     ... and {len(safetensor_files) - 3} more")

    # Try to read basic config info
    try:
        import json

        with open(config_file, "r") as f:
            config = json.load(f)

        print(f"   - Model type: {config.get('model_type', 'N/A')}")
        print(f"   - Hidden size: {config.get('hidden_size', 'N/A')}")
        print(f"   - Num layers: {config.get('num_hidden_layers', 'N/A')}")
        print(f"   - Num heads: {config.get('num_attention_heads', 'N/A')}")
        print(f"   - Num KV heads: {config.get('num_key_value_heads', 'N/A')}")
        return True
    except Exception as e:
        print(f"   - Config read error: {e}")
        # Still return True if the files exist
        return True


# Check llama path


# Llama load
def load_llama_attention_weights(device, model_name=None, layer_idx=0):
    """
    Load LLaMA attention weights for a specific decoder block

    Args:
        device: TT-Metal device (single device or T3K)
        model_name: Optional model name override (defaults to HF_MODEL env var)
        layer_idx: Decoder block index (0 for first block)

    Returns:
        dict: Dictionary containing Q, K, V, O weights for the attention block

    Note:
        This function requires:
        1. HF_MODEL environment variable set to model path
        2. LLaMA model weights with config.json and safetensors files
        3. Sufficient disk space for model weights (~140GB)
        4. No internet connection required - only uses local files
    """

    try:
        # For single device or T3K, we'll use a simpler approach
        # Load weights from local directory
        import os
        from pathlib import Path

        import torch
        from transformers import AutoModelForCausalLM

        # Get model path from HF_MODEL environment variable
        if model_name is None:
            model_name = os.getenv("HF_MODEL")
            if not model_name:
                raise ValueError(
                    "HF_MODEL environment variable not set. Please set: export HF_MODEL='/path/to/your/llama/model'"
                )

        model_path = Path(model_name)

        print(f"Loading model from local weights: {model_path}")

        # Verify the model path exists and has required files
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory does not exist: {model_path}")

        config_file = model_path / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"config.json not found at: {config_file}")

        # Check for safetensors files
        safetensor_files = list(model_path.glob("*.safetensors"))
        index_file = model_path / "model.safetensors.index.json"

        if not safetensor_files and not index_file.exists():
            raise FileNotFoundError(f"No safetensors files found at: {model_path}")

        print(f"✅ Verified model files at: {model_path}")
        print(f"   - Config: {config_file}")
        if safetensor_files:
            print(f"   - Safetensors: {len(safetensor_files)} files")
        if index_file.exists():
            print(f"   - Index: {index_file}")

        # Load the model from local path
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",  # Load to CPU first to avoid GPU memory issues
            trust_remote_code=True,
            local_files_only=True,  # Only use local files, no downloads
        )

        print("Model loaded successfully from local weights")

        # Extract attention weights for the specific layer
        attention_weights = {}

        # Q, K, V, O projection weights
        weight_keys = {
            "q_proj": f"model.layers.{layer_idx}.self_attn.q_proj.weight",
            "k_proj": f"model.layers.{layer_idx}.self_attn.k_proj.weight",
            "v_proj": f"model.layers.{layer_idx}.self_attn.v_proj.weight",
            "o_proj": f"model.layers.{layer_idx}.self_attn.o_proj.weight",
        }

        for name, key in weight_keys.items():
            # Navigate the model structure to get the weight
            keys = key.split(".")
            current = model
            for k in keys:
                current = getattr(current, k)

            # Get the actual tensor data from the Parameter object
            if hasattr(current, "weight"):
                weight = current.weight.data
                print(f"Loaded {name} weight: {weight.shape}")
                attention_weights[name] = weight
            else:
                # Try alternative access patterns
                if hasattr(current, "data"):
                    weight = current.data
                    print(f"Loaded {name} weight (direct data): {weight.shape}")
                    attention_weights[name] = weight
                else:
                    raise AttributeError(f"Could not find weight for {name} at path {key}")

        # Get model info
        config = model.config
        print(f"Model configuration:")
        print(f"  - Hidden dimension: {config.hidden_size}")
        print(f"  - Number of heads: {config.num_attention_heads}")
        print(f"  - Number of KV heads: {config.num_key_value_heads}")
        print(f"  - Head dimension: {config.hidden_size // config.num_attention_heads}")
        print(f"  - Number of layers: {config.num_hidden_layers}")

        # Convert weights to TT-Metal tensors and push to device
        print("Converting weights to TT-Metal tensors...")
        tt_weights = {}
        print("Attn weight items : ", attention_weights.items())
        for name, weight in attention_weights.items():
            # Convert to bfloat16 and push to device
            weight_tt = ttnn.from_torch(
                weight.to(torch.bfloat16),
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            tt_weights[name] = weight_tt
            print(f"Pushed {name} to device: {weight_tt.shape}")

        # Clean up the loaded model to free memory
        del model
        import gc

        gc.collect()

        return {"weights": tt_weights, "config": config}

    except Exception as e:
        print(f"Error loading LLaMA attention weights: {e}")
        print("Falling back to random weights...")

        # Fallback to random weights if loading fails
        random_weights = QKVO_generate_and_push_random(device)
        return {
            "q_proj": random_weights[0],
            "k_proj": random_weights[1],
            "v_proj": random_weights[2],
            "o_proj": random_weights[3],
        }


# Llama load


# Llama RoPE, no boundary conditions/rescaling
def apply_custom_rope_to_qk(
    q_proj_reshaped,
    k_proj_reshaped,
    position_indices,
    device,
    head_dim=128,
    theta=500000.0,
    scale_factor=8,
    orig_context_len=8192,
):
    """
    Apply RoPE to Q and K tensors with arbitrary position indices (including duplicates)

    Args:
        q_proj_reshaped: [seq_len, batch, n_heads, head_dim] - Q projections
        k_proj_reshaped: [seq_len, batch, n_heads, head_dim] - K projections
        position_indices: [seq_len] - Actual position indices for each token (can have duplicates)
        device: TT-Metal device
        head_dim: Head dimension (default: 128 for LLaMA)
        theta: RoPE base frequency (default: 500000.0 for LLaMA)
        scale_factor: RoPE scaling factor (default: 8 for LLaMA 3.3)
        orig_context_len: Original context length before scaling (default: 8192 for LLaMA 3.3)
    Returns:
        q_rotated: Q with RoPE applied
        k_rotated: K with RoPE applied
    """

    # Step 1: Precompute cos/sin frequencies for all possible positions
    # We need to cover the maximum position index + some buffer
    max_pos = max(position_indices) if len(position_indices) > 0 else 0
    buffer_size = 100  # Add buffer for safety
    cos, sin = precompute_freqs(
        dim=head_dim,
        end=max_pos + buffer_size + 1,
        theta=theta,
        scale_factor=scale_factor,  # Use RoPE scaling for LLaMA 3.3
        orig_context_len=orig_context_len,
    )

    # Step 2: Gather cos/sin values for the specific positions
    # Convert position_indices to torch tensor if it's not already
    if not isinstance(position_indices, torch.Tensor):
        position_indices = torch.tensor(position_indices, dtype=torch.long)

    # Gather cos/sin for each position
    cos_gathered, sin_gathered = gather_cos_sin(position_indices, cos, sin)

    # Step 3: Get the transformation matrix (reuse existing function)
    trans_mat = get_rot_transformation_mat(head_dim)
    trans_mat_tt = ttnn.from_torch(
        trans_mat, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    # Step 4: Convert cos/sin to TT-Metal tensors
    cos_tt = ttnn.from_torch(
        cos_gathered, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    sin_tt = ttnn.from_torch(
        sin_gathered, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    # Step 5: Ensure Q and K are in bfloat16
    if q_proj_reshaped.dtype != ttnn.bfloat16:
        q_proj_reshaped = ttnn.typecast(q_proj_reshaped, dtype=ttnn.bfloat16)
    if k_proj_reshaped.dtype != ttnn.bfloat16:
        k_proj_reshaped = ttnn.typecast(k_proj_reshaped, dtype=ttnn.bfloat16)

    # Step 6: Apply RoPE using the existing TT-Metal function
    # This will work because we've prepared the cos/sin matrices correctly
    q_rotated = ttnn.experimental.rotary_embedding_llama(
        q_proj_reshaped, cos_tt, sin_tt, trans_mat_tt, is_decode_mode=False  # We're in prefill mode
    )

    k_rotated = ttnn.experimental.rotary_embedding_llama(
        k_proj_reshaped, cos_tt, sin_tt, trans_mat_tt, is_decode_mode=False  # We're in prefill mode
    )

    # Clean up intermediate tensors
    cos_tt.deallocate(True)
    sin_tt.deallocate(True)
    trans_mat_tt.deallocate(True)

    return q_rotated, k_rotated


# Llama RoPE, no boundary conditions/rescaling

if __name__ == "__main__":
    print("Opening device")
    device = ttnn.open_device(device_id=0)

    rope_setup = RotarySetup(
        device,
        MAX_BATCH_SIZE,
        HEAD_DIM,
        MAX_SEQ_LEN,
        hf_model_config["rope_theta"],
        None,
        hf_model_config["original_max_position_embeddings"],
    )
    trans_mats_dict = rope_setup.get_both_trans_mats()

    print("")
    print("Initializing K and V caches")
    [K_past, V_past, kv_len] = init_kv_cache(device=device)
    print("K and V initialized with length : ", kv_len)

    print("")
    print("Making rotation bases, for 0 to max seq len")
    start_pos = kv_len
    draft_seq_len = 28
    tt_rot_mats_draft = [
        rope_setup.cos_matrix[:, :, start_pos : start_pos + draft_seq_len, :],
        rope_setup.sin_matrix[:, :, start_pos : start_pos + draft_seq_len, :],
    ]
    print(type(tt_rot_mats_draft), type(tt_rot_mats_draft[0]))
    print("KV ready for draft sequence")

    """
    [q_tt, k_tt, v_tt, attn_mask_tt] = generate_qkv_sequence_and_mask(draft_seq_len, device=device)
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
    """
    print("")
    print("Checking LLaMA 3.3 70B availability...")

    # Check if the model is available before attempting to load
    # Check if the model specified in HF_MODEL environment variable is available
    if check_llama_availability():
        # Configure which layer to load (0-indexed, so layer 1 = index 1)
        target_layer_idx = 1  # Change this to load different layers
        print(f"Loading LLaMA attention weights for decoder block {target_layer_idx}...")

        # Load real attention weights and config
        result = load_llama_attention_weights(device, layer_idx=target_layer_idx)

        attention_weights = result["weights"]
        # Extract the weights
        Q = attention_weights["q_proj"]  # [head_dim * n_heads, hidden_dim]
        K = attention_weights["k_proj"]  # [head_dim * n_kv_heads, hidden_dim]
        V = attention_weights["v_proj"]  # [head_dim * n_kv_heads, hidden_dim]
        O = attention_weights["o_proj"]  # [hidden_dim, hidden_dim]

        print("✅ Loaded real LLaMA attention weights:")
        print(f"Q: {Q.shape}, K: {K.shape}, V: {V.shape}, O: {O.shape}")

        # Update global parameters from config if available
        if result["config"] is not None:
            config = result["config"]
            print("🔄 Updating model parameters from config file...")

            # Update global variables (these are module-level variables)
            MODEL_HIDDEN = config.hidden_size
            N_QHEADS = config.num_attention_heads
            N_KVHEADS = config.num_key_value_heads
            HEAD_DIM = config.hidden_size // config.num_attention_heads

            # Store the entire model config for future use
            hf_model_config = config

            print(f"✅ Updated model parameters:")
            print(f"   - Hidden dimension: {MODEL_HIDDEN}")
            print(f"   - Number of query heads: {N_QHEADS}")
            print(f"   - Number of KV heads: {N_KVHEADS}")
            print(f"   - Head dimension: {HEAD_DIM}")
            print(f"   - RoPE theta: {getattr(config, 'rope_theta', 'N/A')}")
            print(f"   - RoPE scaling factor: {getattr(config, 'rope_scaling_factor', 'N/A')}")
            print(f"   - Model type: {getattr(config, 'model_type', 'N/A')}")
            print(f"   - Total layers: {getattr(config, 'num_hidden_layers', 'N/A')}")

            # Now hf_model_config contains the full model config - access any parameter like:
            # hf_model_config.hidden_size, hf_model_config.num_attention_heads, etc.
        else:
            print("⚠️  No config available, using hardcoded parameters")
            print("   - Using fallback RoPE config from hf_model_config")
    else:
        print("⚠️  Using random weights as fallback...")

        # Fallback to random weights
        random_weights = QKVO_generate_and_push_random(device)
        Q, K, V, O = random_weights

        print("📝 Using random attention weights:")
        print(f"Q: {Q.shape}, K: {K.shape}, V: {V.shape}, O: {O.shape}")

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
    # NEW: Apply custom RoPE to Q and K after QKV projection
    print("Applying custom RoPE to Q and K...")

    # Define the actual position indices for your draft tokens
    # This is where you specify the real positions (can have duplicates)
    draft_position_indices = [current_sequence_length + int(i % 2) for i in range(draft_sequence_length)]
    print("Draft position indices : ", draft_position_indices)
    # Example: if current_sequence_length=100 and draft_sequence_length=5
    # draft_position_indices = [100, 101, 102, 103, 104]
    # But in Eagle, you might have: [100, 100, 101, 102, 100] (duplicates!)

    q_rotated, k_rotated = apply_custom_rope_to_qk(
        q_proj_reshaped,
        k_proj_reshaped,
        draft_position_indices,  # Pass actual position indices
        device=device,
        head_dim=HEAD_DIM,  # Use the model's head dimension
        theta=hf_model_config["rope_theta"],  # LLaMA 3.3 RoPE base frequency
        scale_factor=hf_model_config["rope_scaling_factor"],  # LLaMA 3.3 RoPE scaling
        orig_context_len=hf_model_config["original_max_position_embeddings"],  # Original context length
    )

    print("Custom RoPE applied successfully")
    print("Q rotated shape: ", q_rotated.shape)
    print("K rotated shape: ", k_rotated.shape)

    # Deallocate original Q and K (we now use rotated versions)
    q_proj_reshaped.deallocate(True)
    k_proj_reshaped.deallocate(True)
    # RoPE q and k

    # Add to cache, k and v
    print("K past shape before update : ", K_past.shape)
    # K_past = ttnn.concat([K_past, k_proj_reshaped], dim=0)
    K_past = ttnn.concat([K_past, k_rotated], dim=0)
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
    # q_proj_reshaped = ttnn.permute(q_proj_reshaped, (1, 2, 0, 3))
    q_proj_reshaped = ttnn.permute(q_rotated, (1, 2, 0, 3))
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
