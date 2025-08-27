"""
Utility functions for AttentionBlock class.

This module contains helper functions for:
- Model availability checking
- Weight loading and management
- Input generation and preprocessing
- Memory management utilities
"""

import gc
import os
from pathlib import Path

import torch

import ttnn


def check_llama_availability(model_name=None):
    """
    Check if LLaMA model weights exist locally using HF_MODEL environment variable.

    Args:
        model_name: Optional model name override (defaults to HF_MODEL env var)

    Returns:
        bool: True if model weights exist locally, False otherwise
    """
    print("")

    # Get model path from HF_MODEL environment variable
    if model_name is None:
        model_name = os.getenv("HF_MODEL")
        if not model_name:
            print("HF_MODEL environment variable not set")
            print("Set: export HF_MODEL='/path/to/your/llama/model'")
            return False

    model_path = Path(model_name)
    print(f"Checking model path for config.json and .safetensors : {model_path}")

    # Check if the directory exists
    if not model_path.exists():
        print(f"Model directory does not exist: {model_path}")
        return False

    # Check for config.json
    config_file = model_path / "config.json"
    if not config_file.exists():
        print(f"config.json not found at: {config_file}")
        return False

    # Check for safetensors files
    safetensor_files = list(model_path.glob("*.safetensors"))
    if not safetensor_files:
        # Also check for model.safetensors.index.json
        index_file = model_path / "model.safetensors.index.json"
        if not index_file.exists():
            print(f"No safetensors files found at: {model_path}")
            print("Expected: *.safetensors files or model.safetensors.index.json")
            return False
        else:
            print(f" Found safetensors index file: {index_file}")

    print("")
    print("Search summary")
    print(f"Found local model weights at: {model_path}")
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
        print(f"   - Model type     : {config.get('model_type', 'N/A')}")
        print(f"   - Hidden size    : {config.get('hidden_size', 'N/A')}")
        print(f"   - Num layers     : {config.get('num_hidden_layers', 'N/A')}")
        print(f"   - Num heads      : {config.get('num_attention_heads', 'N/A')}")
        print(f"   - Num KV heads   : {config.get('num_key_value_heads', 'N/A')}")
        return True
    except Exception as e:
        print(f"   - Config read error: {e}")
        # Still return True if the files exist
        return True


def load_llama_attention_weights(device, model_path, layer_idx=0):
    """
    Load LLaMA attention weights for a specific decoder block.

    Args:
        device: TT-Metal device
        model_path: Path to LLaMA model weights
        layer_idx: Decoder block index (0 for first block)

    Returns:
        dict: Dictionary containing Q, K, V, O weights and config
    """
    try:
        from transformers import AutoModelForCausalLM

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
                raise AttributeError(f"Could not find weight for {name}")

        # Get model config
        config = model.config
        print(f"Model configuration:")
        print(f"  - Hidden dimension: {config.hidden_size}")
        print(f"  - Number of heads: {config.num_attention_heads}")
        print(f"  - Number of KV heads: {config.num_key_value_heads}")
        print(f"  - Head dimension: {config.hidden_size // config.num_attention_heads}")
        print(f"  - Number of layers: {config.num_hidden_layers}")

        # Clean up
        del model
        gc.collect()

        return {"weights": attention_weights, "config": config}

    except Exception as e:
        print(f"Error loading LLaMA attention weights: {e}")
        raise


def generate_random_weights(device, model_hidden, n_qheads, n_kvheads, head_dim):
    """
    Generate random attention weights for testing.

    Args:
        device: TT-Metal device
        model_hidden: Hidden dimension
        n_qheads: Number of query heads
        n_kvheads: Number of KV heads
        head_dim: Head dimension

    Returns:
        dict: Dictionary containing Q, K, V, O weights
    """
    print("Generating random attention weights...")

    Q_random = torch.rand(1, 1, head_dim * n_qheads, model_hidden)
    K_random = torch.rand(1, 1, head_dim * n_kvheads, model_hidden)
    V_random = torch.rand(1, 1, head_dim * n_kvheads, model_hidden)
    O_random = torch.rand(1, 1, model_hidden, model_hidden)

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

    print("📝 Random weights generated and pushed to device")

    return {"q_proj": Q_tt, "k_proj": K_tt, "v_proj": V_tt, "o_proj": O_tt}


def generate_random_embeddings_and_mask(batch_size, in_seq_len, model_hidden, device):
    attn_input = torch.rand(batch_size, in_seq_len, model_hidden)
    attn_mask = torch.rand(batch_size, 1, in_seq_len, in_seq_len)
    attn_inputs_tt = ttnn.as_tensor(
        attn_input, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    return [attn_inputs_tt, attn_mask]


def generate_random_embeddings_and_mask_torch(batch_size, in_seq_len, model_hidden, device):
    attn_input = torch.rand(batch_size, in_seq_len, model_hidden)
    attn_mask = torch.rand(batch_size, 1, in_seq_len, in_seq_len)
    return [attn_input, attn_mask]


def generate_random_embeddings(seq_len, model_hidden, batch_size=1, device=None):
    """
    Generate random token embeddings for testing.

    Args:
        seq_len: Sequence length
        model_hidden: Hidden dimension
        batch_size: Batch size (default: 1)
        device: TT-Metal device (optional, for direct TT-Metal tensor creation)

    Returns:
        torch.Tensor or ttnn.Tensor: Random embeddings
    """
    embeddings = torch.randn(batch_size, seq_len, model_hidden)

    if device is not None:
        # Convert to TT-Metal tensor
        embeddings_tt = ttnn.as_tensor(
            embeddings,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        return embeddings_tt

    return embeddings


def generate_attention_mask(seq_len, batch_size=1, n_heads=1, causal=True, device=None):
    """
    Generate attention mask for testing.

    Args:
        seq_len: Sequence length
        batch_size: Batch size (default: 1)
        n_heads: Number of attention heads (default: 1)
        causal: Whether to generate causal mask (default: True)
        device: TT-Metal device (optional, for direct TT-Metal tensor creation)

    Returns:
        torch.Tensor or ttnn.Tensor: Attention mask
    """
    if causal:
        # Create causal mask (lower triangular)
        mask = torch.tril(torch.ones(seq_len, seq_len))
    else:
        # Create full attention mask
        mask = torch.ones(seq_len, seq_len)

    # Add batch and head dimensions
    mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    mask = mask.repeat(batch_size, n_heads, 1, 1)  # [batch_size, n_heads, seq_len, seq_len]

    # Convert to log space and clamp
    mask = torch.clamp(torch.log(mask), min=-1e06)

    if device is not None:
        # Convert to TT-Metal tensor
        mask_tt = ttnn.as_tensor(
            mask, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        return mask_tt

    return mask


def generate_rectangular_attention_mask(current_seq_len, new_seq_len, batch_size=1, n_heads=1, device=None):
    """
    Generate rectangular attention mask for Eagle-style speculative decoding.

    Args:
        current_seq_len: Current sequence length in KV cache
        new_seq_len: New sequence length being added
        batch_size: Batch size (default: 1)
        n_heads: Number of attention heads (default: 1)
        device: TT-Metal device (optional, for direct TT-Metal tensor creation)

    Returns:
        torch.Tensor or ttnn.Tensor: Rectangular attention mask
    """
    # Create prefix mask: new tokens can attend to all previous tokens
    prefix_mask = torch.ones(batch_size, n_heads, new_seq_len, current_seq_len)

    # Create main mask: new tokens can attend to each other
    main_mask = torch.ones(batch_size, n_heads, new_seq_len, new_seq_len)

    # Concatenate along the last dimension
    full_mask = torch.cat([prefix_mask, main_mask], dim=-1)

    # Convert to log space and clamp
    full_mask = torch.clamp(torch.log(full_mask), min=-1e06)

    if device is not None:
        # Convert to TT-Metal tensor
        mask_tt = ttnn.as_tensor(
            full_mask,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        return mask_tt

    return full_mask


def create_position_indices(start_pos, seq_len, allow_duplicates=False, duplicate_pattern=None):
    """
    Create position indices for tokens.

    Args:
        start_pos: Starting position
        seq_len: Sequence length
        allow_duplicates: Whether to allow duplicate positions (for Eagle)
        duplicate_pattern: Pattern for duplicates (e.g., [0, 0, 1, 1, 2] for 5 tokens)

    Returns:
        list: Position indices
    """
    if allow_duplicates and duplicate_pattern is not None:
        # Use custom duplicate pattern
        if len(duplicate_pattern) != seq_len:
            raise ValueError(f"Duplicate pattern length {len(duplicate_pattern)} must match seq_len {seq_len}")
        return [start_pos + pos for pos in duplicate_pattern]

    elif allow_duplicates:
        # Create some duplicates for testing (every other position)
        positions = []
        for i in range(seq_len):
            if i % 2 == 0:
                positions.append(start_pos + i // 2)
            else:
                positions.append(start_pos + i // 2)
        return positions

    else:
        # Sequential positions (no duplicates)
        return list(range(start_pos, start_pos + seq_len))


def validate_model_config(config):
    """
    Validate model configuration for GQA compatibility.

    Args:
        config: Model configuration object

    Returns:
        bool: True if valid, raises error if not
    """
    # Check required attributes
    required_attrs = ["hidden_size", "num_attention_heads", "num_key_value_heads"]
    for attr in required_attrs:
        if not hasattr(config, attr):
            raise ValueError(f"Config missing required attribute: {attr}")

    # Check GQA compatibility
    n_qheads = config.num_attention_heads
    n_kvheads = config.num_key_value_heads

    if n_qheads % n_kvheads != 0:
        raise ValueError(
            f"GQA requires num_attention_heads ({n_qheads}) to be divisible by " f"num_key_value_heads ({n_kvheads})"
        )

    # Check head dimension
    head_dim = config.hidden_size // n_qheads
    if head_dim * n_qheads != config.hidden_size:
        raise ValueError(
            f"Hidden size ({config.hidden_size}) must be divisible by " f"num_attention_heads ({n_qheads})"
        )

    print(f"✅ Model config validation passed:")
    print(f"   - Hidden size: {config.hidden_size}")
    print(f"   - Query heads: {n_qheads}")
    print(f"   - KV heads: {n_kvheads}")
    print(f"   - Head dimension: {head_dim}")
    print(f"   - GQA group size: {n_qheads // n_kvheads}")

    return True


def cleanup_tensors(tensors):
    """
    Clean up a list of TT-Metal tensors.

    Args:
        tensors: List of tensors to deallocate
    """
    for tensor in tensors:
        if tensor is not None:
            try:
                tensor.deallocate(True)
            except Exception as e:
                print(f"Warning: Could not deallocate tensor: {e}")


def print_tensor_info(tensor, name="Tensor"):
    """
    Print information about a TT-Metal tensor.

    Args:
        tensor: TT-Metal tensor
        name: Name for identification
    """
    if tensor is None:
        print(f"{name}: None")
        return

    try:
        print(f"{name}:")
        print(f"  - Shape: {tensor.shape}")
        print(f"  - Dtype: {tensor.dtype}")
        print(f"  - Layout: {tensor.layout}")
        print(f"  - Memory config: {tensor.memory_config}")
    except Exception as e:
        print(f"{name}: Error getting info - {e}")


def benchmark_forward_pass(attention_block, input_tokens, position_indices, num_runs=5):
    """
    Benchmark the forward pass of an attention block.

    Args:
        attention_block: AttentionBlock instance
        input_tokens: Input token embeddings
        position_indices: Position indices
        num_runs: Number of runs for averaging

    Returns:
        dict: Benchmark results
    """
    import time

    print(f"Benchmarking forward pass with {num_runs} runs...")

    # Warmup run
    _ = attention_block.forward(input_tokens, position_indices)

    # Benchmark runs
    times = []
    for i in range(num_runs):
        start_time = time.time()
        output = attention_block.forward(input_tokens, position_indices)
        end_time = time.time()

        run_time = (end_time - start_time) * 1000  # Convert to milliseconds
        times.append(run_time)

        print(f"  Run {i+1}: {run_time:.2f} ms")

    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"Benchmark Results:")
    print(f"  - Average time: {avg_time:.2f} ms")
    print(f"  - Min time: {min_time:.2f} ms")
    print(f"  - Max time: {max_time:.2f} ms")

    return {
        "average_time_ms": avg_time,
        "min_time_ms": min_time,
        "max_time_ms": max_time,
        "num_runs": num_runs,
        "times_ms": times,
    }
