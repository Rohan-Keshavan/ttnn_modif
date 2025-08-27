import os

import torch
from attention_utils import check_llama_availability

import ttnn
from models.tt_transformers.tt.common import gather_cos_sin, get_rot_transformation_mat, precompute_freqs
from models.tt_transformers.tt.rope import RotarySetup


class AttentionBlock:
    """
    GQA Llama Attention , Ins : Draft token tree, Manual KV management (Static, DRAM pushed)

    This class handles:
    - Model weight loading and GQA expansion
    - RoPE setup and management
    - KV cache initialization and management
    - Forward pass computation for K tokens
    - Memory management and cleanup
    """

    def __init__(self, device, model_path=None, layer_idx=0):
        """
        Initialize the AttentionBlock.

        Args:
            device: TT-Metal device
            model_path: Path to LLaMA model weights (optional)
            layer_idx: Decoder block index to load (default: 0)
        """
        print("")
        print("------------------------------Layer setup start--------------------------")
        print("")
        self.device = device
        self.model_path = model_path
        self.layer_idx = layer_idx

        # Model parameters (will be updated from config)
        self.MODEL_HIDDEN = 4096  # Default for LLaMA 8B
        self.N_QHEADS = 32  # Default for LLaMA 8B
        self.N_KVHEADS = 8  # Default for LLaMA 8B (GQA)
        self.HEAD_DIM = 128  # Default: 4096 // 32
        self.GQA_GROUP_SIZE = int(self.N_QHEADS / self.N_KVHEADS)
        self.KV_MAX_CHUNK_SIZE = 32

        # TT-Metal settings
        self.TILE_SIZE = 32
        self.BATCH_SIZE = 1
        self.MAX_BATCH_SIZE = 1
        self.MAX_SEQ_LEN = 1024
        self.ATTN_SCALE = 1 / (self.HEAD_DIM**0.5)

        # RoPE configuration
        self.hf_model_config = {
            "rope_theta": 500000.0,
            "original_max_position_embeddings": 8192,
            "rope_scaling_factor": 8,
            "max_position_embeddings": 128 * 1024,
        }

        # Model weights and components
        self.Q = None
        self.K = None
        self.V = None
        self.O = None

        # RoPE setup
        self.rope_setup = None
        self.trans_mats_dict = None

        # KV cache
        self.K_past = None
        self.V_past = None
        self.kv_len = 0
        self.max_kv_len = 0

        # Load weights if model path provided
        if model_path:
            self.load_weights(model_path, layer_idx)

        # Setup RoPE
        self.setup_rope()

        # Initialize KV cache
        self.init_kv_cache(max_capacity=2048)
        print("")
        print("------------------------------Layer setup end ---------------------------")
        print("")

    def load_weights(self, model_path, layer_idx):
        """
        Load LLaMA attention weights for a specific decoder block.

        Args:
            model_path: Path to LLaMA model weights
            layer_idx: Decoder block index
        """
        if check_llama_availability(model_name=model_path):
            try:
                import os
                from pathlib import Path

                from transformers import AutoModelForCausalLM

                if model_path is None:
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
                        # Try alternative access patterns
                        if hasattr(current, "data"):
                            weight = current.data
                            print(f"Loaded {name} weight (direct data): {weight.shape}")
                            attention_weights[name] = weight
                        else:
                            raise AttributeError(f"Could not find weight for {name} at path {key}")

                # Update model parameters from config
                config = model.config
                self.MODEL_HIDDEN = config.hidden_size
                self.N_QHEADS = config.num_attention_heads
                self.N_KVHEADS = config.num_key_value_heads
                self.HEAD_DIM = config.hidden_size // config.num_attention_heads
                self.GQA_GROUP_SIZE = int(self.N_QHEADS / self.N_KVHEADS)

                # Update RoPE config
                if hasattr(config, "rope_theta"):
                    self.hf_model_config["rope_theta"] = config.rope_theta
                if hasattr(config, "rope_scaling"):
                    if hasattr(config.rope_scaling, "factor"):
                        self.hf_model_config["rope_scaling_factor"] = config.rope_scaling.factor
                    if hasattr(config.rope_scaling, "original_max_position_embeddings"):
                        self.hf_model_config[
                            "original_max_position_embeddings"
                        ] = config.rope_scaling.original_max_position_embeddings

                print(f"Updated model parameters:")
                print(f"   - Hidden dimension       : {self.MODEL_HIDDEN}")
                print(f"   - Number of query heads  : {self.N_QHEADS}")
                print(f"   - Number of KV heads     : {self.N_KVHEADS}")
                print(f"   - Head dimension         : {self.HEAD_DIM}")
                print(f"   - GQA group size         : {self.GQA_GROUP_SIZE}")

                # Convert weights to TT-Metal tensors
                print("Converting weights to TT-Metal tensors...")
                self.Q = ttnn.from_torch(
                    attention_weights["q_proj"].to(torch.bfloat16),
                    device=self.device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

                self.K = ttnn.from_torch(
                    attention_weights["k_proj"].to(torch.bfloat16),
                    device=self.device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

                self.V = ttnn.from_torch(
                    attention_weights["v_proj"].to(torch.bfloat16),
                    device=self.device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

                self.O = ttnn.from_torch(
                    attention_weights["o_proj"].to(torch.bfloat16),
                    device=self.device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

                print("Weights loaded and converted to TT-Metal tensors")

                # Clean up
                del model
                import gc

                gc.collect()

            except Exception as e:
                print(f"Error loading LLaMA attention weights: {e}")
                print("Falling back to random weights...")
                self._generate_random_weights()

        else:
            print("Falling back to random weights...")
            self._generate_random_weights()

    def _generate_random_weights(self):
        """Generate random weights as fallback."""
        print("Generating random attention weights...")

        Q_random = torch.rand(1, 1, self.HEAD_DIM * self.N_QHEADS, self.MODEL_HIDDEN)
        K_random = torch.rand(1, 1, self.HEAD_DIM * self.N_KVHEADS, self.MODEL_HIDDEN)
        V_random = torch.rand(1, 1, self.HEAD_DIM * self.N_KVHEADS, self.MODEL_HIDDEN)
        O_random = torch.rand(1, 1, self.MODEL_HIDDEN, self.MODEL_HIDDEN)

        self.Q = ttnn.as_tensor(
            Q_random,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        self.K = ttnn.as_tensor(
            K_random,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        self.V = ttnn.as_tensor(
            V_random,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        self.O = ttnn.as_tensor(
            O_random,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

        print("📝 Random weights generated and pushed to device")

    def setup_rope(self):
        """Setup RoPE matrices."""
        print("Setting up RoPE...")

        self.rope_setup = RotarySetup(
            self.device,
            self.MAX_BATCH_SIZE,
            self.HEAD_DIM,
            self.MAX_SEQ_LEN,
            self.hf_model_config["rope_theta"],
            None,
            self.hf_model_config["original_max_position_embeddings"],
        )
        self.trans_mats_dict = self.rope_setup.get_both_trans_mats()
        print("✅ RoPE setup complete")

    def init_kv_cache(self, max_capacity=2048):
        """
        Initialize static KV Cache.
        Args:
            max_capacity: Maximum allowed cache capacity (default: 8192 tokens)
        """
        print(f"Initializing static KV cache: max={max_capacity}")
        self.max_capacity = max_capacity
        # KV cache dimensions: [batch, n_heads, seq_len, head_dim]
        # GQA: n_heads = N_QHEADS (after expansion)
        k_cache = torch.zeros(self.BATCH_SIZE, self.N_QHEADS, max_capacity, self.HEAD_DIM)
        v_cache = torch.zeros(self.BATCH_SIZE, self.N_QHEADS, max_capacity, self.HEAD_DIM)

        self.K_past = ttnn.as_tensor(
            k_cache,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

        self.V_past = ttnn.as_tensor(
            v_cache,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

        self.kv_len = 0
        self.max_capacity = max_capacity

        print(f"Static KV cache initialized: {self.K_past.shape}, {self.V_past.shape}")
        print(f"   - Max capacity: {self.max_capacity}")

    def forward(self, input_tokens, position_indices, attention_mask=None):
        """
        Forward pass for K tokens.

        Args:
            input_tokens: Input token embeddings [batch, seq_len, hidden_dim]
            position_indices: Position indices for each token [seq_len]
            attention_mask: Optional attention mask [batch, seq_len, seq_len]

        Returns:
            output: Attention output [batch, seq_len, hidden_dim]
        """
        print("")
        print("----------------------Model forward start--------------------------")
        print(f"Forward pass: {input_tokens.shape}, positions: {position_indices}")

        # Step 1: GQA Weight Expansion
        K_expanded, V_expanded = self._expand_gqa_weights()

        # Step 2: QKV Projection
        q_proj, k_proj, v_proj = self._project_qkv(input_tokens, K_expanded, V_expanded)

        # Step 3: Reshape to separate heads
        q_reshaped, k_reshaped, v_reshaped = self._reshape_to_heads(q_proj, k_proj, v_proj)

        # Step 4: Apply RoPE
        q_rotated, k_rotated = self._apply_rope(q_reshaped, k_reshaped, position_indices)

        # Step 5: Update KV cache (after computing attention)
        self._update_kv_cache(k_rotated, v_reshaped)
        print(f"KV cache updated: {self.K_past.shape}, {self.V_past.shape}")
        print(f"Current cache length: {self.kv_len}")

        # Step 6: Compute attention (this will handle cache vs. no-cache cases)
        attn_out = self._compute_attention_with_cache(q_rotated, k_rotated, v_reshaped, attention_mask)
        print(f"Attention output: {attn_out.shape}")

        # Step 7: Output projection
        output = self._output_projection(attn_out)

        # Cleanup intermediate tensors
        print("Cleaning up intermediate tensors...")
        self._cleanup_intermediate_tensors([q_proj, k_proj, v_proj, q_reshaped, k_reshaped, v_reshaped])

        print("Forward pass complete.")
        print("----------------------Model forward end ---------------------------")
        print("")

        return output

    def _expand_gqa_weights(self):
        """Expand K and V weights for GQA."""
        print("GQA Weight Expansion: Expanding K and V weights...")
        print("K shape : ", self.K.shape)
        # Reshape to separate heads
        K_reshaped = ttnn.reshape(self.K, (1, 1, self.N_KVHEADS, self.HEAD_DIM, self.MODEL_HIDDEN))
        V_reshaped = ttnn.reshape(self.V, (1, 1, self.N_KVHEADS, self.HEAD_DIM, self.MODEL_HIDDEN))

        # Repeat each KV head GQA_GROUP_SIZE times
        K_expanded = ttnn.repeat(K_reshaped, [1, 1, self.GQA_GROUP_SIZE, 1, 1])
        V_expanded = ttnn.repeat(V_reshaped, [1, 1, self.GQA_GROUP_SIZE, 1, 1])

        # Reshape back to flat format
        K_expanded_flat = ttnn.reshape(K_expanded, (1, 1, self.N_QHEADS * self.HEAD_DIM, self.MODEL_HIDDEN))
        V_expanded_flat = ttnn.reshape(V_expanded, (1, 1, self.N_QHEADS * self.HEAD_DIM, self.MODEL_HIDDEN))

        print(f"K expanded: {self.K.shape} -> {K_expanded_flat.shape}")
        print(f"V expanded: {self.V.shape} -> {V_expanded_flat.shape}")

        # Cleanup intermediate tensors
        K_reshaped.deallocate(True)
        K_expanded.deallocate(True)
        V_reshaped.deallocate(True)
        V_expanded.deallocate(True)

        return K_expanded_flat, V_expanded_flat

    def _project_qkv(self, input_tokens, K_expanded, V_expanded):
        """Project input tokens to Q, K, V."""
        print("QKV Projection...")

        q_proj = ttnn.linear(input_tokens, self.Q)
        k_proj = ttnn.linear(input_tokens, K_expanded)
        v_proj = ttnn.linear(input_tokens, V_expanded)

        print(f"QKV shapes: {q_proj.shape}, {k_proj.shape}, {v_proj.shape}")

        # Cleanup expanded weights
        K_expanded.deallocate(True)
        V_expanded.deallocate(True)

        return q_proj, k_proj, v_proj

    def _reshape_to_heads(self, q_proj, k_proj, v_proj):
        """Reshape QKV projections to separate heads."""
        print("Reshaping to heads...")

        q_reshaped = ttnn.reshape(q_proj, (self.BATCH_SIZE, -1, self.N_QHEADS, self.HEAD_DIM))
        q_reshaped = ttnn.transpose(q_reshaped, 0, 1)

        k_reshaped = ttnn.reshape(k_proj, (self.BATCH_SIZE, -1, self.N_QHEADS, self.HEAD_DIM))
        k_reshaped = ttnn.transpose(k_reshaped, 0, 1)

        v_reshaped = ttnn.reshape(v_proj, (self.BATCH_SIZE, -1, self.N_QHEADS, self.HEAD_DIM))
        v_reshaped = ttnn.transpose(v_reshaped, 0, 1)

        print(f"Reshaped: {q_reshaped.shape}, {k_reshaped.shape}, {v_reshaped.shape}")

        return q_reshaped, k_reshaped, v_reshaped

    def _apply_rope(self, q_reshaped, k_reshaped, position_indices):
        """Apply RoPE to Q and K."""
        print("Applying RoPE...")

        # Use the custom RoPE function
        q_rotated, k_rotated = apply_custom_rope_to_qk(
            q_reshaped,
            k_reshaped,
            position_indices,
            self.device,
            head_dim=self.HEAD_DIM,
            theta=self.hf_model_config["rope_theta"],
            scale_factor=self.hf_model_config["rope_scaling_factor"],
            orig_context_len=self.hf_model_config["original_max_position_embeddings"],
        )

        print(f"RoPE applied: {q_rotated.shape}, {k_rotated.shape}")
        return q_rotated, k_rotated

    def _grow_cache_if_needed(self, required_capacity):
        """
        Grow the KV cache if needed to accommodate the required capacity.

        Args:
            required_capacity: Required cache capacity
        """
        if required_capacity <= self.cache_capacity:
            return  # No growth needed

        # Calculate new capacity
        new_capacity = max(int(self.cache_capacity * self.growth_factor), required_capacity)

        # Cap at maximum capacity
        if new_capacity > self.max_capacity:
            if required_capacity > self.max_capacity:
                raise ValueError(f"Required capacity {required_capacity} exceeds maximum capacity {self.max_capacity}")
            new_capacity = self.max_capacity

        print(f"🔄 Growing KV cache: {self.cache_capacity} -> {new_capacity}")

        # Create new larger cache
        new_k_cache = torch.zeros(new_capacity, self.BATCH_SIZE, self.N_QHEADS, self.HEAD_DIM)
        new_v_cache = torch.zeros(new_capacity, self.BATCH_SIZE, self.N_QHEADS, self.HEAD_DIM)

        # Copy existing data to new cache
        if self.kv_len > 0:
            new_k_cache[: self.kv_len] = self.K_past.to_torch()
            new_v_cache[: self.kv_len] = self.V_past.to_torch()

        # Deallocate old cache
        self.K_past.deallocate(True)
        self.V_past.deallocate(True)

        # Create new TT-Metal tensors
        self.K_past = ttnn.as_tensor(
            new_k_cache,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

        self.V_past = ttnn.as_tensor(
            new_v_cache,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

        # Update capacity
        old_capacity = self.cache_capacity
        self.cache_capacity = new_capacity

        print(f"✅ Cache grown successfully: {old_capacity} -> {self.cache_capacity}")
        print(f"   - New cache shapes: {self.K_past.shape}, {self.V_past.shape}")

    def _update_kv_cache(self, k_new, v_new):
        print("Updating KV cache...")

        # Check if we need to grow the cache
        # new_total_length = self.kv_len + k_new.shape[0]
        # self._grow_cache_if_needed(new_total_length)

        # Simple concatenation - the cache is already the right size!
        if self.kv_len == 0:
            ttnn.fill_cache(self.K_past, ttnn.permute(k_new, (1, 2, 0, 3)), batch_idx=0)
            ttnn.fill_cache(self.V_past, ttnn.permute(v_new, (1, 2, 0, 3)), batch_idx=0)
        else:
            ttnn.update_cache(self.K_past, ttnn.permute(k_new, (1, 2, 0, 3)), update_idx=self.kv_len, batch_offset=0)
            ttnn.update_cache(self.V_past, ttnn.permute(v_new, (1, 2, 0, 3)), update_idx=self.kv_len, batch_offset=0)

        # If  cache overfill
        # For later
        # If  cache overfill

        # Update cache length
        self.kv_len += k_new.shape[0]

    def manual_kv_add(self, k_ext, v_ext):
        # Torch inputs

        k_ext = k_ext.repeat(1, self.GQA_GROUP_SIZE, 1, 1)
        v_ext = v_ext.repeat(1, self.GQA_GROUP_SIZE, 1, 1)
        print("Manual KV update....")
        print("k_ext , v_ext : ", k_ext.shape, v_ext.shape)
        print("Chunking and pushing into pre-allocated cache.")

        # Chunk this
        n_tokens_in_ext = k_ext.shape[2]
        n_chunks = int(n_tokens_in_ext / self.KV_MAX_CHUNK_SIZE)
        if n_tokens_in_ext / self.KV_MAX_CHUNK_SIZE > n_chunks:
            n_chunks += 1

        for j in range(n_chunks):
            print("Populating KV from ext : chunk size ", self.KV_MAX_CHUNK_SIZE)
            start = int(j * self.KV_MAX_CHUNK_SIZE)
            end = int((j + 1) * self.KV_MAX_CHUNK_SIZE)
            if end > n_tokens_in_ext:
                end = n_tokens_in_ext
            k_chunk = k_ext[:, :, start:end, :]
            v_chunk = v_ext[:, :, start:end, :]

            k_chunk = ttnn.from_torch(
                k_chunk,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            v_chunk = ttnn.from_torch(
                v_chunk,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            print("Trying to push : ", k_chunk.shape, v_chunk.shape)
            if self.kv_len == 0:
                ttnn.fill_cache(self.K_past, k_chunk, batch_idx=0)
                ttnn.fill_cache(self.V_past, v_chunk, batch_idx=0)
            else:
                ttnn.update_cache(self.K_past, k_chunk, update_idx=self.kv_len, batch_offset=0)
                ttnn.update_cache(self.V_past, v_chunk, update_idx=self.kv_len, batch_offset=0)
            self.kv_len += k_chunk.shape[2]
            print("KV updated... current kv length : ", self.kv_len)

        print("Manual KV add complete...")

    def _compute_attention_with_cache(self, q_rotated, k_current, v_current, attention_mask=None):
        """
        Compute attention using KV cache if available, otherwise use current tokens.

        Args:
            q_rotated: Current Q tokens [seq_len, batch, heads, head_dim]
            k_current: Current K tokens [seq_len, batch, heads, head_dim]
            v_current: Current V tokens [seq_len, batch, heads, head_dim]
            attention_mask: Optional attention mask
        """
        print("Computing attention with cache handling...")
        """
        if self.kv_len == 0:
            # First forward pass: use only current tokens
            print("First forward pass: using current tokens only")
            return self._compute_attention_current_tokens_only(q_rotated, k_current, v_current, attention_mask)
        else:
            # Subsequent passes: use cache + current tokens
            print(f"Using KV cache + current tokens: cache={self.kv_len}, current={q_rotated.shape[0]}")
            return self._compute_attention_with_cache_and_current(q_rotated, k_current, v_current, attention_mask)
        """
        print(f"Using KV cache + current tokens: cache={self.kv_len}, current={q_rotated.shape[0]}")
        return self._compute_attention_with_cache_and_current(q_rotated, k_current, v_current, attention_mask)

    def _compute_attention_current_tokens_only(self, q_rotated, k_current, v_current, attention_mask=None):
        """Compute attention using only current tokens (no cache)."""
        print("Computing attention with current tokens only...")

        # For the first forward pass, compute self-attention among current tokens
        seq_len = q_rotated.shape[0]

        # Prepare tensors for self-attention
        q_permuted = ttnn.permute(q_rotated, (1, 2, 0, 3))  # [batch, heads, seq_len, head_dim]
        k_permuted = ttnn.permute(k_current, (1, 2, 3, 0))  # [batch, heads, head_dim, seq_len]
        v_permuted = ttnn.permute(v_current, (1, 2, 0, 3))  # [batch, heads, seq_len, head_dim]

        # Compute attention scores: Q @ K^T
        QKT = ttnn.matmul(q_permuted, k_permuted)  # [batch, heads, seq_len, seq_len]
        QKT = QKT * self.ATTN_SCALE

        print("QKT shape : ", QKT.shape)
        print("Attn mask shape : ", attention_mask.shape)
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = torch.clamp(torch.log(attention_mask), min=-1e06)
            attention_mask = attention_mask.repeat(1, self.N_QHEADS, 1, 1)
            attention_mask = ttnn.as_tensor(
                attention_mask,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            QKT += attention_mask

        # Softmax and attention output
        QKT = ttnn.softmax(QKT, dim=-1)
        attn_out = ttnn.matmul(QKT, v_permuted)  # [batch, heads, seq_len, head_dim]

        # Reshape back to [seq_len, batch, heads, head_dim]
        attn_out = ttnn.permute(attn_out, (2, 0, 1, 3))

        print(f"Self-attention output: {attn_out.shape}")
        return attn_out

    def _compute_attention_with_cache_and_current(self, q_rotated, k_current, v_current, attention_mask=None):
        """Compute attention using KV cache + current tokens."""
        print("Computing attention with cache + current tokens...")
        seq_len = q_rotated.shape[0]

        # Slice the cache to only use filled portion
        print("Attn in shapes : (K,V,q) ", self.K_past.shape, self.V_past.shape, q_rotated.shape)
        K_filled = ttnn.slice(self.K_past, (0, 0, 0, 0), (self.BATCH_SIZE, self.N_QHEADS, self.kv_len, self.HEAD_DIM))
        V_filled = ttnn.slice(self.V_past, (0, 0, 0, 0), (self.BATCH_SIZE, self.N_QHEADS, self.kv_len, self.HEAD_DIM))

        print(f"Using filled KV cache: {K_filled.shape}, {V_filled.shape} (filled: {self.kv_len}/{self.max_kv_len})")

        # Prepare tensors for attention
        q_permuted = ttnn.permute(q_rotated, (1, 2, 0, 3))  # [batch, heads, seq_len, head_dim]
        K_permuted = ttnn.permute(K_filled, (0, 1, 3, 2))  # [batch, heads, head_dim, total_seq_len]
        V_permuted = ttnn.permute(V_filled, (0, 1, 2, 3))  # [batch, heads, total_seq_len, head_dim]

        # Compute attention scores: Q @ K^T
        QKT = ttnn.matmul(q_permuted, K_permuted)  # [batch, heads, seq_len, total_seq_len]
        QKT = QKT * self.ATTN_SCALE
        print("QKT shape : ", QKT.shape)
        # Apply attention mask if provided
        if attention_mask is not None:
            print("Attn mask shape : ", attention_mask.shape)
            print("Sequence length, kv length : ", seq_len, self.kv_len)
            # Encode batch size better.
            if seq_len == self.kv_len:
                _ = 0
            else:
                # Assert batch size
                M_prefix = torch.ones(self.BATCH_SIZE, 1, seq_len, self.kv_len - seq_len)
                attention_mask = torch.cat([M_prefix, attention_mask], dim=-1)
            attention_mask = torch.clamp(torch.log(attention_mask), min=-1e09)
            attention_mask = attention_mask.repeat(1, self.N_QHEADS, 1, 1)
            print("Attn mask shape : ", attention_mask.shape)
            attention_mask = ttnn.as_tensor(
                attention_mask,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            QKT += attention_mask

        # Softmax and attention output
        QKT = ttnn.softmax(QKT, dim=-1)
        attn_out = ttnn.matmul(QKT, V_permuted)  # [batch, heads, seq_len, head_dim]

        # Reshape back to [seq_len, batch, heads, head_dim]
        attn_out = ttnn.permute(attn_out, (2, 0, 1, 3))

        print(f"Cache + current attention output: {attn_out.shape}")

        # Clean up intermediate tensors
        K_filled.deallocate(True)
        V_filled.deallocate(True)
        K_permuted.deallocate(True)
        V_permuted.deallocate(True)

        return attn_out

    def _output_projection(self, attn_out):
        """Apply output projection."""
        print("Output projection...")

        # Reshape and apply output projection
        attn_out = ttnn.to_layout(attn_out, layout=ttnn.ROW_MAJOR_LAYOUT)
        attn_out = ttnn.reshape(attn_out, (self.BATCH_SIZE, -1, self.MODEL_HIDDEN))
        attn_out = ttnn.to_layout(attn_out, layout=ttnn.TILE_LAYOUT)

        output = ttnn.linear(attn_out, self.O)

        print(f"Final output: {output.shape}")
        return output

    def _cleanup_intermediate_tensors(self, tensors):
        """Clean up intermediate tensors."""
        for tensor in tensors:
            if tensor is not None:
                tensor.deallocate(True)

    # KV Cache Management Methods
    def get_kv_cache_state(self):
        """Get current KV cache state."""
        return {
            "current_length": self.kv_len,
            "max_capacity": self.max_capacity,
            "utilization": f"{self.kv_len}/{self.max_capacity} ({100 * self.kv_len / self.max_capacity:.1f}%)",
            "K_shape": self.K_past.shape if self.K_past else None,
            "V_shape": self.V_past.shape if self.V_past else None,
            "batch_size": self.BATCH_SIZE,
            "n_heads": self.N_QHEADS,
            "head_dim": self.HEAD_DIM,
        }

    def reset_kv_cache(self):
        """Reset KV cache to initial state."""
        print("Resetting KV cache...")

        if self.K_past is not None:
            self.K_past.deallocate(True)
        if self.V_past is not None:
            self.V_past.deallocate(True)

        # Reset to initial capacity
        self.init_kv_cache(max_capacity=self.max_capacity)
        print("✅ KV cache reset to initial capacity")

    def resize_kv_cache(self, new_capacity):
        """
        Resize KV cache to a specific capacity.

        Args:
            new_capacity: New cache capacity
        """
        if new_capacity < self.kv_len:
            raise ValueError(f"New capacity {new_capacity} cannot be less than current length {self.kv_len}")

        if new_capacity > self.max_capacity:
            raise ValueError(f"New capacity {new_capacity} cannot exceed max capacity {self.max_capacity}")

        print(f"🔄 Resizing KV cache: {self.cache_capacity} -> {new_capacity}")

        # Create new cache with target capacity
        new_k_cache = torch.zeros(new_capacity, self.BATCH_SIZE, self.N_QHEADS, self.HEAD_DIM)
        new_v_cache = torch.zeros(new_capacity, self.BATCH_SIZE, self.N_QHEADS, self.HEAD_DIM)

        # Copy existing data (up to the smaller of current length or new capacity)
        copy_length = min(self.kv_len, new_capacity)
        if copy_length > 0:
            new_k_cache[:copy_length] = self.K_past.to_torch()
            new_v_cache[:copy_length] = self.V_past.to_torch()

        # Deallocate old cache
        self.K_past.deallocate(True)
        self.V_past.deallocate(True)

        # Create new TT-Metal tensors
        self.K_past = ttnn.as_tensor(
            new_k_cache,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

        self.V_past = ttnn.as_tensor(
            new_v_cache,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

        # Update capacity and length
        old_capacity = self.cache_capacity
        self.cache_capacity = new_capacity
        self.kv_len = copy_length

        print(f"✅ Cache resized successfully: {old_capacity} -> {self.cache_capacity}")
        print(f"   - New cache shapes: {self.K_past.shape}, {self.V_past.shape}")
        print(f"   - Current length: {self.kv_len}")

    def clear_kv_cache(self):
        """Clear KV cache completely."""
        print("Clearing KV cache...")

        if self.K_past is not None:
            self.K_past.deallocate(True)
            self.K_past = None
        if self.V_past is not None:
            self.V_past.deallocate(True)
            self.V_past = None

        self.kv_len = 0
        print("KV cache cleared.")

    def get_cache_utilization(self):
        """Get cache utilization statistics."""
        if self.cache_capacity == 0:
            return 0.0

        utilization = self.kv_len / self.max_capacity
        return {
            "current_length": self.kv_len,
            "utilization_percent": utilization * 100,
            "free_space": self.max_capacity - self.kv_len,
        }

    def optimize_cache_size(self, target_utilization=0.8):
        """
        Optimize cache size based on target utilization.

        Args:
            target_utilization: Target utilization ratio (default: 0.8 = 80%)
        """
        if self.kv_len == 0:
            return  # Nothing to optimize

        target_capacity = int(self.kv_len / target_utilization)

        # Ensure we don't go below initial capacity
        target_capacity = max(target_capacity, 64)

        # Ensure we don't exceed max capacity
        target_capacity = min(target_capacity, self.max_capacity)

        if target_capacity != self.cache_capacity:
            print(f"🔄 Optimizing cache size for {target_utilization*100:.0f}% utilization")
            self.resize_kv_cache(target_capacity)
        else:
            print(f"✅ Cache size already optimal for {target_utilization*100:.0f}% utilization")

    def cleanup(self):
        """Clean up all resources."""
        print("")
        print("Cleaning up AttentionBlock resources...")

        # Clean up weights
        for weight_name in ["Q", "K", "V", "O"]:
            weight = getattr(self, weight_name)
            if weight is not None:
                weight.deallocate(True)
                setattr(self, weight_name, None)

        # Clean up KV cache
        self.clear_kv_cache()

        # Clean up RoPE setup
        if self.rope_setup is not None:
            # Note: RotarySetup doesn't have a cleanup method, so we just set to None
            self.rope_setup = None
            self.trans_mats_dict = None

        print("AttentionBlock cleanup complete")


# Keep the existing utility functions for now (will move to separate file)
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
    """
    # Step 1: Precompute cos/sin frequencies for all possible positions
    max_pos = max(position_indices) if len(position_indices) > 0 else 0
    buffer_size = 100  # Add buffer for safety
    cos, sin = precompute_freqs(
        dim=head_dim,
        end=max_pos + buffer_size + 1,
        theta=theta,
        scale_factor=scale_factor,
        orig_context_len=orig_context_len,
    )

    # Step 2: Gather cos/sin values for the specific positions
    if not isinstance(position_indices, torch.Tensor):
        position_indices = torch.tensor(position_indices, dtype=torch.long)

    cos_gathered, sin_gathered = gather_cos_sin(position_indices, cos, sin)

    # Step 3: Get the transformation matrix
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
    q_rotated = ttnn.experimental.rotary_embedding_llama(
        q_proj_reshaped, cos_tt, sin_tt, trans_mat_tt, is_decode_mode=False
    )

    k_rotated = ttnn.experimental.rotary_embedding_llama(
        k_proj_reshaped, cos_tt, sin_tt, trans_mat_tt, is_decode_mode=False
    )

    # Clean up intermediate tensors
    cos_tt.deallocate(True)
    sin_tt.deallocate(True)
    trans_mat_tt.deallocate(True)

    return q_rotated, k_rotated


# Calculating RMSNorm
def rms_norm(x, norm_weights):
    # Calculate the mean of the square of tensor values along the last dimension
    squared_mean = x.pow(2).mean(-1, keepdim=True)

    # Add a small value to avoid division by zero
    normalized = torch.rsqrt(squared_mean + torch.finfo(x.dtype).eps)

    # Multiply normalized tensor by the provided normalization weights
    return (x * normalized) * norm_weights


if __name__ == "__main__":
    interest = "decoder_0.pt"
    root = os.getcwd()
    ref_data_path = os.path.join(root, "reference_data")
    ref_data_file = os.path.join(ref_data_path, interest)

    ref_data = torch.load(ref_data_file)
    # example         = ref_data[len(ref_data)-1]

    example = torch.load(os.path.join(ref_data_path, "example_with_intermediates.pt"))

    # Example usage of the AttentionBlock class
    print("Opening device")
    device = ttnn.open_device(device_id=0)

    # print('Loading RMS Norm weights for layer 0...')
    # llama_ckpt_path = "/data/models/meta-llama/Llama-3.1-8B-Instruct/model-00001-of-00004.safetensors"
    # rms_norm_ckpt   = load_file(llama_ckpt_path)["model.layers.0.input_layernorm.weight"]
    # print('Layernorm checkpoint : ' , rms_norm_ckpt.shape)

    try:
        # Initialize attention block
        attention_block = AttentionBlock(
            device, model_path="/data/models/meta-llama/Llama-3.1-8B-Instruct", layer_idx=0
        )
        """
        # Generate test input
        batch_size          = 1
        seq_len             = 60
        E , M               = generate_random_embeddings_and_mask_torch(batch_size,seq_len,attention_block.MODEL_HIDDEN,device)
        position_indices    = [int(i%2) for i in range(seq_len)]
        E                   = ttnn.as_tensor(
                            E, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                            )
        print('Simulated E and M : ' , E.shape, M.shape)
        """
        # Ref Inputs

        hidden_states = example["args/hidden_states_post_norm"]
        # hidden_states_post_norm = rms_norm(hidden_states,rms_norm_ckpt)
        # print('Hidden states post norm : ' , hidden_states_post_norm.shape)

        attention_mask = example["args/attention_mask"]
        position_ids = example["args/position_ids"]
        position_ids = torch.squeeze(position_ids).tolist()
        past_key_value = example["args/past_key_value"]
        past_k = past_key_value[0][:, :, 0 : attention_mask.shape[3] - attention_mask.shape[2], :]
        past_v = past_key_value[1][:, :, 0 : attention_mask.shape[3] - attention_mask.shape[2], :]
        attention_mask = attention_mask[:, :, :, attention_mask.shape[3] - attention_mask.shape[2] :]
        # Ref Inputs
        hidden_states = ttnn.as_tensor(
            hidden_states,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        # Keep past_k and past_v in torch
        print("Ref inputs E and M   : ", hidden_states.shape, attention_mask.shape)
        print("Past kv              : ", past_k.shape, past_v.shape)
        attention_block.manual_kv_add(past_k, past_v)
        # Generate test input

        # Forward pass
        output = attention_block.forward(
            input_tokens=hidden_states, position_indices=position_ids, attention_mask=attention_mask
        )
        print(f"First forward pass output: {output.shape}")
        # Forward pass

        # Get reference output
        torch_output = example["outputs/hidden_states"]
        tt_output = ttnn.to_torch(ttnn.from_device(output))
        tt_output = tt_output
        squared_errors = (torch_output - tt_output) ** 2
        mean_error = torch.mean(squared_errors)
        max_error = torch.max(squared_errors)
        print("Errors (mean , max) : ", mean_error, max_error)
        print(
            "torch allclose (atol 1e-04, rtol 1e-05) : ",
            torch.allclose(torch_output, tt_output, atol=1e-04, rtol=1e-05),
        )
        # Get reference output

        # Check KV cache state
        kv_state = attention_block.get_kv_cache_state()
        print(f"KV cache state: {kv_state}")
        # Check KV cache state

        """
        # Another forward pass (KV cache maintained)
        batch_size = 1
        seq_len    = 12
        E , M = generate_random_embeddings_and_mask_torch(batch_size,seq_len,attention_block.MODEL_HIDDEN,device)
        position_indices = [int(i%3) for i in range(seq_len)]
        E = ttnn.as_tensor(
            E, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

        output2 = attention_block.forward(input_tokens = E, position_indices = position_indices, attention_mask = M)
        print(f"Second forward pass output: {output2.shape}")

        # Check updated KV cache state
        kv_state2 = attention_block.get_kv_cache_state()
        print(f"Updated KV cache state: {kv_state2}")
        """
        # Cleanup
        attention_block.cleanup()

    finally:
        print("Closing device")
        ttnn.close_device(device)
