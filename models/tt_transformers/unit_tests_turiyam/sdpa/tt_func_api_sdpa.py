import os
from typing import Optional

import torch
from attention_utils import check_llama_availability
from torch import nn
from transformers import LlamaConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

import ttnn
from models.tt_transformers.tt.common import (
    gather_cos_sin,
    get_prefill_rot_mat,
    get_rot_transformation_mat,
    precompute_freqs,
)
from models.tt_transformers.tt.rope import RotarySetup


# CPU Rope
def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """
    Apply rotary position embeddings to query and key tensors.

    Args:
        q (torch.Tensor): Query tensor.
        k (torch.Tensor): Key tensor.
        cos (torch.Tensor): Cosine values.
        sin (torch.Tensor): Sine values.
        position_ids (torch.Tensor): Position IDs.

    Returns:
        torch.Tensor: Query and key tensors with rotary position embeddings applied.
    """
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_L31(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaRotaryEmbedding_L31(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[LlamaConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            print("CONFIG NOT NONE")
            # BC: "rope_type" was originally "type"
            print("Check : ", config.rope_scaling)
            if config.rope_scaling is not None:
                print("ROPE TYPE NOT DEFAULT")
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# CPU Rope
def compute_attention_with_cache_and_current(K_past, V_past, q_rotated, k_current, v_current, attention_mask=None):
    """Compute attention using KV cache + current tokens."""

    kv_len = K_past.shape[2]
    compute_kernel_config_hifi4 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    ATTN_SCALE = 1 / (128**0.5)

    # attention_mask is                                                                                                                                                                                                                                                                                                                                                                    square
    seq_len = q_rotated.shape[2]

    # Get the relevant KV slice, Expand K and V
    K_filled = ttnn.slice(K_past, (0, 0, 0, 0), (1, 8, kv_len, 128))
    V_filled = ttnn.slice(V_past, (0, 0, 0, 0), (1, 8, kv_len, 128))
    K_filled = ttnn.repeat(K_filled, [1, 4, 1, 1])
    V_filled = ttnn.repeat(V_filled, [1, 4, 1, 1])
    # Get the relevant KV slice, Expand K and V

    # Compute attention scores: Q @ K^T
    qkt_past = ttnn.linear(
        q_rotated, K_filled, transpose_b=True, compute_kernel_config=compute_kernel_config_hifi4
    )  # was q_permuted  # [batch, heads, seq_len, total_seq_len]
    qkt_past = qkt_past * ATTN_SCALE  # No mask needed

    # repeat and permute k
    k_current = ttnn.repeat(k_current, [1, 4, 1, 1])
    qkt_current = ttnn.linear(q_rotated, k_current, transpose_b=True, compute_kernel_config=compute_kernel_config_hifi4)
    print("qkt_past and qkt shapes  : ", qkt_past.shape, qkt_current.shape)

    # Apply attention mask if provided
    if attention_mask is not None:
        # print('TT attention mask : ' , attention_mask)

        print("Attn mask shape | dtype      : ", attention_mask.shape, attention_mask.dtype)
        qkt_current = ttnn.typecast(qkt_current, dtype=ttnn.float32)
        qkt_current += attention_mask
        qkt_current = qkt_current * ATTN_SCALE
        qkt_current = ttnn.typecast(qkt_current, dtype=ttnn.bfloat16)

    # Concat prefix and current qkts
    qkt = ttnn.concat([qkt_past, qkt_current], dim=-1)
    print("QKT shape : ", qkt.shape)

    # Softmax and attention output
    qkt = ttnn.typecast(qkt, dtype=ttnn.float32)
    qkt = ttnn.softmax(qkt, dim=-1)
    qkt = ttnn.typecast(qkt, dtype=ttnn.bfloat16)
    print("Overall qkt and v shape : ", qkt.shape, v_current.shape, V_filled.shape)
    # Softmax and attention output
    return qkt


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
        print("Setting up the attention layer...")

        self.device = device
        self.model_path = model_path
        self.layer_idx = layer_idx

        # Model parameters (will be updated from config)
        self.MODEL_HIDDEN = 4096  # Default for LLaMA 8B
        self.N_QHEADS = 32  # Default for LLaMA 8B
        self.N_KVHEADS = 8  # Default for LLaMA 8B (GQA)
        self.HEAD_DIM = 128  # Default: 4096 // 32
        self.GQA_GROUP_SIZE = int(self.N_QHEADS / self.N_KVHEADS)
        # self.KV_MAX_CHUNK_SIZE = 32

        # TT-Metal settings
        self.TILE_SIZE = 32
        self.BATCH_SIZE = 1
        self.MAX_BATCH_SIZE = 1
        self.MAX_SEQ_LEN = 1024
        self.ATTN_SCALE = 1 / (self.HEAD_DIM**0.5)
        # self.ATTN_SCALE = ttnn.from_torch(self.ATTN_SCALE,dtype=ttnn.float32,device=self.device,memory_config=ttnn.DRAM_MEMORY_CONFIG,layout=ttnn.TILE_LAYOUT)

        self.hf_config_file = None

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
        # KV cache

        # Load weights if model path provided
        self.rope_cpu = False
        if model_path:
            self.load_weights(model_path, layer_idx)

        # Setup RoPE
        if self.hf_config_file is not None:
            self.setup_rope_cpu()

        self.compute_kernel_config_hifi4 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )

        # Initialize KV cache
        self.init_kv_cache(max_capacity=2048)
        print("Setup complete...")
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

                # Check if the directory exists
                if not model_path.exists():
                    print(f"Model directory does not exist: {model_path}")
                # Check for config.json
                config_file = model_path / "config.json"
                if not config_file.exists():
                    print(f"config.json not found at: {config_file}")
                self.hf_config_file = config_file
                # Update RoPE config
                """
                if hasattr(config, "rope_theta"):
                    self.hf_model_config["rope_theta"] = config.rope_theta
                if hasattr(config, "rope_scaling"):
                    if hasattr(config.rope_scaling, "factor"):
                        self.hf_model_config["rope_scaling_factor"] = config.rope_scaling.factor
                    if hasattr(config.rope_scaling, "original_max_position_embeddings"):
                        self.hf_model_config[
                            "original_max_position_embeddings"
                        ] = config.rope_scaling.original_max_position_embeddings
                """

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

                print("Weights before transposing (qkvo) : ", self.Q.shape, self.K.shape, self.V.shape, self.O.shape)
                self.Q = ttnn.permute(self.Q, (1, 0))
                self.K = ttnn.permute(self.K, (1, 0))
                self.V = ttnn.permute(self.V, (1, 0))
                self.O = ttnn.permute(self.O, (1, 0))
                print(
                    "Weights loaded and converted to TT-Metal tensors (qkvo) : ",
                    self.Q.shape,
                    self.K.shape,
                    self.V.shape,
                    self.O.shape,
                )

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
        print("RoPE theta          : ", self.hf_model_config["rope_theta"])
        print("RoPE mpe            : ", self.hf_model_config["original_max_position_embeddings"])
        self.rope_setup = RotarySetup(
            self.device,
            self.MAX_BATCH_SIZE,
            self.HEAD_DIM,
            self.MAX_SEQ_LEN,
            self.hf_model_config["rope_theta"],
            8,
            self.hf_model_config["max_position_embeddings"],
        )
        self.trans_mats_dict = self.rope_setup.get_both_trans_mats()
        print("RoPE init complete")

    def setup_rope_cpu(self):
        self.rope_cpu = True
        self.rotary_emb = LlamaRotaryEmbedding_L31(config=LlamaConfig.from_json_file(self.hf_config_file))

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

        intermediates = {}
        print("")
        print(f"Forward pass: {input_tokens.shape}")
        intermediates["inputs"] = input_tokens

        # Step 1: QKV Projection
        q_proj, k_proj, v_proj = self._project_qkv(input_tokens, reshape_and_expand=True)
        intermediates["q_proj"] = q_proj
        intermediates["k_proj"] = k_proj
        intermediates["v_proj"] = v_proj

        # Step 2: Rope Prepare + Apply
        if self.rope_cpu:
            q_rotated, k_rotated = self._apply_rope_cpu(q_proj, k_proj, position_indices)
            intermediates["k_post_rope"] = k_rotated
            intermediates["q_post_rope"] = q_rotated
        else:
            rot_mats = self._prepare_step_rope(position_ids)
            q_rotated, k_rotated = self._apply_rope(q_proj, k_proj, rot_mats=rot_mats)
            intermediates["k_post_rope"] = k_rotated
            intermediates["q_post_rope"] = q_rotated

        # Step 3 : Head broadcast, K and V : Can keep cache smaller, optimize for later
        # k_rotated                           = ttnn.repeat(k_rotated , [1, self.GQA_GROUP_SIZE, 1, 1])
        # v_proj                              = ttnn.repeat(v_proj    , [1, self.GQA_GROUP_SIZE, 1, 1])
        # Head broadcast,         K and V : Can keep cache smaller, optimize for later

        # Step 4: Update KV cache : Skip the update ? Makes more sense actually. Update post verification.
        # self._update_kv_cache(k_rotated, v_proj)
        # print(f"KV updated. Current cache length: {self.kv_len}")
        intermediates["K_cache_pre_attention"] = self.K_past
        intermediates["V_cache_pre_attention"] = self.V_past
        intermediates["kv_len_pre_attention"] = self.kv_len
        # Step 4: Update KV cache

        # Step 5: Attention
        attn_out, attention_intermediates = self._compute_attention_with_cache(
            q_rotated, k_rotated, v_proj, attention_mask
        )
        print(f"Attention output: {attn_out.shape}")
        intermediates["attention_intermediates"] = attention_intermediates
        intermediates["attention_output"] = attn_out
        # Step 5: Attention

        # Step 6: Output projection
        output = self._output_projection(attn_out)
        # Step 6: Output projection

        # Cleanup intermediate tensors
        print("Cleaning up intermediates...")
        # self._cleanup_intermediate_tensors([q_reshaped, k_reshaped, v_reshaped])#[q_proj, k_proj, v_proj]

        print("Forward pass complete.")
        print("")

        return output, intermediates

    def _forward_attention():
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

        K_expanded_flat = ttnn.permute(K_expanded_flat, (0, 1, 3, 2))
        K_expanded_flat = ttnn.permute(K_expanded_flat, (0, 1, 3, 2))

        print(f"K expanded: {self.K.shape} -> {K_expanded_flat.shape}")
        print(f"V expanded: {self.V.shape} -> {V_expanded_flat.shape}")

        # Cleanup intermediate tensors
        K_reshaped.deallocate(True)
        K_expanded.deallocate(True)
        V_reshaped.deallocate(True)
        V_expanded.deallocate(True)

        return K_expanded_flat, V_expanded_flat

    #    def _project_qkv(self, input_tokens, K_expanded, V_expanded):
    def _project_qkv(self, input_tokens, reshape_and_expand=False):
        """Project input tokens to Q, K, V."""
        print("QKV Projection...")

        q_proj = ttnn.linear(
            input_tokens,
            self.Q,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
        )
        #        k_proj = ttnn.linear(input_tokens, K_expanded,dtype = ttnn.bfloat16,memory_config=ttnn.DRAM_MEMORY_CONFIG,compute_kernel_config=self.compute_kernel_config_hifi4)
        k_proj = ttnn.linear(
            input_tokens,
            self.K,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
        )
        #        v_proj = ttnn.linear(input_tokens, V_expanded,dtype = ttnn.bfloat16,memory_config=ttnn.DRAM_MEMORY_CONFIG,compute_kernel_config=self.compute_kernel_config_hifi4)
        v_proj = ttnn.linear(
            input_tokens,
            self.V,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
        )

        print(f" qkv out: {q_proj.shape}, {k_proj.shape}, {v_proj.shape}")
        if reshape_and_expand:
            q_proj = ttnn.reshape(q_proj, (q_proj.shape[0], q_proj.shape[1], self.N_QHEADS, self.HEAD_DIM))
            q_proj = ttnn.permute(q_proj, (0, 2, 1, 3))

            k_proj = ttnn.reshape(k_proj, (k_proj.shape[0], k_proj.shape[1], self.N_KVHEADS, self.HEAD_DIM))
            k_proj = ttnn.permute(k_proj, (0, 2, 1, 3))
            # k_proj = ttnn.repeat(k_proj, [1, 1, self.GQA_GROUP_SIZE, 1])

            v_proj = ttnn.reshape(v_proj, (v_proj.shape[0], v_proj.shape[1], self.N_KVHEADS, self.HEAD_DIM))
            v_proj = ttnn.permute(v_proj, (0, 2, 1, 3))
            # v_proj = ttnn.repeat(v_proj, [1, 1, self.GQA_GROUP_SIZE, 1])
        print("qkv rope in: ", q_proj.shape, k_proj.shape, v_proj.shape)
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

    def _apply_rope(self, q_reshaped, k_reshaped, rot_mats=None):
        """Apply RoPE to Q and K."""
        print("Applying RoPE...")

        """
        q_heads_1QSD = ttnn.experimental.rotary_embedding_llama(
            q_reshaped,
            rot_mats[0],
            rot_mats[1],
            self.transformation_mats["prefill"],
            is_decode_mode=False,
        )

        k_heads_1KSD = ttnn.experimental.rotary_embedding_llama(
            k_heads_1KSD_pre_rot,
            rot_mats[0],
            rot_mats[1],
            self.transformation_mats["prefill"],
            is_decode_mode=False,
        )
        print('Shape post Rope (Q,K) : ' , q_heads_1QSD.shape, k_heads_1KSD.shape)

        """

        trans_mat = get_rot_transformation_mat(self.HEAD_DIM)
        trans_mat_tt = ttnn.from_torch(
            trans_mat,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q_rotated = ttnn.experimental.rotary_embedding_llama(
            q_reshaped, cos_cache=rot_mats[0], sin_cache=rot_mats[1], trans_mat=trans_mat_tt, is_decode_mode=False
        )
        k_rotated = ttnn.experimental.rotary_embedding_llama(
            k_reshaped, cos_cache=rot_mats[0], sin_cache=rot_mats[1], trans_mat=trans_mat_tt, is_decode_mode=False
        )

        print(f"QK rotated: {q_rotated.shape}, {k_rotated.shape}")
        return q_rotated, k_rotated

    def _apply_rope_cpu(self, q, k, pos_ids):
        q = ttnn.to_torch(ttnn.from_device(q)).to(torch.bfloat16)
        k = ttnn.to_torch(ttnn.from_device(k)).to(torch.bfloat16)
        cos, sin = self.rotary_emb(q, torch.tensor(pos_ids).unsqueeze(0).to(torch.long))
        q_rotated, k_rotated = apply_rotary_pos_emb_L31(q, k, cos, sin)
        q_rotated = ttnn.from_torch(
            q_rotated,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        k_rotated = ttnn.from_torch(
            k_rotated,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return q_rotated, k_rotated

    def _prepare_step_rope(self, position_indices):
        max_position = max(position_indices)
        min_position = min(position_indices)

        tt_cos_matrix = self.rope_setup.cos_matrix[:, :, self.kv_len : self.kv_len + max_position + 1, :]
        tt_sin_matrix = self.rope_setup.sin_matrix[:, :, self.kv_len : self.kv_len + max_position + 1, :]
        torch_cos_matrix = ttnn.to_torch(ttnn.from_device(tt_cos_matrix))
        torch_sin_matrix = ttnn.to_torch(ttnn.from_device(tt_sin_matrix))
        idx = torch.tensor(position_ids, dtype=torch.long)
        torch_cosines_full = torch_cos_matrix[:, :, idx - min_position, :]
        torch_sines_full = torch_sin_matrix[:, :, idx - min_position, :]
        tt_cosines_full = ttnn.from_torch(
            torch_cosines_full, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        tt_sines_full = ttnn.from_torch(torch_sines_full, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        # Need the full thing
        return [tt_cosines_full, tt_sines_full]

    def manual_kv_add(self, k_ext, v_ext):  # Torch inputs
        # k_ext = k_ext.repeat(1, self.GQA_GROUP_SIZE, 1, 1)
        # v_ext = v_ext.repeat(1, self.GQA_GROUP_SIZE, 1, 1)
        print("Manual KV update....")
        print("k_ext , v_ext : ", k_ext.shape, v_ext.shape)
        print("Chunking and pushing into pre-allocated cache.")
        k = ttnn.from_torch(
            k_ext, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        v = ttnn.from_torch(
            v_ext, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        ttnn.fill_cache(self.K_past, k, batch_idx=0)  # might have a max seq len limitation. Unit test this. Or go paged
        ttnn.fill_cache(
            self.V_past, v, batch_idx=0
        )  # might have a max seq len limitation. Unit test this. Or go paged.
        self.kv_len = k.shape[2]

        # Chunk this
        """
        n_tokens_in_ext = k_ext.shape[2]
        n_chunks        = int(n_tokens_in_ext / self.KV_MAX_CHUNK_SIZE)
        if (n_tokens_in_ext / self.KV_MAX_CHUNK_SIZE) > n_chunks:
            n_chunks    += 1

        print("Populating KV from ext : chunk size ", self.KV_MAX_CHUNK_SIZE)
        for j in range(n_chunks):
            start = int(j * self.KV_MAX_CHUNK_SIZE)
            end   = int((j + 1) * self.KV_MAX_CHUNK_SIZE)
            if end > n_tokens_in_ext:
                end = n_tokens_in_ext
            k_chunk = k_ext[:, :, start:end, :]
            v_chunk = v_ext[:, :, start:end, :]

            k_chunk = ttnn.from_torch(k_chunk,device=device,memory_config=ttnn.DRAM_MEMORY_CONFIG,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT)
            v_chunk = ttnn.from_torch(v_chunk,device=device,memory_config=ttnn.DRAM_MEMORY_CONFIG,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT)
            print("Trying to push : ", k_chunk.shape, v_chunk.shape)
            if self.kv_len == 0:
                ttnn.fill_cache(self.K_past, k_chunk, batch_idx=0)
                ttnn.fill_cache(self.V_past, v_chunk, batch_idx=0)
            else:
                ttnn.update_cache(self.K_past, k_chunk, update_idx=self.kv_len, batch_offset=0)
                ttnn.update_cache(self.V_past, v_chunk, update_idx=self.kv_len, batch_offset=0)
            self.kv_len += k_chunk.shape[2]
            print("KV updated... current kv length : ", self.kv_len)
        """
        # chunk this

        print("Manual KV add complete. Verifying adds..")

        """
        K_filled = ttnn.slice(self.K_past, (0, 0, 0, 0), (self.BATCH_SIZE, self.N_KVHEADS, self.kv_len, self.HEAD_DIM))
        V_filled = ttnn.slice(self.V_past, (0, 0, 0, 0), (self.BATCH_SIZE, self.N_KVHEADS, self.kv_len, self.HEAD_DIM))
        print("K and V filled shapes : ", K_filled.shape, V_filled.shape)
        K_filled = ttnn.to_torch(ttnn.from_device(K_filled)).to(torch.bfloat16)
        V_filled = ttnn.to_torch(ttnn.from_device(V_filled)).to(torch.bfloat16)
        errors_k = torch.abs((K_filled - k_ext))
        mean_error_k = torch.mean(errors_k)
        max_error_k = torch.max(errors_k)
        print("K errors        : ", mean_error_k, max_error_k)
        print("Pcc (K) : ", pcc(K_filled, k_ext))
        print(
            "All close check : ",
            torch.allclose(K_filled, k_ext, atol=1e-04, rtol=1e-02),
        )
        errors_v = torch.abs((V_filled - v_ext))
        mean_error_v = torch.mean(errors_v)
        max_error_v = torch.max(errors_v)
        print("V errors        : ", mean_error_v, max_error_v)
        print("Pcc (V) : ", pcc(V_filled, v_ext))
        print(
            "All close check : ",
            torch.allclose(V_filled, v_ext, atol=1e-04, rtol=1e-02),
        )
        print("Verification complete..")
        """
        # K_filled.deallocate(True)
        # V_filled.deallocate(True)

    def _compute_attention_with_cache(self, q_rotated, k_current, v_current, attention_mask=None):
        """
        Compute attention using KV cache if available, otherwise use current tokens.

        Args:
            q_rotated: Current Q tokens [seq_len, batch, heads, head_dim]
            k_current: Current K tokens [seq_len, batch, heads, head_dim]
            v_current: Current V tokens [seq_len, batch, heads, head_dim]
            attention_mask: Optional attention mask
        """
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

    def _compute_attention_block(self, q, k, v, mask=None):
        # q, k , v are ttnn tensors on device
        qkt = ttnn.linear(
            q, k, transpose_b=True, dtype=ttnn.bfloat16, compute_kernel_config=self.compute_kernel_config_hifi4
        )
        if mask is not None:
            qkt = ttnn.typecast(qkt, dtype=ttnn.float32)
            qkt += attention_mask
            qkt = ttnn.typecast(qkt, dtype=ttnn.bfloat16)

        qkt = qkt * self.ATTN_SCALE

        qkt = ttnn.typecast(qkt, dtype=ttnn.float32)
        qkt = ttnn.softmax(qkt, dim=-1)
        log_sum_exponent = ttnn.sum(qkt, dim=-1)
        qkt = ttnn.typecast(qkt, dtype=ttnn.bfloat16)

        attn = ttnn.linear(qkt, v)
        attn = ttnn.permute(attn, (0, 2, 1, 3))
        return qkt, log_sum_exponent

    def _compute_attention_with_cache_and_current(self, q_rotated, k_current, v_current, attention_mask=None):
        """Compute attention using KV cache + current tokens."""

        print("Computing attention with cache + current tokens...")
        attention_intermediates = {}

        # attention_mask is                                                                                                                                                                                                                                                                                                                                                                    square
        seq_len = q_rotated.shape[2]

        # Get the relevant KV slice, Expand K and V
        K_filled = ttnn.slice(self.K_past, (0, 0, 0, 0), (self.BATCH_SIZE, self.N_KVHEADS, self.kv_len, self.HEAD_DIM))
        V_filled = ttnn.slice(self.V_past, (0, 0, 0, 0), (self.BATCH_SIZE, self.N_KVHEADS, self.kv_len, self.HEAD_DIM))
        K_filled = ttnn.repeat(K_filled, [1, self.GQA_GROUP_SIZE, 1, 1])
        V_filled = ttnn.repeat(V_filled, [1, self.GQA_GROUP_SIZE, 1, 1])
        print(f"Using filled KV cache: {K_filled.shape}, {V_filled.shape} (filled: {self.kv_len}/{self.max_capacity})")
        # Get the relevant KV slice, Expand K and V

        # Compute attention scores: Q @ K^T
        qkt_past = ttnn.linear(
            q_rotated, K_filled, transpose_b=True, compute_kernel_config=self.compute_kernel_config_hifi4
        )  # was q_permuted  # [batch, heads, seq_len, total_seq_len]
        qkt_past = qkt_past * self.ATTN_SCALE  # No mask needed
        attention_intermediates["qkt_past"] = qkt_past

        # repeat and permute k
        k_current = ttnn.repeat(k_current, [1, self.GQA_GROUP_SIZE, 1, 1])
        qkt_current = ttnn.linear(
            q_rotated, k_current, transpose_b=True, compute_kernel_config=self.compute_kernel_config_hifi4
        )
        print("qkt_past and qkt shapes  : ", qkt_past.shape, qkt_current.shape)

        # Apply attention mask if provided
        if attention_mask is not None:
            # print('TT attention mask : ' , attention_mask)

            print("Attn mask shape | dtype      : ", attention_mask.shape, attention_mask.dtype)
            print("Sequence length, kv length   : ", seq_len, self.kv_len)
            qkt_current = ttnn.typecast(qkt_current, dtype=ttnn.float32)
            # qkt_current                     += attention_mask
            qkt_current = qkt_current * self.ATTN_SCALE
            qkt_current = ttnn.typecast(qkt_current, dtype=ttnn.bfloat16)
            attention_intermediates["qkt_current"] = qkt_current

        # Concat prefix and current qkts
        qkt = ttnn.concat([qkt_past, qkt_current], dim=-1)
        attention_intermediates["qkt"] = qkt
        print("QKT shape : ", qkt.shape)

        # Softmax and attention output
        qkt = ttnn.typecast(qkt, dtype=ttnn.float32)
        qkt = ttnn.softmax(qkt, dim=-1)
        qkt = ttnn.typecast(qkt, dtype=ttnn.bfloat16)
        print("Overall qkt and v shape : ", qkt.shape, v_current.shape, V_filled.shape)
        # Softmax and attention output

        v_current = ttnn.repeat(v_current, [1, self.GQA_GROUP_SIZE, 1, 1])
        v_complete = ttnn.concat([V_filled, v_current], dim=2)

        attn_out = ttnn.linear(qkt, v_complete)  # [batch, heads, seq_len, head_dim] #v mul can also be split
        print("Attn out shape : ", attn_out.shape)

        attn_out = ttnn.permute(attn_out, (0, 2, 1, 3))  # Reshape back to [seq_len, batch, heads, head_dim]
        attn_out = ttnn.to_layout(attn_out, layout=ttnn.ROW_MAJOR_LAYOUT)
        attn_out = ttnn.reshape(attn_out, (self.BATCH_SIZE, -1, self.MODEL_HIDDEN))
        attn_out = ttnn.to_layout(attn_out, layout=ttnn.TILE_LAYOUT)

        # Clean up intermediate tensors
        K_filled.deallocate(True)
        V_filled.deallocate(True)
        qkt_past.deallocate(True)
        qkt_current.deallocate(True)
        qkt.deallocate(True)
        # Clean up intermediate tensors

        return attn_out, attention_intermediates

    def _output_projection(self, attn_out):
        """Apply output projection."""
        print("Output projection...")
        output = ttnn.linear(
            attn_out,
            self.O,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
        )
        print(f"Final output: {output.shape}")
        return output

    # KV Cache management methods
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
        k_cache = torch.zeros((self.BATCH_SIZE, self.N_KVHEADS, max_capacity, self.HEAD_DIM), dtype=torch.bfloat16)
        v_cache = torch.zeros((self.BATCH_SIZE, self.N_KVHEADS, max_capacity, self.HEAD_DIM), dtype=torch.bfloat16)

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

    def _update_kv_cache(self, k_new, v_new):
        print("Updating KV cache...")
        print("KV shapes : ", k_new.shape, v_new.shape)
        n_updates = k_new.shape[2]
        if self.kv_len == 0:
            ttnn.fill_cache(self.K_past, k_new, batch_idx=0)
            ttnn.fill_cache(self.V_past, v_new, batch_idx=0)
        else:
            for j in range(n_updates):
                ttnn.update_cache(self.K_past, k_new[:, :, j, :], update_idx=self.kv_len + j, batch_offset=0)
                ttnn.update_cache(self.V_past, v_new[:, :, j, :], update_idx=self.kv_len + j, batch_offset=0)

        # Update cache length
        self.kv_len += k_new.shape[2]

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
        print("KV cache reset.")

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

    # KV Cache management methods

    # cleanup
    def _cleanup_intermediate_tensors(self, tensors):
        """Clean up intermediate tensors."""
        for tensor in tensors:
            if tensor is not None:
                tensor.deallocate(True)

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

    # cleanup


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
    cos, sin = precompute_freqs(
        dim=head_dim, end=max_pos + 1, theta=theta, scale_factor=scale_factor, orig_context_len=orig_context_len
    )
    print("Rope prep cos and sine shapes : ", cos.shape, sin.shape)

    pos_ids = position_indices
    # Step 2: Gather cos/sin values for the specific positions
    if not isinstance(position_indices, torch.Tensor):
        position_indices = torch.tensor(position_indices, dtype=torch.long)
    cos_gathered, sin_gathered = gather_cos_sin(position_indices, cos, sin)

    # Step 3: Get the transformation matrix
    trans_mat = get_rot_transformation_mat(head_dim)
    trans_mat_tt = ttnn.from_torch(
        trans_mat, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    trans_mat_tt = get_prefill_rot_mat(
        head_dim=head_dim,
        mesh_device=None,
        seq_len=len(pos_ids),
        theta=theta,
        scale_factor=scale_factor,
        orig_context_len=orig_context_len,
        start_pos=min(pos_ids),
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


def pcc(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.reshape(-1).float()
    y = y.reshape(-1).float()

    vx = x - x.mean()
    vy = y - y.mean()

    corr = torch.sum(vx * vy) / torch.sqrt(torch.sum(vx**2) * torch.sum(vy**2))
    return corr


def prepare_attention_in(example, device):
    K_past = example["outputs/intermediates"]["K_past"]
    V_past = example["outputs/intermediates"]["V_past"]
    q_rotated = example["outputs/intermediates"]["q_post_rope"]
    k_rotated = example["outputs/intermediates"]["k_post_rope"]
    v_states = example["outputs/intermediates"]["v_pre_rope"]
    attn_mask = example["args/attention_mask"]
    K_past = ttnn.from_torch(
        K_past, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    V_past = ttnn.from_torch(
        V_past, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    q_rotated = ttnn.from_torch(
        q_rotated, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    k_rotated = ttnn.from_torch(
        k_rotated, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    v_states = ttnn.from_torch(
        v_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    attn_mask = attn_mask[:, :, :, attn_mask.shape[3] - attn_mask.shape[2] :]
    attn_mask = ttnn.as_tensor(
        attn_mask, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
    )
    ref_out = example["outputs/intermediates"]["attn_weights_post_softmax"]
    return K_past, V_past, q_rotated, k_rotated, v_states, attn_mask, ref_out


def compare_torch_tt(x_tt, x_torch, device):
    # Shape and dtype check
    x_tt = ttnn.to_torch(ttnn.from_device(x_tt))
    mask = torch.isclose(x_tt, x_torch, atol=1e-03, rtol=1e-01)
    fail_fraction = mask.float().mean().item()
    fail_percent = fail_fraction * 100
    print("Input range (tt)     : ", torch.min(x_tt), torch.max(x_tt))
    print("Input range  (torch) : ", torch.min(x_torch), torch.max(x_torch))
    print("Fail percent         : ", fail_percent)
    print("Pcc                  : ", pcc(x_tt, x_torch))

    # Get indices where it's False
    mismatch_idx = torch.nonzero(~mask, as_tuple=False)

    # Collect mismatched values from both tensors
    mismatches = [(idx.tolist(), x_tt[tuple(idx)].item(), x_torch[tuple(idx)].item()) for idx in mismatch_idx]

    for idx, va, vb in mismatches:
        print(f"Index {idx}: a={va}, b={vb}")

    return


if __name__ == "__main__":
    # Load reference example
    root = os.getcwd()
    ref_data_path = os.path.join(root, "reference_data")
    example = torch.load(os.path.join(ref_data_path, "example_with_intermediates.pt"))
    # Load reference example

    # Example use of the AttentionBlock class
    print("Opening device")
    device = ttnn.open_device(device_id=0)
    try:
        # Initialize attention block
        attention_block = AttentionBlock(
            device, model_path="/data/models/meta-llama/Llama-3.1-8B-Instruct", layer_idx=0
        )

        # Setup tt inputs
        hidden_states = example["args/hidden_states_post_norm"]
        attention_mask = example["args/attention_mask"]
        position_ids = example["args/position_ids"]
        position_ids = torch.squeeze(position_ids).tolist()
        past_key_value = example["args/past_key_value"]
        past_k = past_key_value[0][:, :, 0 : attention_mask.shape[3] - attention_mask.shape[2], :]
        past_v = past_key_value[1][:, :, 0 : attention_mask.shape[3] - attention_mask.shape[2], :]
        attention_mask = attention_mask[:, :, :, attention_mask.shape[3] - attention_mask.shape[2] :]
        hidden_states = ttnn.as_tensor(
            hidden_states,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        # Setup tt inputs

        print("Input attention mask : ", attention_mask.shape, attention_mask.dtype)
        print("Attention mask m/M   : ", torch.min(attention_mask), torch.max(attention_mask))
        attention_mask = torch.clamp(attention_mask, min=-1e09)
        attention_mask = ttnn.as_tensor(
            attention_mask,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
        )

        # Setup prefix KV cache on tt
        print("")
        print("Trying to forward with..")
        print("Ref inputs E and M   : ", hidden_states.shape, attention_mask.shape)
        print("Past kv              : ", past_k.shape, past_v.shape)
        attention_block.manual_kv_add(past_k, past_v)
        # Setup prefix KV cache on tt

        MODEL_FORWARD = False
        ATTN_TEST = True

        if ATTN_TEST:
            kv_k, kv_v, qr, kr, v, attn_mask, ref_out = prepare_attention_in(example=example, device=device)
            qkt_tt = compute_attention_with_cache_and_current(kv_k, kv_v, qr, kr, v, attn_mask)
            print("")
            print("Attn test : Single example")
            compare_torch_tt(qkt_tt, ref_out, device)

        if MODEL_FORWARD:
            # Forward pass
            output, tt_intermediates = attention_block.forward(
                input_tokens=hidden_states, position_indices=position_ids, attention_mask=attention_mask
            )
            print("")
            print(f"First forward pass output: {output.shape}")
            # Forward pass

            atol = 1e-04
            rtol = 1e-02

            # Check RoPE : Inputs
            print("")
            print("RoPE inputs / QKV out")
            rope_inputs_q_tt_model = tt_intermediates["q_proj"]
            rope_inputs_k_tt_model = tt_intermediates["k_proj"]
            rope_inputs_q_tt_model = ttnn.to_torch(ttnn.from_device(rope_inputs_q_tt_model)).to(torch.bfloat16)
            rope_inputs_k_tt_model = ttnn.to_torch(ttnn.from_device(rope_inputs_k_tt_model)).to(torch.bfloat16)

            rope_inputs_q_reference = example["outputs/intermediates"]["q_pre_rope"]
            rope_inputs_k_reference = example["outputs/intermediates"]["k_pre_rope"]

            errors_q = torch.abs((rope_inputs_q_tt_model - rope_inputs_q_reference))
            mean_error_q = torch.mean(errors_q)
            max_error_q = torch.max(errors_q)
            print("Q errors        : ", mean_error_q, max_error_q)
            print("Pcc (Q) : ", pcc(rope_inputs_q_tt_model, rope_inputs_q_reference))
            # print(rope_inputs_q_tt_model.dtype, rope_inputs_q_reference.dtype)
            print(
                "All close check : ",
                torch.allclose(rope_inputs_q_tt_model, rope_inputs_q_reference, atol=1e-04, rtol=1e-02),
            )
            errors_k = torch.abs((rope_inputs_k_tt_model - rope_inputs_k_reference))
            mean_error_k = torch.mean(errors_k)
            max_error_k = torch.max(errors_k)
            print("K errors        : ", mean_error_k, max_error_k)
            print("Pcc (K) : ", pcc(rope_inputs_k_tt_model, rope_inputs_k_reference))
            print(
                "All close check : ",
                torch.allclose(rope_inputs_k_tt_model, rope_inputs_k_reference, atol=1e-04, rtol=1e-02),
            )
            # Check RoPE : Inputs

            # Check RoPE : Outputs
            print("")
            print("RoPE outputs")
            rope_outputs_q_tt_model = tt_intermediates["q_post_rope"]
            rope_outputs_k_tt_model = tt_intermediates["k_post_rope"]
            rope_outputs_q_reference = example["outputs/intermediates"]["q_post_rope"]
            rope_outputs_k_reference = example["outputs/intermediates"]["k_post_rope"]
            rope_outputs_q_tt_model = ttnn.to_torch(ttnn.from_device(rope_outputs_q_tt_model)).to(torch.bfloat16)
            rope_outputs_k_tt_model = ttnn.to_torch(ttnn.from_device(rope_outputs_k_tt_model)).to(torch.bfloat16)
            errors_q = torch.abs((rope_outputs_q_tt_model - rope_outputs_q_reference))
            mean_error_q = torch.mean(errors_q)
            max_error_q = torch.max(errors_q)
            print("Q errors        : ", mean_error_q, max_error_q)
            print("Pcc (Q) : ", pcc(rope_outputs_q_tt_model, rope_outputs_q_reference))
            print(
                "All close check : ",
                torch.allclose(rope_outputs_q_tt_model, rope_outputs_q_reference, atol=1e-04, rtol=1e-02),
            )

            errors_k = torch.abs((rope_outputs_k_tt_model - rope_outputs_k_reference))
            mean_error_k = torch.mean(errors_k)
            max_error_k = torch.max(errors_k)
            print("K errors        : ", mean_error_k, max_error_k)
            print("Pcc (K) : ", pcc(rope_inputs_k_tt_model, rope_inputs_k_reference))
            print(
                "All close check : ",
                torch.allclose(rope_outputs_q_tt_model, rope_outputs_q_reference, atol=1e-04, rtol=1e-02),
            )
            # Check RoPE : Outputs

            # Check stuff added to secondary KV cache. Initial KV cache has been verified.
            """
            print("")
            print("KV cache")
            kv_filled_len = tt_intermediates["kv_len_pre_attention"]
            tt_k_past = ttnn.to_torch(ttnn.from_device(tt_intermediates["K_cache_pre_attention"])).to(torch.bfloat16)[
                :, :, 0:kv_filled_len, :
            ]
            tt_k_past = ttnn.to_torch(ttnn.from_device(tt_intermediates["K_cache_pre_attention"])).to(torch.bfloat16)[
                :, :, 0:kv_filled_len, :
            ]
            k_past_reference = torch.cat(
                [example["outputs/intermediates"]["K_past"], example["outputs/intermediates"]["k_post_rope"]], dim=2
            )
            v_past_reference = torch.cat(
                [example["outputs/intermediates"]["V_past"], example["outputs/intermediates"]["v_pre_rope"]], dim=2
            )
            # k_past_reference= k_past_reference.repeat(1,4,1,1)
            # v_past_reference= v_past_reference.repeat(1,4,1,1)
            print(k_past_reference.shape, tt_k_past.shape)
            errors_k = torch.abs((tt_k_past - k_past_reference))
            mean_error_k = torch.mean(errors_k)
            max_error_k = torch.max(errors_k)
            print("K errors        : ", mean_error_k, max_error_k)
            print("Pcc (K) : ", pcc(tt_k_past, k_past_reference))
            print(
                "All close check : ",
                torch.allclose(tt_k_past, k_past_reference, atol=1e-04, rtol=1e-02),
            )
            """
            # Check stuff added to secondary KV cache. Initial KV cache has been verified.

            # Attention weights
            print("")
            print("Attention weights")
            tt_attn_weights_post_scaling = tt_intermediates["attention_intermediates"]["qkt"]
            ref_attn_weights_post_scaling = example["outputs/intermediates"]["attn_weights_scale_applied"]
            tt_attn_weights_post_scaling = ttnn.to_torch(ttnn.from_device(tt_attn_weights_post_scaling)).to(
                torch.bfloat16
            )
            print("Shape of attn weights : ", tt_attn_weights_post_scaling.shape)

            current_seq_len = tt_attn_weights_post_scaling.shape[2]
            total_length = tt_attn_weights_post_scaling.shape[3]

            print("Attn dot products : Q - Prefix")
            tt_attn_weights_post_scaling_prefix = tt_attn_weights_post_scaling[
                :, :, :, 0 : total_length - current_seq_len
            ]
            ref_attn_weights_post_scaling_prefix = ref_attn_weights_post_scaling[
                :, :, :, 0 : total_length - current_seq_len
            ]
            errors = torch.abs((tt_attn_weights_post_scaling_prefix - ref_attn_weights_post_scaling_prefix))
            mean_error, max_error = torch.mean(errors), torch.max(errors)
            print("Mean and max errors : ", mean_error, max_error)
            mask = torch.isclose(tt_attn_weights_post_scaling_prefix, ref_attn_weights_post_scaling_prefix)
            fail_fraction = mask.float().mean().item()
            fail_percent = fail_fraction * 100
            print("Fail percent : ", fail_percent)
            print("Pcc  : ", pcc(tt_attn_weights_post_scaling_prefix, ref_attn_weights_post_scaling_prefix))

            print("Attn dot products : Q - current")
            tt_attn_weights_post_scaling_current = tt_attn_weights_post_scaling[
                :, :, :, total_length - current_seq_len :
            ]
            ref_attn_weights_post_scaling_current = ref_attn_weights_post_scaling[
                :, :, :, total_length - current_seq_len :
            ]
            errors = torch.abs((tt_attn_weights_post_scaling_current - ref_attn_weights_post_scaling_current))
            mean_error, max_error = torch.mean(errors), torch.max(errors)
            print("Mean and max errors : ", mean_error, max_error)
            mask = torch.isclose(tt_attn_weights_post_scaling_current, ref_attn_weights_post_scaling_current)
            fail_fraction = mask.float().mean().item()
            fail_percent = fail_fraction * 100
            print("Fail percent : ", fail_percent)
            print("Pcc  : ", pcc(tt_attn_weights_post_scaling_current, ref_attn_weights_post_scaling_current))
            # Attention weights

            # Get reference output
            print("")
            print("Output check.")
            torch_output = example["outputs/ref_output"]
            tt_output = ttnn.to_torch(ttnn.from_device(output)).to(dtype=torch.bfloat16)
            errors = torch.abs((torch_output - tt_output))
            mean_error = torch.mean(errors)
            max_error = torch.max(errors)
            print(
                "torch allclose (atol 1e-04, rtol 1e-02) output : ",
                torch.allclose(torch_output, tt_output, atol=1e-04, rtol=1e-02),
            )
            print("Mean and Max L1 errors : ", mean_error, max_error)
            mask = torch.abs(torch_output - tt_output) > (atol + rtol * torch.abs(tt_output))
            # fraction (or %) of elements failing the criterion
            fail_fraction = mask.float().mean().item()
            fail_percent = fail_fraction * 100
            print("Fail percent : ", fail_percent)
            print("Pcc (q) : ", pcc(torch_output, tt_output))
            # Get reference output

            # Check KV cache state
            kv_state = attention_block.get_kv_cache_state()
            print(f"KV cache state: {kv_state}")
            # Check KV cache state

            # Cleanup
            attention_block.cleanup()

    finally:
        print("Closing device")
        ttnn.close_device(device)
