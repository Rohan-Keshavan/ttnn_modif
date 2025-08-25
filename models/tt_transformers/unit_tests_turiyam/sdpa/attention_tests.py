"""
Unit tests for AttentionBlock class.

This module contains comprehensive tests for:
- AttentionBlock initialization and configuration
- GQA weight expansion
- RoPE application
- KV cache management
- Forward pass computation
- Memory management and cleanup
"""

import pytest
import torch

# Import utility functions
from attention_utils import benchmark_forward_pass, generate_rectangular_attention_mask

# Import the AttentionBlock class
from tt_func_api_sdpa import AttentionBlock

import ttnn


class TestAttentionBlock:
    """Test suite for AttentionBlock class."""

    @pytest.fixture(scope="class")
    def device(self):
        """Create TT-Metal device for testing."""
        device = ttnn.open_device(device_id=0)
        yield device
        ttnn.close_device(device)

    @pytest.fixture
    def attention_block(self, device):
        """Create AttentionBlock instance for testing."""
        # Use random weights for testing
        attention_block = AttentionBlock(device, model_path=None, layer_idx=0)
        yield attention_block
        # Cleanup after each test
        attention_block.cleanup()

    def test_initialization(self, attention_block):
        """Test AttentionBlock initialization."""
        print("\n=== Testing Initialization ===")

        # Check default parameters
        assert attention_block.MODEL_HIDDEN == 4096
        assert attention_block.N_QHEADS == 32
        assert attention_block.N_KVHEADS == 8
        assert attention_block.HEAD_DIM == 128
        assert attention_block.GQA_GROUP_SIZE == 4

        # Check that weights are generated
        assert attention_block.Q is not None
        assert attention_block.K is not None
        assert attention_block.V is not None
        assert attention_block.O is not None

        # Check RoPE setup
        assert attention_block.rope_setup is not None
        assert attention_block.trans_mats_dict is not None

        # Check KV cache initialization
        assert attention_block.K_past is not None
        assert attention_block.V_past is not None
        assert attention_block.kv_len == 0

        print("✅ Initialization test passed")

    def test_gqa_weight_expansion(self, attention_block):
        """Test GQA weight expansion logic."""
        print("\n=== Testing GQA Weight Expansion ===")

        # Get original shapes
        original_K_shape = attention_block.K.shape
        original_V_shape = attention_block.V.shape

        print(f"Original K shape: {original_K_shape}")
        print(f"Original V shape: {original_V_shape}")

        # Test weight expansion
        K_expanded, V_expanded = attention_block._expand_gqa_weights()

        # Check expanded shapes
        expected_K_shape = (1, 1, attention_block.N_QHEADS * attention_block.HEAD_DIM, attention_block.MODEL_HIDDEN)
        expected_V_shape = (1, 1, attention_block.N_QHEADS * attention_block.HEAD_DIM, attention_block.MODEL_HIDDEN)

        assert K_expanded.shape == expected_K_shape
        assert V_expanded.shape == expected_V_shape

        print(f"Expanded K shape: {K_expanded.shape}")
        print(f"Expanded V shape: {V_expanded.shape}")
        print(f"GQA group size: {attention_block.GQA_GROUP_SIZE}")

        # Cleanup
        K_expanded.deallocate(True)
        V_expanded.deallocate(True)

        print("✅ GQA weight expansion test passed")

    def test_rope_application(self, attention_block):
        """Test RoPE application to Q and K."""
        print("\n=== Testing RoPE Application ===")

        # Create test tensors
        seq_len = 8
        batch_size = 1
        n_heads = attention_block.N_QHEADS
        head_dim = attention_block.HEAD_DIM

        # Create test Q and K tensors
        q_test = torch.randn(seq_len, batch_size, n_heads, head_dim)
        k_test = torch.randn(seq_len, batch_size, n_heads, head_dim)

        # Convert to TT-Metal tensors
        q_tt = ttnn.from_torch(
            q_test,
            device=attention_block.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        k_tt = ttnn.from_torch(
            k_test,
            device=attention_block.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Test position indices
        position_indices = [0, 1, 2, 3, 4, 5, 6, 7]

        # Apply RoPE
        q_rotated, k_rotated = attention_block._apply_rope(q_tt, k_tt, position_indices)

        # Check output shapes
        assert q_rotated.shape == q_tt.shape
        assert k_rotated.shape == k_tt.shape

        print(f"Q rotated shape: {q_rotated.shape}")
        print(f"K rotated shape: {k_rotated.shape}")

        # Cleanup
        q_tt.deallocate(True)
        k_tt.deallocate(True)
        q_rotated.deallocate(True)
        k_rotated.deallocate(True)

        print("✅ RoPE application test passed")

    def test_kv_cache_management(self, attention_block):
        """Test KV cache initialization and management."""
        print("\n=== Testing KV Cache Management ===")

        # Check initial state
        initial_state = attention_block.get_kv_cache_state()
        assert initial_state["current_length"] == 0
        assert initial_state["current_capacity"] == 64  # Default initial capacity
        assert initial_state["max_capacity"] == 8192  # Default max capacity
        assert initial_state["K_shape"] == (64, 1, 32, 128)  # Initial capacity
        assert initial_state["V_shape"] == (64, 1, 32, 128)

        print(f"Initial KV cache state: {initial_state}")

        # Test cache reset
        attention_block.reset_kv_cache()
        reset_state = attention_block.get_kv_cache_state()
        assert reset_state["current_length"] == 0
        assert reset_state["current_capacity"] == 64

        print("✅ KV cache management test passed")

    def test_forward_pass_basic(self, attention_block):
        """Test basic forward pass functionality."""
        print("\n=== Testing Basic Forward Pass ===")

        # Create test input
        seq_len = 16
        input_tokens = torch.randn(1, seq_len, attention_block.MODEL_HIDDEN)
        position_indices = list(range(seq_len))

        print(f"Input tokens shape: {input_tokens.shape}")
        print(f"Position indices: {position_indices}")

        # Forward pass
        output = attention_block.forward(input_tokens, position_indices)

        # Check output shape
        expected_output_shape = (1, seq_len, attention_block.MODEL_HIDDEN)
        assert output.shape == expected_output_shape

        print(f"Output shape: {output.shape}")
        print(f"Expected shape: {expected_output_shape}")

        # Check KV cache was updated
        kv_state = attention_block.get_kv_cache_state()
        assert kv_state["current_length"] == seq_len

        print(f"KV cache state after forward pass: {kv_state}")

        print("✅ Basic forward pass test passed")

    def test_forward_pass_multiple(self, attention_block):
        """Test multiple forward passes with KV cache maintenance."""
        print("\n=== Testing Multiple Forward Passes ===")

        # First forward pass
        seq_len1 = 10
        input_tokens1 = torch.randn(1, seq_len1, attention_block.MODEL_HIDDEN)
        position_indices1 = list(range(seq_len1))

        output1 = attention_block.forward(input_tokens1, position_indices1)
        print(f"First forward pass output: {output1.shape}")

        # Check KV cache state
        kv_state1 = attention_block.get_kv_cache_state()
        print(f"KV cache after first pass: {kv_state1}")
        assert kv_state1["current_length"] == seq_len1

        # Second forward pass
        seq_len2 = 8
        input_tokens2 = torch.randn(1, seq_len2, attention_block.MODEL_HIDDEN)
        position_indices2 = list(range(seq_len1, seq_len1 + seq_len2))

        output2 = attention_block.forward(input_tokens2, position_indices2)
        print(f"Second forward pass output: {output2.shape}")

        # Check KV cache state
        kv_state2 = attention_block.get_kv_cache_state()
        print(f"KV cache after second pass: {kv_state2}")
        assert kv_state2["current_length"] == seq_len1 + seq_len2

        print("✅ Multiple forward passes test passed")

    def test_forward_pass_with_duplicate_positions(self, attention_block):
        """Test forward pass with duplicate position indices (Eagle-style)."""
        print("\n=== Testing Forward Pass with Duplicate Positions ===")

        # Create test input with duplicate positions
        seq_len = 12
        input_tokens = torch.randn(1, seq_len, attention_block.MODEL_HIDDEN)

        # Create duplicate position pattern: [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        position_indices = []
        for i in range(seq_len):
            position_indices.append(i // 2)

        print(f"Input tokens shape: {input_tokens.shape}")
        print(f"Position indices (with duplicates): {position_indices}")

        # Forward pass
        output = attention_block.forward(input_tokens, position_indices)

        # Check output shape
        expected_output_shape = (1, seq_len, attention_block.MODEL_HIDDEN)
        assert output.shape == expected_output_shape

        print(f"Output shape: {output.shape}")

        # Check KV cache was updated
        kv_state = attention_block.get_kv_cache_state()
        print(f"KV cache state: {kv_state}")

        print("✅ Forward pass with duplicate positions test passed")

    def test_attention_mask_handling(self, attention_block):
        """Test attention mask handling in forward pass."""
        print("\n=== Testing Attention Mask Handling ===")

        # Create test input
        seq_len = 8
        input_tokens = torch.randn(1, seq_len, attention_block.MODEL_HIDDEN)
        position_indices = list(range(seq_len))

        # Create rectangular attention mask
        current_seq_len = attention_block.kv_len
        attention_mask = generate_rectangular_attention_mask(
            current_seq_len, seq_len, batch_size=1, n_heads=attention_block.N_QHEADS, device=attention_block.device
        )

        print(f"Input tokens shape: {input_tokens.shape}")
        print(f"Attention mask shape: {attention_mask.shape}")

        # Forward pass with mask
        output = attention_block.forward(input_tokens, position_indices, attention_mask)

        # Check output shape
        expected_output_shape = (1, seq_len, attention_block.MODEL_HIDDEN)
        assert output.shape == expected_output_shape

        print(f"Output shape: {output.shape}")

        # Cleanup mask
        attention_mask.deallocate(True)

        print("✅ Attention mask handling test passed")

    def test_memory_management(self, attention_block):
        """Test memory management and cleanup."""
        print("\n=== Testing Memory Management ===")

        # Check initial state
        initial_state = attention_block.get_kv_cache_state()
        print(f"Initial KV cache state: {initial_state}")

        # Run a forward pass to create some tensors
        seq_len = 16
        input_tokens = torch.randn(1, seq_len, attention_block.MODEL_HIDDEN)
        position_indices = list(range(seq_len))

        output = attention_block.forward(input_tokens, position_indices)
        print(f"Forward pass output: {output.shape}")

        # Check KV cache state
        kv_state = attention_block.get_kv_cache_state()
        print(f"KV cache state after forward pass: {kv_state}")

        # Test cache clearing
        attention_block.clear_kv_cache()
        cleared_state = attention_block.get_kv_cache_state()
        assert cleared_state["K_shape"] is None
        assert cleared_state["V_shape"] is None
        assert cleared_state["current_length"] == 0

        print("✅ Memory management test passed")

    def test_benchmarking(self, attention_block):
        """Test benchmarking functionality."""
        print("\n=== Testing Benchmarking ===")

        # Create test input
        seq_len = 16
        input_tokens = torch.randn(1, seq_len, attention_block.MODEL_HIDDEN)
        position_indices = list(range(seq_len))

        # Run benchmark
        benchmark_results = benchmark_forward_pass(attention_block, input_tokens, position_indices, num_runs=3)

        # Check benchmark results
        assert "average_time_ms" in benchmark_results
        assert "min_time_ms" in benchmark_results
        assert "max_time_ms" in benchmark_results
        assert benchmark_results["num_runs"] == 3

        print(f"Benchmark results: {benchmark_results}")

        print("✅ Benchmarking test passed")

    def test_dynamic_cache_growth(self, attention_block):
        """Test dynamic KV cache growth functionality."""
        print("\n=== Testing Dynamic Cache Growth ===")

        # Check initial capacity
        initial_state = attention_block.get_kv_cache_state()
        assert initial_state["current_capacity"] == 64

        # Add tokens until cache needs to grow
        seq_len = 50  # This should fit in initial capacity
        input_tokens = torch.randn(1, seq_len, attention_block.MODEL_HIDDEN)
        position_indices = list(range(seq_len))

        output = attention_block.forward(input_tokens, position_indices)
        print(f"First forward pass: {output.shape}")

        # Check cache state after first pass
        state_after_first = attention_block.get_kv_cache_state()
        print(f"Cache state after first pass: {state_after_first}")
        assert state_after_first["current_length"] == seq_len
        assert state_after_first["current_capacity"] == 64  # Should still be initial capacity

        # Add more tokens to trigger growth
        seq_len2 = 100  # This should trigger cache growth
        input_tokens2 = torch.randn(1, seq_len2, attention_block.MODEL_HIDDEN)
        position_indices2 = list(range(seq_len, seq_len + seq_len2))

        output2 = attention_block.forward(input_tokens2, position_indices2)
        print(f"Second forward pass: {output2.shape}")

        # Check cache state after growth
        state_after_growth = attention_block.get_kv_cache_state()
        print(f"Cache state after growth: {state_after_growth}")
        assert state_after_growth["current_length"] == seq_len + seq_len2
        assert state_after_growth["current_capacity"] > 64  # Should have grown

        # Test cache optimization
        print("\n--- Testing Cache Optimization ---")
        attention_block.optimize_cache_size(target_utilization=0.8)

        optimized_state = attention_block.get_kv_cache_state()
        print(f"Cache state after optimization: {optimized_state}")

        # Test cache resizing
        print("\n--- Testing Cache Resizing ---")
        attention_block.resize_kv_cache(200)

        resized_state = attention_block.get_kv_cache_state()
        print(f"Cache state after resize: {resized_state}")
        assert resized_state["current_capacity"] == 200

        print("✅ Dynamic cache growth test passed")

    def test_cache_optimization_methods(self, attention_block):
        """Test cache optimization and utility methods."""
        print("\n=== Testing Cache Optimization Methods ===")

        # Add some tokens first
        seq_len = 80
        input_tokens = torch.randn(1, seq_len, attention_block.MODEL_HIDDEN)
        position_indices = list(range(seq_len))

        output = attention_block.forward(input_tokens, position_indices)
        print(f"Added {seq_len} tokens")

        # Test cache utilization
        utilization = attention_block.get_cache_utilization()
        print(f"Cache utilization: {utilization}")

        # Test cache optimization
        print("\n--- Testing Cache Optimization ---")
        attention_block.optimize_cache_size(target_utilization=0.9)

        optimized_utilization = attention_block.get_cache_utilization()
        print(f"Optimized utilization: {optimized_utilization}")

        # Test manual resizing
        print("\n--- Testing Manual Resizing ---")
        attention_block.resize_kv_cache(150)

        resized_utilization = attention_block.get_cache_utilization()
        print(f"Resized utilization: {resized_utilization}")

        print("✅ Cache optimization methods test passed")


def test_attention_block_integration():
    """Integration test for the complete AttentionBlock workflow."""
    print("\n=== Integration Test ===")

    # Open device
    device = ttnn.open_device(device_id=0)

    try:
        # Initialize attention block
        attention_block = AttentionBlock(device, model_path=None, layer_idx=0)

        print("✅ AttentionBlock initialized")

        # Test multiple forward passes with different sequence lengths
        sequence_lengths = [8, 16, 12, 20]
        total_tokens = 0

        for i, seq_len in enumerate(sequence_lengths):
            print(f"\n--- Forward Pass {i+1}: {seq_len} tokens ---")

            # Generate input
            input_tokens = torch.randn(1, seq_len, attention_block.MODEL_HIDDEN)
            position_indices = list(range(total_tokens, total_tokens + seq_len))

            print(f"Input shape: {input_tokens.shape}")
            print(f"Position range: {position_indices[0]} to {position_indices[-1]}")

            # Forward pass
            output = attention_block.forward(input_tokens, position_indices)
            print(f"Output shape: {output.shape}")

            # Check KV cache state
            kv_state = attention_block.get_kv_cache_state()
            print(f"KV cache length: {kv_state['current_length']}")

            total_tokens += seq_len

        print(f"\n✅ Integration test completed successfully!")
        print(f"Total tokens processed: {total_tokens}")
        print(f"Final KV cache length: {attention_block.get_kv_cache_state()['current_length']}")

        # Show final cache utilization
        final_utilization = attention_block.get_cache_utilization()
        print(f"Final cache utilization: {final_utilization}")

        # Cleanup
        attention_block.cleanup()

    finally:
        ttnn.close_device(device)


def test_eagle_style_speculative_decoding():
    """Test Eagle-style speculative decoding with duplicate positions."""
    print("\n=== Eagle-Style Speculative Decoding Test ===")

    # Open device
    device = ttnn.open_device(device_id=0)

    try:
        # Initialize attention block
        attention_block = AttentionBlock(device, model_path=None, layer_idx=0)

        print("✅ AttentionBlock initialized for Eagle test")

        # Simulate Eagle speculative decoding
        # Initial sequence
        initial_seq_len = 20
        input_tokens_initial = torch.randn(1, initial_seq_len, attention_block.MODEL_HIDDEN)
        position_indices_initial = list(range(initial_seq_len))

        print(f"Initial sequence: {initial_seq_len} tokens")
        output_initial = attention_block.forward(input_tokens_initial, position_indices_initial)
        print(f"Initial output shape: {output_initial.shape}")

        # Draft sequence with duplicates (Eagle-style)
        draft_seq_len = 15
        input_tokens_draft = torch.randn(1, draft_seq_len, attention_block.MODEL_HIDDEN)

        # Create duplicate position pattern for Eagle
        # Pattern: [20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27]
        position_indices_draft = []
        for i in range(draft_seq_len):
            if i < 14:  # First 14 tokens have duplicates
                position_indices_draft.append(20 + (i // 2))
            else:  # Last token is unique
                position_indices_draft.append(20 + 7)

        print(f"Draft sequence: {draft_seq_len} tokens")
        print(f"Draft position indices: {position_indices_draft}")

        # Forward pass with duplicate positions
        output_draft = attention_block.forward(input_tokens_draft, position_indices_draft)
        print(f"Draft output shape: {output_draft.shape}")

        # Check final state
        final_kv_state = attention_block.get_kv_cache_state()
        print(f"Final KV cache state: {final_kv_state}")

        print("✅ Eagle-style speculative decoding test completed!")

        # Cleanup
        attention_block.cleanup()

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    print("Running AttentionBlock unit tests...")

    # Run integration test
    test_attention_block_integration()

    # Run Eagle-style test
    test_eagle_style_speculative_decoding()

    print("\n🎉 All tests completed successfully!")
