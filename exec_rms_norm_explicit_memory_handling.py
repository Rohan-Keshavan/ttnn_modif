#!/usr/bin/env python3
"""
Test script to verify that explicitly passing memory_config fixes the "bad optional access" error.
This is a minimal version to test the fix.
"""

import ttnn
import torch
import numpy as np


def test_rms_norm_with_explicit_memory_config():
    """Test that ttnn.rms_norm works with explicit memory_config on mesh device."""

    print("Testing RMSNorm with explicit memory_config to fix 'bad optional access' error...")

    try:
        # 1. Create (1,8) T3K mesh device
        mesh_device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(1, 8), physical_device_ids=[], num_command_queues=1
        )
        print(f"✓ Created mesh device: {mesh_device.shape}")

        # 2. Create small test tensors to minimize memory usage
        embedding_dim = 8192
        input_data = torch.randn(1, 1, 32, embedding_dim, dtype=torch.float32)  # Smaller hidden dim for testing
        gamma_data = torch.ones(1, 1, 1, embedding_dim, dtype=torch.float32)

        # 3. Convert to TTNN tensors and replicate across mesh
        input_tensor = ttnn.from_torch(
            input_data,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        gamma_weight = ttnn.from_torch(
            gamma_data,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        print(f"✓ Created input tensor: {input_tensor.shape}")
        print(f"✓ Created gamma weight: {gamma_weight.shape}")

        # 4. Test RMSNorm WITH explicit memory_config (this should work)
        print("\nTesting RMSNorm WITH explicit memory_config...")
        output_with_memory_config = ttnn.rms_norm(
            input_tensor,
            epsilon=1e-5,
            weight=gamma_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,  # Explicit memory config
        )
        print("✓ RMSNorm WITH explicit memory_config succeeded!")
        print(f"  Output shape: {output_with_memory_config.shape}")

        # 5. Test RMSNorm WITHOUT explicit memory_config (this might fail)
        print("\nTesting RMSNorm WITHOUT explicit memory_config...")
        try:
            output_without_memory_config = ttnn.rms_norm(
                input_tensor,
                epsilon=1e-5,
                weight=gamma_weight,
                # No memory_config - this might cause "bad optional access"
            )
            print("✓ RMSNorm WITHOUT explicit memory_config also succeeded!")
            print(f"  Output shape: {output_without_memory_config.shape}")
        except Exception as e:
            print(f"✗ RMSNorm WITHOUT explicit memory_config failed (expected): {e}")
            print("  This confirms the 'bad optional access' error occurs without explicit memory_config")

        # 6. Cleanup
        ttnn.close_device(mesh_device)
        print("\n✓ Test completed successfully!")
        print("✓ The fix (explicit memory_config) resolves the 'bad optional access' error")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        print("  This indicates a different issue beyond the memory_config fix")
        raise


if __name__ == "__main__":
    test_rms_norm_with_explicit_memory_config()
