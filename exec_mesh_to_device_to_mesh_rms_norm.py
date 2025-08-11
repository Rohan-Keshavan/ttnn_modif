#!/usr/bin/env python3
"""
Test script: RMSNorm on single device from mesh to avoid program_config issues
"""
import ttnn
import torch


def test_rms_norm_single_device():
    print("=== Testing RMSNorm on Single Device from Mesh ===\n")

    try:
        # 1. Create mesh device
        print("1. Creating mesh device...")
        mesh_device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(1, 8), physical_device_ids=[], num_command_queues=1
        )
        print(f"   ✓ Mesh device created: {mesh_device.shape}")

        # 2. Create test tensors
        print("\n2. Creating test tensors...")
        input_data = torch.randn(1, 1, 32, 8192, dtype=torch.float32)
        gamma_data = torch.ones(1, 1, 1, 8192, dtype=torch.float32)

        # 3. Convert to TTNN tensors on mesh (replicated)
        print("3. Converting to TTNN tensors on mesh (replicated)...")
        input_tensor = ttnn.from_torch(
            input_data,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        gamma_weight = ttnn.from_torch(
            gamma_data,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        print(f"   ✓ Input tensor: {input_tensor.shape}")
        print(f"   ✓ Gamma weight: {gamma_weight.shape}")

        # 4. Get single device from mesh
        print("\n4. Getting single device from mesh...")
        mesh_devices = mesh_device.get_devices()
        single_device = mesh_devices[0]
        # single_device = mesh_device.get_device_id(ttnn.MeshCoordinate(0, 0))  # Device at position (0,0)
        print(f"   ✓ Single device: {single_device}")

        # 5. Convert tensors to single device
        print("\n5. Converting tensors to single device...")
        input_single = ttnn.to_device(input_tensor, single_device)
        gamma_single = ttnn.to_device(gamma_weight, single_device)

        print(f"   ✓ Input on single device: {input_single.shape}")
        print(f"   ✓ Gamma on single device: {gamma_single.shape}")

        # 6. Create program config for single device (32 cores)
        print("\n6. Creating program config for single device...")
        program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[8, 4],  # 8x4 = 32 cores
            subblock_w=8,
            block_h=1,
            block_w=8,
            inplace=False,
        )

        # 7. Create compute kernel config
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        print("   ✓ Program config created for 32 cores")
        print("   ✓ Compute kernel config created (HiFi4)")

        # 8. Run RMSNorm on single device
        print("\n7. Running RMSNorm on single device...")
        try:
            output_single = ttnn.rms_norm(
                input_single,
                epsilon=1e-5,
                weight=gamma_single,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            print("   ✓ RMSNorm succeeded on single device!")
            print(f"   Output shape: {output_single.shape}")

        except Exception as e:
            print(f"   ✗ RMSNorm failed on single device: {e}")
            raise

        # 9. Convert result back to mesh device
        print("\n8. Converting result back to mesh device...")
        output_mesh = ttnn.to_device(output_single, mesh_device)
        print(f"   ✓ Output on mesh: {output_mesh.shape}")

        # 10. Convert back to torch for verification
        print("\n9. Converting output back to torch...")
        output_torch = ttnn.to_torch(output_mesh)
        print(f"   ✓ Output converted to torch: {output_torch.shape}")

        # 11. Clean up single device tensors
        print("\n10. Cleaning up single device tensors...")
        input_single.deallocate(True)
        gamma_single.deallocate(True)
        output_single.deallocate(True)
        print("   ✓ Single device tensors cleaned up")

        # 12. Final cleanup
        print("\n11. Final cleanup...")
        ttnn.close_device(mesh_device)
        print("   ✓ Mesh device closed")

        print("\n🎉 Test completed successfully!")
        print("✅ RMSNorm worked on single device with 32 cores")
        print("✅ No 'bad optional access' errors")
        print("✅ Result successfully replicated back to mesh")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print(f"Error type: {type(e)}")
        raise


if __name__ == "__main__":
    test_rms_norm_single_device()
