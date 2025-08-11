"""
Simple example: Call ttnn.rms_norm on bfloat16 tensor with shape (1,1,32,8192)
replicated across (1,8) T3K mesh with core grid of size 32 on each device.
"""

import ttnn
import torch
import numpy as np


def main():
    # 1. Create (1,8) T3K mesh device
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 8),  # 1 row, 8 columns
        physical_device_ids=[],  # Auto-assign from available devices
        num_command_queues=1,
    )

    d_embedding = 8192
    print(f"Created mesh device: {mesh_device.shape}")

    # 2. Create input tensor with shape (1,1,32,8192)
    # This represents: batch=1, seq_len=1, height=32, hidden_dim=8192
    input_data = torch.randn(1, 1, 32, d_embedding, dtype=torch.float32)

    # Convert to TTNN tensor and replicate across mesh
    input_tensor = ttnn.from_torch(
        input_data,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,  # TILE_LAYOUT is optimal for normalization ops
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    print(f"Input tensor created: {input_tensor.shape}")
    print(f"Input tensor layout: {input_tensor.layout}")

    # 3. Create gamma weight tensor with shape (1,1,1,8192) and replicate across mesh
    # Weight must broadcast with the last dimension (hidden_dim=8192)
    gamma_data = torch.ones(1, 1, 1, d_embedding, dtype=torch.float32)

    gamma_weight = ttnn.from_torch(
        gamma_data,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,  # TILE_LAYOUT for optimal performance
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    print(f"Gamma weight created: {gamma_weight.shape}")

    # 4. Create core grid configuration for 32 cores
    # Using 8x4 grid (8 columns, 4 rows = 32 cores)
    core_grid = ttnn.CoreGrid(x=8, y=4)

    # 5. Create program configuration for RMSNorm
    # For input shape (1,1,32,8192) and 32 cores:
    # - Each core handles: 32/32 = 1 tile in height dimension
    # - Each core handles: 8192/32/32 = 8 tiles in width dimension
    # - Tile size is 32x32

    # Calculate block dimensions
    tile_size = 32
    num_cores = core_grid.x * core_grid.y  # 32

    # Height dimension: 32 / 32 = 1 tile per core
    block_h = 1

    # Width dimension: 8192 / 32 / 32 = 8 tiles per core
    block_w = d_embedding // num_cores // tile_size  # = 8

    # subblock_w must divide block_w and be <= 4
    subblock_w = min(4, block_w)  # = 4

    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[core_grid.x, core_grid.y],  # (8, 4)
        subblock_w=subblock_w,  # 4
        block_h=block_h,  # 1
        block_w=block_w,  # 8
        inplace=False,
    )

    program_config = ttnn.LayerNormDefaultProgramConfig()
    print(
        f"Program config: core_grid={core_grid.x}x{core_grid.y}, block_h={block_h}, block_w={block_w}, subblock_w={subblock_w}"
    )

    # 6. Create compute kernel config for T3K (Wormhole)
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    # 6. B Output memory config
    output_memory_config = ttnn.DRAM_MEMORY_CONFIG

    # 7. Call ttnn.rms_norm
    print(f"\nRunning RMSNorm on input shape {input_tensor.shape}...")
    """
    output_tensor = ttnn.rms_norm(
        input_tensor,
        epsilon=1e-5,
        weight=gamma_weight,
        program_config=program_config,
        memory_config=output_memory_config,
        compute_kernel_config=compute_kernel_config,
    )
    """
    output_tensor = ttnn.rms_norm(
        input_tensor,
        epsilon=1e-05,
        weight=gamma_weight,
        compute_kernel_config=compute_kernel_config,
        memory_config=output_memory_config,
        program_config=program_config,
    )
    print(f"RMSNorm completed! Output shape: {output_tensor.shape}")
    print(f"Output tensor layout: {output_tensor.layout}")

    # 9. Cleanup
    ttnn.close_device(mesh_device)
    print("Mesh device closed")


if __name__ == "__main__":
    main()
