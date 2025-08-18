#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Test script to compare numerics of vanilla RMS Norm vs distributed RMS Norm.
This script:
1. Sets seeds for reproducibility
2. Creates a tensor of shape [1,1,32,8192] with elements in range (-2,2)
3. Opens a T3K mesh device (1,8)
4. Compares distributed RMS Norm vs vanilla RMS Norm with all-gather
5. Measures L2 norm error between outputs
6. Runs multiple experiments to generate error histogram
"""

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

import ttnn
from models.common.rmsnorm import RMSNorm
from models.tt_transformers.tt.distributed_norm import DistributedNorm

TILE_SIZE = 32
TILE_PADDED_BATCH_ROWS = 32
DIM = 8192


def set_seeds(seed: int = 26):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_test_tensor(shape: Tuple[int, int, int, int], min_val: float = -2.0, max_val: float = 2.0) -> torch.Tensor:
    """Create a test tensor with elements in the specified range."""
    return torch.rand(shape, dtype=torch.bfloat16) * (max_val - min_val) + min_val


def setup_t3k_device() -> ttnn.Device:
    """Setup T3K mesh device with shape (1,8)."""
    try:
        # Try to create a mesh device with 8 devices
        mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))
        print(f"Successfully created T3K mesh device with shape: {mesh_device.shape}")
        return mesh_device
    except Exception as e:
        print(f"Failed to create T3K mesh device: {e}")
        print("Falling back to single device for testing...")
        return ttnn.open_device(device_id=0)


def create_dummy_weights(dim: int, device: ttnn.Device) -> ttnn.Tensor:
    """Create dummy weights for RMS Norm."""
    weight = torch.ones(dim, dtype=torch.bfloat16)
    return ttnn.as_tensor(
        weight, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def find_grid(N):
    """
    Find the number of rows and columns for a grid of cores such that
    the total number of tiles N can be evenly divided among the cores.
    Each core will have the same integer number of tiles.
    The grid size is limited to a maximum of 2 rows and 8 columns.

    Parameters:
        N (int): Total number of tiles to be distributed.

    Returns:
        tuple: A tuple (rows, cols) representing the grid dimensions.

    Raises:
        AssertionError: If it's not possible to find such a grid configuration.
    """
    max_rows = 8
    max_cols = 8
    max_cores = max_rows * max_cols

    # Find all possible numbers of cores that divide N and are less than or equal to max_cores
    target = 32
    possible_cores = [k for k in range(1, max_cores + 1) if N % k == 0]
    possible_cores.sort(key=lambda x: abs(x - target))  # Sort by closest to target

    for cores in possible_cores:
        # Try to find a grid configuration with the current number of cores
        for rows in range(1, max_rows + 1):
            if cores % rows == 0:
                cols = cores // rows
                if cols <= max_cols:
                    return rows, cols

    # If no configuration is found, assert an error
    raise AssertionError(
        f"Cannot find a grid configuration for {N} tiles that evenly divides into {max_cores} cores of max size {max_rows}x{max_cols}."
    )


def dram_shard_core_grid_for_k(k: int) -> Tuple[int, int]:
    rows, cols = find_grid(k // TILE_SIZE)
    return ttnn.CoreGrid(x=cols, y=rows)


def shard_tensor_across_devices(
    tensor: torch.Tensor, mesh_device: ttnn.Device, force_mem_cfg: bool = True, dim: int = 3
) -> ttnn.Tensor:
    """Shard a tensor across devices along the specified dimension."""
    # Convert to ttnn tensor first
    tt_tensor = ttnn.from_torch(tensor, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3), layout=ttnn.TILE_LAYOUT)
    tt_tensor = ttnn.to_device(tt_tensor, mesh_device)
    print("Created mesh tensor : ")
    ttnn.visualize_mesh_device(mesh_device, tensor=tt_tensor)
    # Create sharded memory config
    shard_shape = list(tensor.shape)
    shard_shape[dim] = shard_shape[dim] // mesh_device.get_num_devices()

    if force_mem_cfg:
        sharded_mem_cfg = ttnn.create_sharded_memory_config(
            shape=shard_shape,
            core_grid=ttnn.CoreGrid(y=1, x=8),  # 8 devices in a row
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        ttnn.to_memory_config(tt_tensor, sharded_mem_cfg)
    # Convert to sharded layout
    return tt_tensor


def create_sharded_norm_config(grid):
    """Helper function to create LayerNormShardedMultiCoreProgramConfig for RMS NORM.

    Args:
        grid (ttnn.CoreGrid): Grid specification for the norm operation
    """
    # compute_grid = ttnn.CoreGrid(x=4, y=8)
    block_w = DIM // grid.num_cores // TILE_SIZE
    # Find largest value <= 4 that evenly divides block_w
    subblock_w = 4
    while subblock_w > 0:
        if block_w % subblock_w == 0:
            break
        subblock_w -= 1

    return ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[grid.x, grid.y],
        subblock_w=subblock_w,
        block_h=TILE_PADDED_BATCH_ROWS // TILE_SIZE,
        block_w=block_w,
        inplace=False,
    )


def perform_distributed_rmsnorm(
    sharded_tensor: ttnn.Tensor, mesh_device: ttnn.Device, weight: ttnn.Tensor, epsilon: float = 1e-5
) -> ttnn.Tensor:
    """Perform distributed RMS Norm using the distributed implementation."""

    attn_input_grid = dram_shard_core_grid_for_k(DIM)
    sharded_norm_program_config = create_sharded_norm_config(attn_input_grid)
    sharded_norm_op_memory_config = ttnn.create_sharded_memory_config(
        (
            TILE_PADDED_BATCH_ROWS,
            DIM // attn_input_grid.num_cores,
        ),  # Shard shape: [32, 128] -> 1 shard per core
        attn_input_grid,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    norm_object = DistributedNorm(
        RMSNorm(
            device=mesh_device,
            dim=DIM,
            eps=epsilon,
            state_dict=torch.rand(1, 1, 1, DIM),
            layer_num=0,
            state_dict_prefix=None,
            weight_cache_path=None,
            weight_dtype=ttnn.bfloat16,
            weight_key="pick_from_state_dict",
            is_distributed=True,
            add_unit_offset=False,
            sharded_program_config=sharded_norm_program_config,
            sharded_output_config=sharded_norm_op_memory_config,
            ccl_topology=ttnn.Topology.Ring,
        ),
        args={},
        TG=False,
    )

    try:
        # Use the distributed RMS Norm function
        output = norm_object.forward(
            inp=sharded_tensor,
            epsilon=epsilon,
            gamma=weight,
            mesh_device=mesh_device,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=True,  # Force FP32 accumulation
                packer_l1_acc=True,
            ),
        )
        return output
    except Exception as e:
        print(f"Error in distributed RMS Norm: {e}")
        raise


def perform_vanilla_rmsnorm_with_allgather(
    sharded_tensor: ttnn.Tensor, mesh_device: ttnn.Device, weight: ttnn.Tensor, epsilon: float = 1e-5
) -> ttnn.Tensor:
    """Perform vanilla RMS Norm by first all-gathering the tensor."""
    try:
        # First, all-gather the sharded tensor to replicate it on all devices
        gathered_tensor = ttnn.all_gather(
            sharded_tensor, dim=3, num_links=1, topology=ttnn.Topology.Ring, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # Now perform vanilla RMS Norm on the replicated tensor
        output = ttnn.rms_norm(
            gathered_tensor,
            weight=weight,
            epsilon=epsilon,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=True,  # Force FP32 accumulation
                packer_l1_acc=True,
            ),
        )

        # Clean up intermediate tensor
        gathered_tensor.deallocate(True)

        return output
    except Exception as e:
        print(f"Error in vanilla RMS Norm with all-gather: {e}")
        raise


def compute_l2_error(tensor1: ttnn.Tensor, tensor2: ttnn.Tensor) -> float:
    """Compute L2 norm error between two tensors."""
    try:
        # Convert both tensors to torch tensors on host
        if hasattr(tensor1, "cpu"):
            t1 = tensor1.cpu()
        else:
            t1 = ttnn.to_torch(tensor1)

        if hasattr(tensor2, "cpu"):
            t2 = tensor2.cpu()
        else:
            t2 = ttnn.to_torch(tensor2)

        # Ensure both tensors are on CPU and have the same shape
        t1 = t1.cpu().float()
        t2 = t2.cpu().float()

        # Compute L2 norm of the difference
        diff = t1 - t2
        l2_error = torch.norm(diff, p=2).item()

        return l2_error
    except Exception as e:
        print(f"Error computing L2 error: {e}")
        return float("inf")


def run_single_experiment(mesh_device: ttnn.Device, weight: ttnn.Tensor, epsilon: float = 1e-5) -> float:
    """Run a single experiment comparing distributed vs vanilla RMS Norm."""
    try:
        print("")
        # Create test tensor
        test_tensor = create_test_tensor((1, 1, TILE_SIZE, DIM))
        print("Test Tensor created.")

        # Shard the tensor across devices
        sharded_tensor = shard_tensor_across_devices(test_tensor, mesh_device)
        print("Tensor sharded across devices")

        # Perform distributed RMS Norm
        distributed_output = perform_distributed_rmsnorm(sharded_tensor, mesh_device, weight, epsilon)
        print("Distributed RMS Norm o/p computed")

        # Perform vanilla RMS Norm with all-gather
        vanilla_output = perform_vanilla_rmsnorm_with_allgather(sharded_tensor, mesh_device, weight, epsilon)

        # Compute L2 error
        l2_error = compute_l2_error(distributed_output, vanilla_output)

        # Clean up
        sharded_tensor.deallocate(True)
        distributed_output.deallocate(True)
        vanilla_output.deallocate(True)

        return l2_error

    except Exception as e:
        print(f"Error in single experiment: {e}")
        return float("inf")


def run_multiple_experiments(
    mesh_device: ttnn.Device, weight: ttnn.Tensor, num_experiments: int = 100, epsilon: float = 1e-5
) -> List[float]:
    """Run multiple experiments and return list of L2 errors."""
    errors = []

    print(f"Running {num_experiments} experiments...")

    for i in range(num_experiments):
        if i % 10 == 0:
            print(f"Progress: {i}/{num_experiments}")

        # Set different seed for each experiment
        set_seeds(42 + i)

        error = run_single_experiment(mesh_device, weight, epsilon)
        errors.append(error)

    print(f"Completed {num_experiments} experiments")
    return errors


def plot_error_histogram(errors: List[float], save_path: str = "rmsnorm_error_histogram.png"):
    """Plot histogram of L2 errors."""
    plt.figure(figsize=(10, 6))

    # Filter out infinite errors
    valid_errors = [e for e in errors if e != float("inf")]

    if not valid_errors:
        print("No valid errors to plot")
        return

    plt.hist(valid_errors, bins=50, alpha=0.7, edgecolor="black")
    plt.xlabel("L2 Error")
    plt.ylabel("Frequency")
    plt.title("Histogram of L2 Errors: Distributed vs Vanilla RMS Norm")
    plt.grid(True, alpha=0.3)

    # Add statistics
    mean_error = np.mean(valid_errors)
    std_error = np.std(valid_errors)
    plt.axvline(mean_error, color="red", linestyle="--", label=f"Mean: {mean_error:.6f}")
    plt.axvline(
        mean_error + std_error, color="orange", linestyle="--", label=f"Mean + Std: {mean_error + std_error:.6f}"
    )
    plt.axvline(
        mean_error - std_error, color="orange", linestyle="--", label=f"Mean - Std: {mean_error - std_error:.6f}"
    )

    plt.legend()
    plt.tight_layout()

    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Histogram saved to {save_path}")

    # Display plot
    plt.show()


def print_error_statistics(errors: List[float]):
    """Print statistics about the L2 errors."""
    valid_errors = [e for e in errors if e != float("inf")]

    if not valid_errors:
        print("No valid errors to analyze")
        return

    print("\n" + "=" * 50)
    print("ERROR STATISTICS")
    print("=" * 50)
    print(f"Total experiments: {len(errors)}")
    print(f"Valid experiments: {len(valid_errors)}")
    print(f"Failed experiments: {len(errors) - len(valid_errors)}")
    print(f"Success rate: {len(valid_errors)/len(errors)*100:.1f}%")
    print()
    print(f"Mean L2 Error: {np.mean(valid_errors):.8f}")
    print(f"Std L2 Error: {np.std(valid_errors):.8f}")
    print(f"Min L2 Error: {np.min(valid_errors):.8f}")
    print(f"Max L2 Error: {np.max(valid_errors):.8f}")
    print(f"Median L2 Error: {np.median(valid_errors):.8f}")
    print()
    print(f"25th percentile: {np.percentile(valid_errors, 25):.8f}")
    print(f"75th percentile: {np.percentile(valid_errors, 75):.8f}")
    print("=" * 50)


def main():
    """Main function to run the RMS Norm comparison test."""
    print("Starting RMS Norm Numerics Comparison Test")
    print("=" * 50)

    # 0. Set seeds
    set_seeds(26)
    print("✓ Seeds set for reproducibility")

    # 1. Create test tensor
    test_tensor = create_test_tensor((1, 1, 32, 8192))
    print(f"✓ Created test tensor with shape {test_tensor.shape}")
    print(f"  Tensor range: [{test_tensor.min():.3f}, {test_tensor.max():.3f}]")

    # 2. Setup T3K mesh device
    mesh_device = setup_t3k_device()
    print(f"✓ Device setup complete: {type(mesh_device).__name__}")

    # Create dummy weights
    weight = create_dummy_weights(8192, mesh_device)
    print("✓ Created dummy weights for RMS Norm")

    # 3. Run single experiment first to test setup
    print("\nRunning single experiment to test setup...")
    try:
        single_error = run_single_experiment(mesh_device, weight)
        print(f"✓ Single experiment successful, L2 error: {single_error:.8f}")
    except Exception as e:
        print(f"✗ Single experiment failed: {e}")
        return

    exit(0)
    # 4. Run multiple experiments
    num_experiments = 50  # Adjust as needed
    print(f"\nRunning {num_experiments} experiments...")

    try:
        errors = run_multiple_experiments(mesh_device, weight, num_experiments)

        # 5. Analyze results
        print_error_statistics(errors)

        # 6. Plot histogram
        plot_error_histogram(errors)

        print("\n✓ Test completed successfully!")

    except Exception as e:
        print(f"✗ Error during multiple experiments: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        try:
            weight.deallocate(True)
            if hasattr(mesh_device, "close"):
                mesh_device.close()
            print("✓ Cleanup completed")
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")


if __name__ == "__main__":
    main()
