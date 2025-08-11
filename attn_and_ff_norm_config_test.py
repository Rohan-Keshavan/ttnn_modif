import ttnn
import torch
from typing import Tuple

tile_padded_batch_rows = 32
dim = 8192
hidden_dim = 8192
num_devices = 8
tile_size = 32


def create_sharded_norm_config(grid):
    """Helper function to create LayerNormShardedMultiCoreProgramConfig for RMS NORM.

    Args:
        grid (ttnn.CoreGrid): Grid specification for the norm operation
    """

    block_w = dim // grid.num_cores // tile_size
    # Find largest value <= 4 that evenly divides block_w
    subblock_w = 4
    while subblock_w > 0:
        if block_w % subblock_w == 0:
            break
        subblock_w -= 1
    return ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[grid.x, grid.y],
        subblock_w=subblock_w,
        block_h=tile_padded_batch_rows // tile_size,
        block_w=block_w,
        inplace=False,
    )


def dram_shard_core_grid_for_k(k: int) -> Tuple[int, int]:
    rows, cols = find_grid(k // tile_size)
    return ttnn.CoreGrid(x=cols, y=rows)


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


def find_grid_k_n(K, N):
    """
    Find the number of rows and columns for a grid of cores such that
    the total number of tiles N can be evenly divided among the cores.
    Each core will have the same integer number of tiles.

    Parameters:
        N (int): Total number of tiles to be distributed.

    Returns:
        tuple: A tuple (rows, cols) representing the grid dimensions.

    Raises:
        AssertionError: If it's not possible to find such a grid configuration.
    """
    max_rows = 8
    max_cols = 8  # Maximum number of rows or columns
    max_cores = max_rows * max_cols  # Maximum number of cores

    # Find all possible numbers of cores that divide N and are less than or equal to max_cores
    possible_cores = [c for c in range(1, max_cores + 1) if K % c == 0 and N % c == 0]
    possible_cores.sort(reverse=True)  # Start checking from the largest number of cores

    for cores in possible_cores:
        # Try to find a grid configuration with the current number of cores
        for rows in range(1, max_rows + 1):
            if cores % rows == 0:
                cols = cores // rows
                if cols <= max_cols:
                    return rows, cols

    # If no configuration is found, assert an error
    raise AssertionError(
        f"Cannot find a grid configuration such that both {K} and {N} tiles evenly divide into cores of max size {max_rows}x{max_cols}."
    )


def dram_shard_core_grid_for_k_and_n(k: int, n: int) -> Tuple[int, int]:
    rows, cols = find_grid_k_n(k // tile_size, n // tile_size)
    return ttnn.CoreGrid(x=cols, y=rows)


if __name__ == "__main__":
    mlp_core_grid = dram_shard_core_grid_for_k_and_n(dim, hidden_dim // num_devices)  # for mlp norm program config
    attn_input_grid = dram_shard_core_grid_for_k(dim)  # for attn norm program config

    model_config = {}
    model_config["SHARDED_NORM_ATTN_PRGM_CFG"] = create_sharded_norm_config(attn_input_grid)
    model_config["SHARDED_NORM_MLP_PRGM_CFG"] = create_sharded_norm_config(mlp_core_grid)

    print("")
    print("Attn norm program config: ", model_config["SHARDED_NORM_ATTN_PRGM_CFG"])
    print("")
    print("MLP norm program config : ", model_config["SHARDED_NORM_MLP_PRGM_CFG"])

    model_config["SHARDED_MLP_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
        (
            tile_padded_batch_rows,
            dim // mlp_core_grid.num_cores,
        ),  # Shard shape: [32, 128] -> 1 shard per core
        mlp_core_grid,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    model_config["SHARDED_ATTN_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
        (
            tile_padded_batch_rows,
            dim // attn_input_grid.num_cores,
        ),  # Shard shape: [32, 128] -> 1 shard per core
        attn_input_grid,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    print("")
    print("MLP input memory config: ", model_config["SHARDED_MLP_INPUT_MEMCFG"])
    print("")
    print("Attn input memory config: ", model_config["SHARDED_ATTN_INPUT_MEMCFG"])

    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))
    torch_tensor = torch.zeros(1, 1, tile_size, dim)
    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        ttnn_tensor = ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)

    compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        gamma_random = ttnn.from_torch(
            torch.randn(1, 1, 1, dim), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
        )

    # Define the norm object : Attn
    print("")
    print("Executing with Attn configs")
    x = ttnn.rms_norm(
        ttnn_tensor,
        epsilon=1e-05,
        weight=gamma_random,
        program_config=model_config["SHARDED_NORM_ATTN_PRGM_CFG"],
        memory_config=model_config["SHARDED_ATTN_INPUT_MEMCFG"],
        compute_kernel_config=compute_kernel_config_hifi2,
    )
    print("Done")
    print("")

    # Define the norm object : MLP
    print("")
    print("Executing with MLP configs")
    x = ttnn.rms_norm(
        ttnn_tensor,
        epsilon=1e-05,
        weight=gamma_random,
        program_config=model_config["SHARDED_NORM_MLP_PRGM_CFG"],
        memory_config=model_config["SHARDED_MLP_INPUT_MEMCFG"],
        compute_kernel_config=compute_kernel_config_hifi2,
    )
    print("Done")
    print("")
    ttnn.close_mesh_device(mesh_device)
