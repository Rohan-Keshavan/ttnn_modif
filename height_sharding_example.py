import torch
import ttnn
import math


def create_dram_sharded_mem_config(k, n, mesh_device):
    """Create DRAM-sharded memory config for width-sharded tensors"""
    dram_cores = mesh_device.dram_grid_size().x  # WH has 12 dram cores, P150 has 8, P100 has 7
    assert mesh_device.dram_grid_size().y == 1, "Current dram sharding assumes y dim is 1"
    padded_size = math.ceil(n / (32 * dram_cores)) * (32 * dram_cores)

    dram_weight_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(mesh_device.dram_grid_size().x - 1, mesh_device.dram_grid_size().y - 1),
            )
        }
    )

    shard_spec = ttnn.ShardSpec(dram_weight_grid, (k, padded_size // dram_cores), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


if __name__ == "__main__":
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))
    wo_mem_config = create_dram_sharded_mem_config(n=(64 * 128) // 8, k=8192, mesh_device=mesh_device)
    torch_tensor = torch.rand((1, 1, 8192, 8192), dtype=torch.bfloat16)
    wo = ttnn.as_tensor(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=wo_mem_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(2, 3),
            mesh_shape=list(mesh_device.shape),
        ),
    )
    print(wo.shape, wo.layout, wo.memory_config)
    ttnn.visualize_mesh_device(mesh_device, tensor=wo)
    ttnn.close_mesh_device(mesh_device)
