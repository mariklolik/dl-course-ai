from datetime import timedelta

import torch
import torch.distributed as dist

from collectives import ring_all_gather, ring_all_reduce, ring_reduce_scatter

torch.set_num_threads(1)
dist.init_process_group("gloo", timeout=timedelta(seconds=30))
try:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    for dtype in (torch.float32, torch.float64, torch.int64):
        values = torch.arange(world_size * 10, dtype=dtype).add_(rank)[::2]
        original = values.clone()
        expected = values.clone()
        dist.all_reduce(expected)
        torch.testing.assert_close(ring_reduce_scatter(values), expected.chunk(world_size)[rank])
        torch.testing.assert_close(ring_all_reduce(values), expected)
        torch.testing.assert_close(values, original, rtol=0, atol=0)
        shard = torch.arange(10, dtype=dtype).add_(rank)[::2]
        original_shard = shard.clone()
        expected_parts = [torch.empty_like(shard) for _ in range(world_size)]
        dist.all_gather(expected_parts, shard)
        torch.testing.assert_close(ring_all_gather(shard), torch.cat(expected_parts))
        torch.testing.assert_close(shard, original_shard, rtol=0, atol=0)
    for operation in (ring_reduce_scatter, ring_all_reduce, ring_all_gather):
        for invalid in (torch.empty(0), torch.ones(2, 2)):
            try:
                operation(invalid)
            except ValueError:
                pass
            else:
                raise AssertionError("Expected invalid tensor to raise ValueError")
    if world_size > 1:
        try:
            ring_reduce_scatter(torch.ones(world_size + 1))
        except ValueError:
            pass
        else:
            raise AssertionError("Expected uneven shards to raise ValueError")
    if rank == 0:
        print(f"OK: edge checks passed on {world_size} processes.")
finally:
    dist.destroy_process_group()
