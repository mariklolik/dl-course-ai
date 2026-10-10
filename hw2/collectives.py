import torch
import torch.distributed as dist


def ring_reduce_scatter(tensor):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if tensor.device.type != "cpu" or tensor.ndim != 1 or tensor.numel() == 0:
        raise ValueError("Expected a nonempty one-dimensional CPU tensor")
    if tensor.numel() % world_size:
        raise ValueError("Tensor length must be divisible by world size")
    chunks = [chunk.clone() for chunk in tensor.chunk(world_size)]
    for step in range(world_size - 1):
        send_index = (rank - step - 1) % world_size
        receive_index = (rank - step - 2) % world_size
        received = torch.empty_like(chunks[receive_index])
        send = dist.isend(chunks[send_index], (rank + 1) % world_size)
        receive = dist.irecv(received, (rank - 1) % world_size)
        receive.wait()
        send.wait()
        chunks[receive_index].add_(received)
    return chunks[rank]


def ring_all_gather(shard):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if shard.device.type != "cpu" or shard.ndim != 1 or shard.numel() == 0:
        raise ValueError("Expected a nonempty one-dimensional CPU tensor")
    chunks = [torch.empty_like(shard) for _ in range(world_size)]
    chunks[rank].copy_(shard)
    for step in range(world_size - 1):
        send_index = (rank - step) % world_size
        receive_index = (rank - step - 1) % world_size
        send = dist.isend(chunks[send_index], (rank + 1) % world_size)
        receive = dist.irecv(chunks[receive_index], (rank - 1) % world_size)
        receive.wait()
        send.wait()
    return torch.cat(chunks)


def ring_all_reduce(tensor):
    return ring_all_gather(ring_reduce_scatter(tensor))
