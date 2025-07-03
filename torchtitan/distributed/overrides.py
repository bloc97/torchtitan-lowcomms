from itertools import chain
from typing import Callable, cast, NamedTuple, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.distributed_c10d import ReduceOp
from torch.distributed.tensor import DTensor

import torch.distributed.fsdp

from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    _div_if_needed,
    foreach_reduce_scatter_copy_in,
)

from torch.distributed.fsdp._fully_shard._fsdp_common import (
    _get_dim0_padded_size,
    _raise_assert_with_print,
    _to_dtype_if_needed,
    compiled_autograd_enabled,
)
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam, ShardedState

# Overrides for and tested on
# torch.__version__ = '2.7.1+cu128'
# https://github.com/pytorch/pytorch/blob/f16053f0c9a09fa337fbf85aaf64f88712b8dcdb/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py#L376
# https://github.com/pytorch/pytorch/releases/tag/v2.7.1
# Other torch versions will not work correctly!

def foreach_reduce(
    fsdp_params: list[FSDPParam],
    unsharded_grads: list[torch.Tensor],
    reduce_scatter_group: dist.ProcessGroup,
    reduce_scatter_stream: torch.Stream,
    orig_dtype: Optional[torch.dtype],
    reduce_dtype: Optional[torch.dtype],
    device: torch.device,
    gradient_divide_factor: Optional[float],
    all_reduce_group: Optional[dist.ProcessGroup],  # not `None` iff HSDP
    all_reduce_stream: torch.Stream,
    all_reduce_grads: bool,
    partial_reduce_output: Optional[torch.Tensor],  # only used for HSDP
    all_reduce_hook: Optional[Callable[[torch.Tensor], None]],
    allocate_memory_from_process_group: bool = False,
    force_sum_reduction_for_comms: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Event,
    torch.Event,
    Optional[torch.Tensor],
    Optional[torch.Event],
    Optional[torch.Tensor],
]:
    """
    ``unsharded_grads`` owns the references to the gradients computed by
    autograd, so clearing the list frees the gradients.
    """
    grad_dtypes = {grad.dtype for grad in unsharded_grads}
    if len(grad_dtypes) != 1:
        # Check this at runtime since it could be a real runtime error if e.g.
        # fp8 weights do not produce the correct higher precision gradients
        _raise_assert_with_print(
            f"FSDP reduce-scatter expects uniform gradient dtype but got {grad_dtypes}"
        )
    grad_dtype = unsharded_grads[0].dtype
    reduce_dtype = reduce_dtype or grad_dtype
    (predivide_factor, postdivide_factor, reduce_scatter_op, all_reduce_op) = (
        _get_gradient_divide_factors(
            reduce_scatter_group,
            all_reduce_group,
            reduce_dtype,
            device.type,
            gradient_divide_factor,
            force_sum_reduction_for_comms,
            all_reduce_grads,
        )
    )
    world_size = reduce_scatter_group.size()
    for i, (fsdp_param, unsharded_grad) in enumerate(zip(fsdp_params, unsharded_grads)):
        if (shard_dim := fsdp_param.fsdp_placement.dim) == 0:
            continue
        assert unsharded_grad.size(shard_dim) % world_size == 0, (
            f"Shard({shard_dim}) requires even sharding: {unsharded_grad.size()=} {world_size=}"
        )
        chunks = torch.chunk(unsharded_grad, world_size, dim=shard_dim)
        unsharded_grads[i] = torch.cat(chunks, dim=0)
    padded_unsharded_sizes = tuple(
        _get_dim0_padded_size(grad.size(), world_size) for grad in unsharded_grads
    )
    reduce_scatter_input_numel = sum(s.numel() for s in padded_unsharded_sizes)
    reduce_scatter_output_numel = reduce_scatter_input_numel // world_size
    reduce_scatter_input = allocate_memory(
        reduce_scatter_input_numel,
        dtype=reduce_dtype,
        device=device,
        group=reduce_scatter_group,
        from_process_group=allocate_memory_from_process_group,
    )
    device_handle = _get_device_handle(device.type)
    foreach_reduce_scatter_copy_in(unsharded_grads, reduce_scatter_input, world_size)
    current_stream = device_handle.current_stream()
    # Only after the copy-in finishes can we free the gradients
    unsharded_grads.clear()
    reduce_scatter_stream.wait_stream(current_stream)
    all_reduce_input = None
    all_reduce_event = None
    with device_handle.stream(reduce_scatter_stream):
        reduce_output = allocate_memory(
            reduce_scatter_output_numel,
            dtype=reduce_dtype,
            device=device,
            group=reduce_scatter_group,
            from_process_group=allocate_memory_from_process_group,
        )
        _div_if_needed(reduce_scatter_input, predivide_factor)
        dist.reduce_scatter_tensor(
            output=reduce_output,
            input=reduce_scatter_input,
            group=reduce_scatter_group,
            op=reduce_scatter_op,
        )
        reduce_scatter_event = reduce_scatter_stream.record_event()
        post_reduce_stream = reduce_scatter_stream
        if all_reduce_group is not None:  # HSDP
            # Accumulations must run in the reduce-scatter stream
            if partial_reduce_output is not None:
                reduce_output += partial_reduce_output
            post_reduce_stream = all_reduce_stream
            all_reduce_stream.wait_stream(reduce_scatter_stream)
            with device_handle.stream(all_reduce_stream):
                if all_reduce_grads:
                    dist.all_reduce(
                        reduce_output,
                        group=all_reduce_group,
                        op=all_reduce_op,
                    )
                all_reduce_input = reduce_output
                all_reduce_event = all_reduce_stream.record_event()
    # -- END: ops in reduce_scatter stream

    if all_reduce_hook is not None:
        # Execute user-specified all reduce hook.
        # If native HSDP is used, this is executed after the HSDP all reduce.
        # If 1-d FSDP is used, this is executed post reduce-scatter.
        post_reduce_stream = all_reduce_stream
        all_reduce_stream.wait_stream(reduce_scatter_stream)
        with device_handle.stream(all_reduce_stream):
            all_reduce_hook(reduce_output)
    # -- END: ops post reduce_scatter

    with device_handle.stream(post_reduce_stream):
        _div_if_needed(reduce_output, postdivide_factor)
        reduce_output = _to_dtype_if_needed(reduce_output, orig_dtype)
        # View out and accumulate sharded gradients
        flat_grad_offset = 0  # [0, reduce_scatter_output_numel - 1]
        for padded_unsharded_size, fsdp_param in zip(
            padded_unsharded_sizes, fsdp_params
        ):
            # Assume even sharding for Shard(i), i > 0; otherwise would require
            # copy-out for contiguous strides
            new_sharded_grad = torch.as_strided(
                reduce_output,
                size=fsdp_param.sharded_size,
                stride=fsdp_param.contiguous_sharded_stride,
                storage_offset=flat_grad_offset,
            )
            to_accumulate_grad = fsdp_param.sharded_param.grad is not None
            if fsdp_param.offload_to_cpu:
                # Only overlap the D2H copy (copying to pinned memory) if not
                # accumulating gradients since the CPU add kernel depends on
                # the copy result and we cannot run the add as a callback
                non_blocking = fsdp_param.pin_memory and not to_accumulate_grad
                # Since the GPU sharded gradient is allocated in the RS stream,
                # we can free it here by not keeping a ref without waiting for
                # the D2H copy since future RS-stream ops run after the copy
                new_sharded_grad = new_sharded_grad.to(
                    torch.device("cpu"), non_blocking=non_blocking
                )
                if non_blocking:
                    # Record an event on which to block the CPU thread to
                    # ensure that the D2H copy finishes before the optimizer
                    fsdp_param.grad_offload_event = reduce_scatter_stream.record_event()
            if to_accumulate_grad:
                assert isinstance(fsdp_param.sharded_param.grad, DTensor)
                fsdp_param.sharded_param.grad._local_tensor += new_sharded_grad
            else:
                new_sharded_dtensor_grad = fsdp_param.to_sharded_dtensor(
                    new_sharded_grad
                )
                fsdp_param.sharded_param.grad = new_sharded_dtensor_grad
            if not compiled_autograd_enabled():
                for hook in (
                    getattr(fsdp_param.sharded_param, "_post_accumulate_grad_hooks", {})
                    or {}
                ).values():
                    hook(fsdp_param.sharded_param)
            padded_sharded_numel = padded_unsharded_size.numel() // world_size
            flat_grad_offset += padded_sharded_numel
        post_reduce_event = post_reduce_stream.record_event()
    # The RS output is allocated in the RS stream and used in the default
    # stream (for optimizer). To ensure its memory is not reused for later
    # RSs, we do not need extra synchronization since the sharded parameters
    # hold refs through the end of backward.
    return (
        reduce_scatter_input,
        reduce_scatter_event,
        post_reduce_event,
        all_reduce_input,
        all_reduce_event,
        None,
    )

def allocate_memory(
    size: int,
    dtype: torch.dtype,
    device: torch.device,
    group: dist.ProcessGroup,
    from_process_group: bool,
) -> torch.Tensor:
    if from_process_group:
        backend = group._get_backend(device)
        if backend.supports_tensor_alloc(device):
            return backend.allocate_tensor(size, dtype=dtype, device=device)
    return torch.empty((size,), dtype=dtype, device=device)

def _get_gradient_divide_factors(
    reduce_scatter_group: dist.ProcessGroup,
    all_reduce_group: Optional[dist.ProcessGroup],
    reduce_dtype: torch.dtype,
    device_type: str = "",
    factor: Optional[float] = None,
    force_sum_reduction_for_comms: bool = False,
    all_reduce_grads: bool = True,
) -> tuple[
    Optional[float],
    Optional[float],
    Union[dist.ReduceOp, dist.ReduceOp.RedOpType],
    Union[dist.ReduceOp, dist.ReduceOp.RedOpType],
]:
    # MTIA appears to only support SUM reduction, hence we force it implicitly
    if device_type == "mtia":
        force_sum_reduction_for_comms = True

    # For fp32/bf16, we do not need to worry about overflow/underflow, so we
    # use NCCL's built-in division to avoid separate div kernels
    overflow_risk = reduce_dtype not in (torch.float32, torch.bfloat16)

    data_parallel_size = reduce_scatter_group.size()
    if all_reduce_group is not None and all_reduce_grads:
        data_parallel_size *= all_reduce_group.size()

    if factor is None:
        factor = float(data_parallel_size)

    if not overflow_risk and not force_sum_reduction_for_comms:
        if factor == data_parallel_size:
            # Warning: NCCL ReduceOp.AVG may produce incorrect results with
            # world size 1.
            return None, None, ReduceOp.AVG, ReduceOp.AVG
        else:
            reduce_scatter_op = torch.distributed._make_nccl_premul_sum(1 / factor)
            return None, None, reduce_scatter_op, ReduceOp.SUM

    pre_factor: Optional[float]
    if overflow_risk:
        # Since fp16 has smaller dynamic range than fp32/bf16, we want to avoid
        # overflow/underflow. For N data parallel workers, each worker computes
        # g_i, and they collectively reduce (g_1 + ... + g_N) / N. To avoid
        # overflow/underflow, we divide by ~sqrt(N) before/after the reduction.
        pre_factor = 1
        while factor % pre_factor == 0 and factor / pre_factor > pre_factor:
            pre_factor *= 2
        post_factor = factor / pre_factor
    else:
        # Prefer post-multiplying as it operates on less data and is thus faster
        pre_factor, post_factor = None, factor

    return pre_factor, post_factor, ReduceOp.SUM, ReduceOp.SUM

# @torch.no_grad()
# def foreach_reduce_custom(
#     fsdp_params: list[FSDPParam],
#     unsharded_grads: list[torch.Tensor],
#     reduce_scatter_group: dist.ProcessGroup,
#     reduce_scatter_stream: torch.Stream,
#     orig_dtype: Optional[torch.dtype],
#     reduce_dtype: Optional[torch.dtype],
#     device: torch.device,
#     reduce_scatter_reduce_op: Optional[Union[dist.ReduceOp, dist.ReduceOp.RedOpType]],
#     all_reduce_group: Optional[dist.ProcessGroup],  # not `None` iff HSDP
#     all_reduce_stream: torch.Stream,
#     all_reduce_grads: bool,
#     partial_reduce_output: Optional[torch.Tensor],  # only used for HSDP
#     all_reduce_hook: Optional[Callable[[torch.Tensor], None]],
# ) -> tuple[
#     torch.Tensor,
#     torch.Event,
#     torch.Event,
#     Optional[torch.Tensor],
#     Optional[torch.Event],
#     Optional[torch.Tensor],
# ]:
#     """
#     ``unsharded_grads`` owns the references to the gradients computed by
#     autograd, so clearing the list frees the gradients.
#     """
#     grad_dtypes = {grad.dtype for grad in unsharded_grads}
#     if len(grad_dtypes) != 1:
#         # Check this at runtime since it could be a real runtime error if e.g.
#         # fp8 weights do not produce the correct higher precision gradients
#         _raise_assert_with_print(
#             f"FSDP reduce-scatter expects uniform gradient dtype but got {grad_dtypes}"
#         )
#     grad_dtype = unsharded_grads[0].dtype
#     reduce_dtype = reduce_dtype or grad_dtype
#     predivide_factor, postdivide_factor = _get_gradient_divide_factors(
#         reduce_scatter_group, all_reduce_group, reduce_dtype, device.type, all_reduce_grads
#     )
#     world_size = reduce_scatter_group.size()
#     for i, (fsdp_param, unsharded_grad) in enumerate(zip(fsdp_params, unsharded_grads)):
#         if (shard_dim := fsdp_param.fsdp_placement.dim) == 0:
#             continue
#         assert unsharded_grad.size(shard_dim) % world_size == 0, (
#             f"Shard({shard_dim}) requires even sharding: {unsharded_grad.size()=} {world_size=}"
#         )
#         chunks = torch.chunk(unsharded_grad, world_size, dim=shard_dim)
#         unsharded_grads[i] = torch.cat(chunks, dim=0)
#     padded_unsharded_sizes = tuple(
#         _get_dim0_padded_size(grad.size(), world_size) for grad in unsharded_grads
#     )
#     reduce_scatter_input_numel = sum(s.numel() for s in padded_unsharded_sizes)
#     reduce_scatter_output_numel = reduce_scatter_input_numel // world_size
#     reduce_scatter_input = torch.empty(
#         (reduce_scatter_input_numel,), dtype=reduce_dtype, device=device
#     )
#     device_handle = _get_device_handle(device.type)
#     foreach_reduce_scatter_copy_in(unsharded_grads, reduce_scatter_input, world_size)
#     current_stream = device_handle.current_stream()
#     # Only after the copy-in finishes can we free the gradients
#     unsharded_grads.clear()
#     reduce_scatter_stream.wait_stream(current_stream)
#     all_reduce_input = None
#     all_reduce_event = None
#     with device_handle.stream(reduce_scatter_stream):
#         reduce_output = reduce_scatter_input.new_empty((reduce_scatter_output_numel,))
#         _div_if_needed(reduce_scatter_input, predivide_factor)
#         if reduce_scatter_reduce_op is None:
#             if predivide_factor is None:
#                 reduce_scatter_reduce_op = ReduceOp.AVG
#             else:
#                 reduce_scatter_reduce_op = ReduceOp.SUM
#         dist.reduce_scatter_tensor(
#             output=reduce_output,
#             input=reduce_scatter_input,
#             group=reduce_scatter_group,
#             op=reduce_scatter_reduce_op,
#         )
#         reduce_scatter_event = reduce_scatter_stream.record_event()
#         post_reduce_stream = reduce_scatter_stream
#         if all_reduce_group is not None:  # HSDP
#             if partial_reduce_output is not None:
#                 reduce_output += partial_reduce_output
#             post_reduce_stream = all_reduce_stream
#             all_reduce_stream.wait_stream(reduce_scatter_stream)
#             with device_handle.stream(all_reduce_stream):
#                 if all_reduce_grads:
#                     dist.all_reduce(
#                         reduce_output,
#                         group=all_reduce_group,
#                         op=ReduceOp.AVG if predivide_factor is None else ReduceOp.SUM,
#                     )
#                     all_reduce_input = reduce_output
#                     all_reduce_event = all_reduce_stream.record_event()
#     # -- END: ops in reduce_scatter stream

#     if all_reduce_hook is not None:
#         # Execute user-specified all reduce hook.
#         # If native HSDP is used, this is executed after the HSDP all reduce.
#         # If 1-d FSDP is used, this is executed post reduce-scatter.
#         post_reduce_stream = all_reduce_stream
#         all_reduce_stream.wait_stream(reduce_scatter_stream)
#         with device_handle.stream(all_reduce_stream):
#             all_reduce_hook(reduce_output)
#     # -- END: ops post reduce_scatter

#     with device_handle.stream(post_reduce_stream):
#         _div_if_needed(reduce_output, postdivide_factor)
#         reduce_output = _to_dtype_if_needed(reduce_output, orig_dtype)
#         # View out and accumulate sharded gradients
#         flat_grad_offset = 0  # [0, reduce_scatter_output_numel - 1]
#         for padded_unsharded_size, fsdp_param in zip(
#             padded_unsharded_sizes, fsdp_params
#         ):
#             # Assume even sharding for Shard(i), i > 0; otherwise would require
#             # copy-out for contiguous strides
#             new_sharded_grad = torch.as_strided(
#                 reduce_output,
#                 size=fsdp_param.sharded_size,
#                 stride=fsdp_param.contiguous_sharded_stride,
#                 storage_offset=flat_grad_offset,
#             )
#             to_accumulate_grad = fsdp_param.sharded_param.grad is not None
#             if fsdp_param.offload_to_cpu:
#                 # Only overlap the D2H copy (copying to pinned memory) if not
#                 # accumulating gradients since the CPU add kernel depends on
#                 # the copy result and we cannot run the add as a callback
#                 non_blocking = fsdp_param.pin_memory and not to_accumulate_grad
#                 # Since the GPU sharded gradient is allocated in the RS stream,
#                 # we can free it here by not keeping a ref without waiting for
#                 # the D2H copy since future RS-stream ops run after the copy
#                 new_sharded_grad = new_sharded_grad.to(
#                     torch.device("cpu"), non_blocking=non_blocking
#                 )
#                 if non_blocking:
#                     # Record an event on which to block the CPU thread to
#                     # ensure that the D2H copy finishes before the optimizer
#                     fsdp_param.grad_offload_event = reduce_scatter_stream.record_event()
#             if to_accumulate_grad:
#                 assert isinstance(fsdp_param.sharded_param.grad, DTensor)
#                 fsdp_param.sharded_param.grad._local_tensor += new_sharded_grad
#             else:
#                 new_sharded_dtensor_grad = fsdp_param.to_sharded_dtensor(
#                     new_sharded_grad
#                 )
#                 fsdp_param.sharded_param.grad = new_sharded_dtensor_grad
#             if not compiled_autograd_enabled():
#                 for hook in (
#                     getattr(fsdp_param.sharded_param, "_post_accumulate_grad_hooks", {})
#                     or {}
#                 ).values():
#                     hook(fsdp_param.sharded_param)
#             padded_sharded_numel = padded_unsharded_size.numel() // world_size
#             flat_grad_offset += padded_sharded_numel
#         post_reduce_event = post_reduce_stream.record_event()
#     # The RS output is allocated in the RS stream and used in the default
#     # stream (for optimizer). To ensure its memory is not reused for later
#     # RSs, we do not need extra synchronization since the sharded parameters
#     # hold refs through the end of backward.
#     return (
#         reduce_scatter_input,
#         reduce_scatter_event,
#         post_reduce_event,
#         all_reduce_input,
#         all_reduce_event,
#         None,
#     )
    
# def _get_gradient_divide_factors(
#     reduce_scatter_group: dist.ProcessGroup,
#     all_reduce_group: Optional[dist.ProcessGroup],
#     reduce_dtype: torch.dtype,
#     device_type: str = "",
#     all_reduce_grads: bool = True,
# ) -> Union[tuple[None, None], tuple[float, float]]:
#     # For fp32/bf16, we do not need to worry about overflow/underflow, so we
#     # use NCCL's built-in division to avoid separate div kernels
#     if reduce_dtype in (torch.float32, torch.bfloat16) and device_type != "mtia":
#         return None, None
#     data_parallel_size = reduce_scatter_group.size()
#     if all_reduce_group is not None and all_reduce_grads:
#         data_parallel_size *= all_reduce_group.size()
#     # Since fp16 has smaller dynamic range than fp32/bf16, we want to avoid
#     # overflow/underflow. For N data parallel workers, each worker computes
#     # g_i, and they collectively reduce (g_1 + ... + g_N) / N. To avoid
#     # overflow/underflow, we divide by ~sqrt(N) before/after the reduction.
#     factor: int = 1
#     while data_parallel_size % factor == 0 and data_parallel_size / factor > factor:
#         factor *= 2
#     factor = float(factor)
#     return (factor, data_parallel_size / factor)


# from itertools import chain
# from typing import Callable, cast, NamedTuple, Optional, Union

# import torch
# import torch.distributed as dist
# from torch.distributed.device_mesh import _get_device_handle
# from torch.distributed.distributed_c10d import ReduceOp
# from torch.distributed.tensor import DTensor

# import torch.distributed.fsdp

# from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
#     _div_if_needed,
#     foreach_reduce_scatter_copy_in,
#     allocate_memory,
# )

# from torch.distributed.fsdp._fully_shard._fsdp_common import (
#     _get_dim0_padded_size,
#     _raise_assert_with_print,
#     _to_dtype_if_needed,
#     compiled_autograd_enabled,
# )
# from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam, ShardedState

# @torch.no_grad()
# def foreach_reduce_custom(
#     fsdp_params: list[FSDPParam],
#     unsharded_grads: list[torch.Tensor],
#     reduce_scatter_group: dist.ProcessGroup,
#     reduce_scatter_stream: torch.Stream,
#     orig_dtype: Optional[torch.dtype],
#     reduce_dtype: Optional[torch.dtype],
#     device: torch.device,
#     gradient_divide_factor: Optional[float],
#     all_reduce_group: Optional[dist.ProcessGroup],  # not `None` iff HSDP
#     all_reduce_stream: torch.Stream,
#     all_reduce_grads: bool,
#     partial_reduce_output: Optional[torch.Tensor],  # only used for HSDP
#     all_reduce_hook: Optional[Callable[[torch.Tensor], None]],
#     allocate_memory_from_process_group: bool = False,
#     force_sum_reduction_for_comms: bool = False,
# ) -> tuple[
#     torch.Tensor,
#     torch.Event,
#     torch.Event,
#     Optional[torch.Tensor],
#     Optional[torch.Event],
#     Optional[torch.Tensor],
# ]:
#     """
#     ``unsharded_grads`` owns the references to the gradients computed by
#     autograd, so clearing the list frees the gradients.
#     """
#     grad_dtypes = {grad.dtype for grad in unsharded_grads}
#     if len(grad_dtypes) != 1:
#         # Check this at runtime since it could be a real runtime error if e.g.
#         # fp8 weights do not produce the correct higher precision gradients
#         _raise_assert_with_print(
#             f"FSDP reduce-scatter expects uniform gradient dtype but got {grad_dtypes}"
#         )
#     grad_dtype = unsharded_grads[0].dtype
#     reduce_dtype = reduce_dtype or grad_dtype
#     (predivide_factor, postdivide_factor, reduce_scatter_op, all_reduce_op) = (
#         _get_gradient_divide_factors(
#             reduce_scatter_group,
#             all_reduce_group,
#             reduce_dtype,
#             device.type,
#             all_reduce_grads,
#             gradient_divide_factor,
#             force_sum_reduction_for_comms,
#         )
#     )
#     world_size = reduce_scatter_group.size()
#     for i, (fsdp_param, unsharded_grad) in enumerate(zip(fsdp_params, unsharded_grads)):
#         if (shard_dim := fsdp_param.fsdp_placement.dim) == 0:
#             continue
#         assert unsharded_grad.size(shard_dim) % world_size == 0, (
#             f"Shard({shard_dim}) requires even sharding: {unsharded_grad.size()=} {world_size=}"
#         )
#         chunks = torch.chunk(unsharded_grad, world_size, dim=shard_dim)
#         unsharded_grads[i] = torch.cat(chunks, dim=0)
#     padded_unsharded_sizes = tuple(
#         _get_dim0_padded_size(grad.size(), world_size) for grad in unsharded_grads
#     )
#     reduce_scatter_input_numel = sum(s.numel() for s in padded_unsharded_sizes)
#     reduce_scatter_output_numel = reduce_scatter_input_numel // world_size
#     reduce_scatter_input = allocate_memory(
#         reduce_scatter_input_numel,
#         dtype=reduce_dtype,
#         device=device,
#         group=reduce_scatter_group,
#         from_process_group=allocate_memory_from_process_group,
#     )
#     device_handle = _get_device_handle(device.type)
#     foreach_reduce_scatter_copy_in(unsharded_grads, reduce_scatter_input, world_size)
#     current_stream = device_handle.current_stream()
#     # Only after the copy-in finishes can we free the gradients
#     unsharded_grads.clear()
#     reduce_scatter_stream.wait_stream(current_stream)
#     all_reduce_input = None
#     all_reduce_event = None
#     with device_handle.stream(reduce_scatter_stream):
#         reduce_output = allocate_memory(
#             reduce_scatter_output_numel,
#             dtype=reduce_dtype,
#             device=device,
#             group=reduce_scatter_group,
#             from_process_group=allocate_memory_from_process_group,
#         )
#         _div_if_needed(reduce_scatter_input, predivide_factor)
#         dist.reduce_scatter_tensor(
#             output=reduce_output,
#             input=reduce_scatter_input,
#             group=reduce_scatter_group,
#             op=reduce_scatter_op,
#         )
#         reduce_scatter_event = reduce_scatter_stream.record_event()
#         post_reduce_stream = reduce_scatter_stream
#         if all_reduce_group is not None:  # HSDP
#             # Accumulations must run in the reduce-scatter stream
#             if partial_reduce_output is not None:
#                 reduce_output += partial_reduce_output
#             post_reduce_stream = all_reduce_stream
#             all_reduce_stream.wait_stream(reduce_scatter_stream)
#             with device_handle.stream(all_reduce_stream):
#                 if all_reduce_grads:
#                     dist.all_reduce(
#                         reduce_output,
#                         group=all_reduce_group,
#                         op=all_reduce_op,
#                     )
#                 all_reduce_input = reduce_output
#                 all_reduce_event = all_reduce_stream.record_event()
#     # -- END: ops in reduce_scatter stream

#     if all_reduce_hook is not None:
#         # Execute user-specified all reduce hook.
#         # If native HSDP is used, this is executed after the HSDP all reduce.
#         # If 1-d FSDP is used, this is executed post reduce-scatter.
#         post_reduce_stream = all_reduce_stream
#         all_reduce_stream.wait_stream(reduce_scatter_stream)
#         with device_handle.stream(all_reduce_stream):
#             all_reduce_hook(reduce_output)
#     # -- END: ops post reduce_scatter

#     with device_handle.stream(post_reduce_stream):
#         _div_if_needed(reduce_output, postdivide_factor)
#         reduce_output = _to_dtype_if_needed(reduce_output, orig_dtype)
#         # View out and accumulate sharded gradients
#         flat_grad_offset = 0  # [0, reduce_scatter_output_numel - 1]
#         for padded_unsharded_size, fsdp_param in zip(
#             padded_unsharded_sizes, fsdp_params
#         ):
#             # Assume even sharding for Shard(i), i > 0; otherwise would require
#             # copy-out for contiguous strides
#             new_sharded_grad = torch.as_strided(
#                 reduce_output,
#                 size=fsdp_param.sharded_size,
#                 stride=fsdp_param.contiguous_sharded_stride,
#                 storage_offset=flat_grad_offset,
#             )
#             to_accumulate_grad = fsdp_param.sharded_param.grad is not None
#             if fsdp_param.offload_to_cpu:
#                 # Only overlap the D2H copy (copying to pinned memory) if not
#                 # accumulating gradients since the CPU add kernel depends on
#                 # the copy result and we cannot run the add as a callback
#                 non_blocking = fsdp_param.pin_memory and not to_accumulate_grad
#                 # Since the GPU sharded gradient is allocated in the RS stream,
#                 # we can free it here by not keeping a ref without waiting for
#                 # the D2H copy since future RS-stream ops run after the copy
#                 new_sharded_grad = new_sharded_grad.to(
#                     torch.device("cpu"), non_blocking=non_blocking
#                 )
#                 if non_blocking:
#                     # Record an event on which to block the CPU thread to
#                     # ensure that the D2H copy finishes before the optimizer
#                     fsdp_param.grad_offload_event = reduce_scatter_stream.record_event()
#             if to_accumulate_grad:
#                 assert isinstance(fsdp_param.sharded_param.grad, DTensor)
#                 fsdp_param.sharded_param.grad._local_tensor += new_sharded_grad
#             else:
#                 new_sharded_dtensor_grad = fsdp_param.to_sharded_dtensor(
#                     new_sharded_grad
#                 )
#                 fsdp_param.sharded_param.grad = new_sharded_dtensor_grad
#             if not compiled_autograd_enabled():
#                 for hook in (
#                     getattr(fsdp_param.sharded_param, "_post_accumulate_grad_hooks", {})
#                     or {}
#                 ).values():
#                     hook(fsdp_param.sharded_param)
#             padded_sharded_numel = padded_unsharded_size.numel() // world_size
#             flat_grad_offset += padded_sharded_numel
#         post_reduce_event = post_reduce_stream.record_event()
#     # The RS output is allocated in the RS stream and used in the default
#     # stream (for optimizer). To ensure its memory is not reused for later
#     # RSs, we do not need extra synchronization since the sharded parameters
#     # hold refs through the end of backward.
#     return (
#         reduce_scatter_input,
#         reduce_scatter_event,
#         post_reduce_event,
#         all_reduce_input,
#         all_reduce_event,
#         None,
#     )

# def _get_gradient_divide_factors(
#     reduce_scatter_group: dist.ProcessGroup,
#     all_reduce_group: Optional[dist.ProcessGroup],
#     reduce_dtype: torch.dtype,
#     device_type: str = "",
#     all_reduce_grads: bool = True,
#     factor: Optional[float] = None,
#     force_sum_reduction_for_comms: bool = False,
# ) -> tuple[
#     Optional[float],
#     Optional[float],
#     Union[dist.ReduceOp, dist.ReduceOp.RedOpType],
#     Union[dist.ReduceOp, dist.ReduceOp.RedOpType],
# ]:
#     # MTIA appears to only support SUM reduction, hence we force it implicitly
#     if device_type == "mtia":
#         force_sum_reduction_for_comms = True

#     # For fp32/bf16, we do not need to worry about overflow/underflow, so we
#     # use NCCL's built-in division to avoid separate div kernels
#     overflow_risk = reduce_dtype not in (torch.float32, torch.bfloat16)

#     data_parallel_size = reduce_scatter_group.size()
#     if all_reduce_group is not None and all_reduce_grads:
#         data_parallel_size *= all_reduce_group.size()

#     if factor is None:
#         factor = float(data_parallel_size)

#     if not overflow_risk and not force_sum_reduction_for_comms:
#         if factor == data_parallel_size:
#             # Warning: NCCL ReduceOp.AVG may produce incorrect results with
#             # world size 1.
#             return None, None, ReduceOp.AVG, ReduceOp.AVG
#         else:
#             reduce_scatter_op = torch.distributed._make_nccl_premul_sum(1 / factor)
#             return None, None, reduce_scatter_op, ReduceOp.SUM

#     pre_factor: Optional[float]
#     if overflow_risk:
#         # Since fp16 has smaller dynamic range than fp32/bf16, we want to avoid
#         # overflow/underflow. For N data parallel workers, each worker computes
#         # g_i, and they collectively reduce (g_1 + ... + g_N) / N. To avoid
#         # overflow/underflow, we divide by ~sqrt(N) before/after the reduction.
#         pre_factor = 1
#         while factor % pre_factor == 0 and factor / pre_factor > pre_factor:
#             pre_factor *= 2
#         post_factor = factor / pre_factor
#     else:
#         # Prefer post-multiplying as it operates on less data and is thus faster
#         pre_factor, post_factor = None, factor

#     return pre_factor, post_factor, ReduceOp.SUM, ReduceOp.SUM