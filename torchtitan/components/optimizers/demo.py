
"""DeMo: Decoupled Momentum Optimization

This implements the DeMo fused optimizer and data parallel algorithm.
It is recommended to use DeMo as the base data parallelism.
In an exisiting codebase that uses PyTorch DDP, wrap your forward-backward in 
`torch.distributed.DistributedDataParallel.no_sync` to disable external gradient synchronization.
See https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.no_sync
"""

import math
import torch
import torch.fft
import torch.distributed as dist

from torch.optim import Optimizer

from einops import rearrange
from typing import Optional, Callable
    
from abc import ABC, abstractmethod
from typing import TypeAlias, Any, Generator, Iterable, Optional
from collections.abc import Callable

Tensor: TypeAlias = torch.Tensor
ProcessGroup: TypeAlias = dist.ProcessGroup

import queue

class TensorCollective(ABC):
    
    @abstractmethod
    def _synchronize_send(self, input: Tensor) -> None:
        # Takes in input Tensor and the ProcessGroup
        # Awaits synchronization (can be async), enqueues in a fifo queue
        pass
    @abstractmethod
    def _synchronize_receive(self) -> Tensor | None:
        # Returns the collected tensor across the group
        # Forces the synchronization to finish, pops from queue
        # For efficiency reasons, this method might collect the input in-place and will return a reference to the input tensor
        pass

class TransformCompressCollective(TensorCollective):
    
    @abstractmethod
    def _encode(self, input: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # Takes in input tensor
        # Returns the encoded tensor and some local metadata
        pass
    
    @abstractmethod
    def _decode(self, input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        # Takes in encoded tensor and the local metadata
        # Returns the decoded tensor
        pass
    
    @abstractmethod
    def _compress(self, input: torch.Tensor, compression_ratio: float) -> tuple[list[Tensor], dict]:
        # Takes in input tensor and the compression ratio
        # Returns the compressed data payload with a local metadata dict
        pass
    
    @abstractmethod
    def _decompress(self, input: list[Tensor], **kwargs: Any) -> torch.Tensor:
        # Takes in the compressed data payload with a local metadata dict
        # Returns the decompressed tensor
        pass
    
    @abstractmethod
    def _collate(self, input: list[torch.Tensor]) -> tuple[torch.Tensor, dict]:
        # Takes the compressed data payload
        # Returns a collated tensor suitable for synchronization and some local metadata
        pass
    
    @abstractmethod
    def _disperse(self, input: torch.Tensor, **kwargs: Any) -> list[torch.Tensor]:
        # Takes the synchronized collated tensor and the local metadata
        # Returns the compressed data payload
        pass
    

class DeMo(Optimizer):
    def __init__(
        self,
        params,
        lr: float | Tensor = 1e-3,
        momentum: float = 0.99,
        compression_ratio: float = 0.01,
        feedback_strength: float = 1.0,
        weight_decay: float = 1e-2,
        nesterov: bool = False,
        overlapped: bool = False,
        collective: TransformCompressCollective = None,
    ):
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if feedback_strength < 0.0:
            raise ValueError(f"Invalid feedback_strength value: {feedback_strength}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
        
        defaults = dict(
            lr=lr,
            momentum=momentum,
            compression_ratio=compression_ratio,
            feedback_strength=feedback_strength,
            weight_decay=weight_decay,
            nesterov=nesterov,
            overlapped=overlapped,
        )
        super().__init__(params, defaults)
        
        self.collective = collective
        self.custom_state = {}

    def _get_custom_state(self, p):
        if p not in self.custom_state:
            self.custom_state[p] = {}
        return self.custom_state[p]
    
    def _init_group(self, group, params, weights, grads, momentum_buffer_list, compressed_buffer_list):
        for p in group["params"]:
            if p.grad is not None:
                if p not in self.custom_state:
                    self.custom_state[p] = {}
                
                params.append(p)
                pweight = p.data._local_tensor if hasattr(p.data, '_local_tensor') else p.data #If DTensor, return the local tensor instead
                weights.append(pweight)
                
                pgrad = p.grad._local_tensor if hasattr(p.grad, '_local_tensor') else p.grad #If DTensor, return the local tensor instead
                grads.append(pgrad)
                
                state = self.state[p]
                custom_state = self._get_custom_state(p)
                
                if group["momentum"] != 0: # If momentum is being used
                    momentum_buffer_list.append(state.get("momentum_buffer"))
                    
                if group["nesterov"] and group["overlapped"]:
                    compressed_buffer_list.append(custom_state.get("compressed_buffer"))
                    
                    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        assert len(self.param_groups) <= 1, f"The current DeMo collective implementation does not support a param_group bigger than 1. Current param group size is {len(self.param_groups)}."

        for group in self.param_groups:
            params: list[Tensor] = []
            grads: list[Tensor] = []
            weights: list[Tensor] = []
            momentum_buffer_list: list[Optional[Tensor]] = []
            compressed_buffer_list: list[Optional[Tensor]] = []

            self._init_group(
                group, params, weights, grads, momentum_buffer_list, compressed_buffer_list
            )
            
            if "last_lr" not in group:
                group["last_lr"] = group["lr"]

            demo(
                weights,
                grads,
                momentum_buffer_list,
                compressed_buffer_list,
                lr=group["lr"],
                momentum=group["momentum"],
                compression_ratio=group["compression_ratio"],
                feedback_strength=group["feedback_strength"],
                weight_decay=group["weight_decay"],
                nesterov=group["nesterov"],
                overlapped=group["overlapped"],
                last_lr=group["last_lr"],
                collective=self.collective,
            )
            
            group["last_lr"] = group["lr"]

            if group["momentum"] != 0:
                # update momentum_buffers in state
                for p, momentum_buffer in zip(params, momentum_buffer_list):
                    state = self.state[p]
                    state["momentum_buffer"] = momentum_buffer
            
            if group["nesterov"] and group["overlapped"]:
                # update compressed_buffer in state
                for p, compressed_buffer in zip(params, compressed_buffer_list):
                    custom_state = self._get_custom_state(p)
                    custom_state["compressed_buffer"] = compressed_buffer

        return loss
    
def demo(
    weights: list[Tensor],
    grads: list[Tensor],
    momentum_buffer_list: list[Optional[Tensor]],
    compressed_buffer_list: list[Optional[Tensor]],
    lr: float,
    momentum: float,
    compression_ratio: float,
    feedback_strength: float,
    weight_decay: float,
    nesterov: bool,
    overlapped: bool,
    last_lr: float,
    collective: TransformCompressCollective | None,
):
    
    all_encoded_meta = []
    all_compressed = []
    all_compressed_meta = []
    all_size = []

    # Optim
    for i, (weight, grad) in enumerate(zip(weights, grads)):
        
        # Encode using collective
        encoded, encoded_meta = collective._encode(grad)
        
        # If nesterov was enabled, undo lookahead
        # TODO: Careful, this is nondeterministic across different CUDA devices! might cause errors to accumulate between nodes!
        if nesterov:
            if momentum_buffer_list[i] is not None:
                decoded = collective._decode(momentum_buffer_list[i], **encoded_meta).sign_()
                weight.add_(decoded, alpha=last_lr)
                
        # Perform delayed feedback
        if overlapped and nesterov and compressed_buffer_list[i] is not None:
            cd, cm = compressed_buffer_list[i]
            momentum_buffer_list[i].add_(collective._decompress(cd, **cm), alpha=-feedback_strength)
        
        # Step-weight decay (also known as decoupled weight decay)
        if weight_decay != 0.0:
            weight.mul_(1.0 - lr * weight_decay)
            
        # Handle momentum
        if momentum != 0:
            if momentum_buffer_list[i] is None:
                momentum_buffer_list[i] = torch.zeros_like(encoded).detach()
            
            momentum_buffer_list[i].mul_(momentum).add_(encoded, alpha=lr)
            encoded = momentum_buffer_list[i]
        
        # Compress
        compressed_data, compressed_meta = collective._compress(encoded, compression_ratio)
        
        # Perform feedback if not overlapped, otherwise delay feedback
        if overlapped and nesterov:
            compressed_buffer_list[i] = (compressed_data, compressed_meta)
        else:
            # Perform feedback if momentum exists, otherwise gradient gets discarded after compression
            if momentum != 0:
                encoded.add_(collective._decompress(compressed_data, **compressed_meta), alpha=-feedback_strength)
        
        # If nesterov was enabled, add in new lookahead
        # TODO: Careful, this is nondeterministic across different CUDA devices! might cause errors to accumulate between nodes!
        if nesterov:
            weight.add_(collective._decode(encoded, **encoded_meta).sign_(), alpha=-lr)
            
        # Save data for collective comms
        for k in range(len(compressed_data)):
            if len(all_compressed) <= k:
                all_compressed.append([])
                all_size.append([])
            
            all_compressed[k].append(compressed_data[k])
            all_size[k].append(compressed_data[k].shape[0])
            
        all_encoded_meta.append(encoded_meta)
        all_compressed_meta.append(compressed_meta)
    
    # Pre-collate
    for i in range(len(all_compressed)):
        all_compressed[i] = torch.concatenate(all_compressed[i], dim=0)
        
    # Collate
    collated_tensor, collated_meta = collective._collate(all_compressed)
    
    # Synchronize
    if overlapped:
        synchronized_tensor = collective._synchronize_receive()
        collective._synchronize_send(collated_tensor)
    else:
        collective._synchronize_send(collated_tensor)
        synchronized_tensor = collective._synchronize_receive()
    
    # Halt if there's no data
    if synchronized_tensor is None:
        return    
    
    # Disperse
    all_synchronized = collective._disperse(synchronized_tensor, **collated_meta)
    
    # Post-disperse
    for i in range(len(all_synchronized)):
        all_synchronized[i] = torch.split(all_synchronized[i], all_size[i], dim=0)
    
    # Update
    for i, (weight, encoded_meta, compressed_meta) in enumerate(zip(weights, all_encoded_meta, all_compressed_meta)):
        
        # Load data from collective comms
        compressed_data = [all_synchronized[k][i] for k in range(len(all_synchronized))]
        
        # Decompress and decode
        new_grad = collective._decode(collective._decompress(compressed_data, **compressed_meta), **encoded_meta)
        
        # Signum
        new_grad.sign_()
        
        # TODO: Test variance reduction method that checks for sign flipping and dampens oscillations
        # Use int8 with probabilistic decay?
        
        # Update params
        weight.add_(new_grad, alpha=-lr)
    
def _pad_to_divisible(tensor, divisor):
    """Pads a 1D tensor to the right to make its size divisible by a divisor."""
    size = tensor.size(0)
    remainder = size % divisor
    if remainder != 0:
        padding_amount = divisor - remainder
        padded_tensor = torch.nn.functional.pad(tensor, (0, padding_amount), mode='constant', value=0)
        return padded_tensor, size
    else:
        return tensor, None


def _build_einsum_dot_str(dim_num):
    chars_left = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm']
    chars_right = ['n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    assert(dim_num <= len(chars_left) and dim_num <= len(chars_right))
    es = "..."
    for i in range(dim_num):
        es += chars_left[i]
    es += ", "
    for i in range(dim_num):
        es += chars_left[i]
        es += chars_right[i]
        if i < dim_num - 1:
            es += ", "
    es += " -> ..."
    for i in range(dim_num):
        es += chars_right[i]
    return es

def _bits_to_bytes(bits):
    b = bits // 8
    if bits % 8 > 0:
        b += 1
    return b

def _truncate_idx_dtype(idx, bits):
    b = _bits_to_bytes(bits)
    
    x = idx.view(dtype=torch.uint8)
    x = rearrange(x, "... (d b) -> ... d b", d=idx.shape[-1], b=8)
    return x[..., :b]
    
def _mask_upcast_idx_dtype(idx, bits):
    upcast_bytes = 8 - idx.shape[-1]
    mask_index = bits // 8
    mask_bits = int((2 ** (bits % 8)) - 1)
    
    x = idx.clone()
    
    # If mask is inside of tensor truncation, mask away uneeded bits
    if mask_index < idx.shape[-1]:
        x[..., mask_index].bitwise_and_(mask_bits)
    
    # If need to upcast
    if upcast_bytes > 0:
        newshape = list(idx.shape)
        newshape[-1] = upcast_bytes
        zeros = torch.zeros(tuple(newshape), dtype=x.dtype, device=x.device)
        x = torch.concatenate((x, zeros), dim=-1)
        
    x = x.view(dtype=torch.int64)
    x = rearrange(x, "... d b -> ... (d b)")
    
    return x

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

class DCTTopKCollective(TransformCompressCollective):
    @torch.no_grad()
    def __init__(self, dct_chunk=64, dct_dim_num=2, dct_norm="ortho", indices_bits=7, quantize_values_1bit=True, dp_replicate_group: ProcessGroup | None = None):
        self.dct_chunk = dct_chunk
        self.dct_dim_num = dct_dim_num
        self.dct_pad_multiple = int(dct_chunk ** dct_dim_num)
        self.dp_replicate_group = dp_replicate_group

        I = torch.eye(self.dct_chunk, dtype=torch.float32)
        self.f = _dct(I, norm=dct_norm)
        self.b = _idct(I, norm=dct_norm)
        
        self.current_device = None
        self.current_dtype = None
        self.f_tuple = None
        self.b_tuple = None
        
        self.reshape_tuple = ((dct_chunk, ) * self.dct_dim_num)
        self.einsum_str = _build_einsum_dot_str(self.dct_dim_num)
        
        self.indices_bits = indices_bits
        self.indices_size = int(2 ** indices_bits)
        self.quantize_values_1bit = quantize_values_1bit
        self.fast_collate = (indices_bits == 7 and quantize_values_1bit is True)
        
        self.sync_queue = queue.Queue()
    
    def _to_valid(self, input):
        if input.device is not self.current_device or input.dtype is not self.current_dtype:
            self.current_device = input.device
            self.current_dtype = input.dtype
            self.f_tuple = ((self.f.to(self.current_dtype).to(self.current_device), ) * self.dct_dim_num)
            self.b_tuple = ((self.b.to(self.current_dtype).to(self.current_device), ) * self.dct_dim_num)
    
    def _encode(self, input):
        self._to_valid(input)
        x, size = _pad_to_divisible(input.flatten(), self.dct_pad_multiple)
        
        x = x.reshape(-1, *self.reshape_tuple)
        x = torch.einsum(self.einsum_str, x, *self.f_tuple)
        
        x = x.flatten()
        
        return x, {"size": size, "shape": input.shape}

    def _decode(self, input, size, shape):
        self._to_valid(input)
        x = input
        
        x = x.reshape(-1, *self.reshape_tuple)
        x = torch.einsum(self.einsum_str, x, *self.b_tuple)
        
        x = x.flatten()[:size].reshape(shape)
            
        return x

    def _compress(self, input: torch.Tensor, compression_ratio: float) -> tuple[list[Tensor], dict]:
        # Compute topk based on compression ratio
        model_bytes = input.element_size()
        compress_bytes = _bits_to_bytes(self.indices_bits + (1 if self.quantize_values_1bit else (input.element_size() * 8)))
        target_topk = int(compression_ratio * (model_bytes / compress_bytes) * self.indices_size)
        target_topk = clamp(target_topk, min_value=1, max_value=self.indices_size)
        
        # Pad tensor and reshape for topk
        x, size = _pad_to_divisible(input.flatten(), self.indices_size)
        x = x.reshape(-1, self.indices_size)
        
        # Compression using topk
        idx = torch.topk(x.abs(), k=target_topk, dim=-1, largest=True, sorted=False).indices
        val = torch.gather(x, dim=-1, index=idx)
        
        # Early compression to save on memory
        if self.fast_collate:
            idx = idx.to(dtype=torch.uint8)

        return [idx, val], {"size": size, "input_shape": input.shape, "scatter_shape": x.shape, "dtype": input.dtype, "device": input.device}
    
    def _decompress(self, input, size, input_shape, scatter_shape, dtype, device) -> torch.Tensor:
        idx = input[0]
        val = input[1]
        x = torch.zeros(scatter_shape, dtype=dtype, device=device)
        
        # TODO: Careful, this is nondeterministic across different CUDA devices! might cause errors to accumulate between nodes!
        x.scatter_reduce_(dim=-1, index=idx.to(dtype=torch.int64, device=device), src=val.to(dtype=dtype, device=device), reduce="mean", include_self=False)
        x = x.flatten()[:size].reshape(input_shape)
        
        return x
    
    def _collate(self, input: list[torch.Tensor]) -> tuple[torch.Tensor, dict]:
        idx = input[0]
        val = input[1]
        
        # Fast collate, inline optimizations for special case
        if self.fast_collate:
            cv = val.sign().to(dtype=torch.int8).add_(1).view(dtype=torch.uint8).bitwise_left_shift_(6)
            cv.add_(idx.to(dtype=torch.uint8))
            
            collated_tensor = cv[..., None]
            
            return collated_tensor, {"trunc_idx_shape": None, "val_dtype": None}

        elif self.quantize_values_1bit is False:
            trunc_idx = _truncate_idx_dtype(idx, self.indices_bits)
            trunc_val = rearrange(val.view(dtype=torch.uint8), "... (d b) -> ... d b", d=val.shape[-1])
            collated_tensor = torch.concatenate((trunc_idx, trunc_val), dim=-1).contiguous()
        else:
            trunc_idx = _truncate_idx_dtype(idx, self.indices_bits + 1)
            val_1bit = val.sign().to(dtype=torch.int8).add_(1).view(dtype=torch.uint8).bitwise_right_shift_(1).bitwise_left_shift_(self.indices_bits % 8)
            trunc_idx[..., -1].add_(val_1bit)
            collated_tensor = trunc_idx.contiguous()
            
        return collated_tensor, {"trunc_idx_shape": trunc_idx.shape, "val_dtype": val.dtype}
                
    def _disperse(self, input: torch.Tensor, trunc_idx_shape, val_dtype) -> list[torch.Tensor]:
        
        # Fast collate, inline optimizations for special case
        if self.fast_collate:
            idx = input[..., 0].bitwise_and(127)
            val = input[..., 0].bitwise_and(128).bitwise_right_shift_(6).view(dtype=torch.int8).sub_(1)
            
        elif self.quantize_values_1bit is False:
            trunc_idx = input[..., :trunc_idx_shape[-1]]
            trunc_val = input[..., trunc_idx_shape[-1]:]
            
            idx = _mask_upcast_idx_dtype(trunc_idx, self.indices_bits)
            val = rearrange(trunc_val.contiguous().view(dtype=val_dtype), "... d b -> ... (d b)")
            
        else:
            trunc_idx = input
            trunc_val = trunc_idx[..., -1].bitwise_right_shift(self.indices_bits % 8).view(dtype=torch.int8).bitwise_left_shift_(1).sub_(1)
            
            idx = _mask_upcast_idx_dtype(trunc_idx, self.indices_bits)
            val = trunc_val.to(dtype=val_dtype)
                
        return [idx, val]
    
    def _synchronize_send(self, input: Tensor) -> None:
        if self.dp_replicate_group is None:
            self.sync_queue.put(input)
        else:
            tensor_list = [torch.zeros_like(input) for _ in range(self.dp_replicate_group.size())]
            handle = dist.all_gather(tensor_list, input, group=self.dp_replicate_group, async_op=True)
            self.sync_queue.put((tensor_list, handle))
    
    def _synchronize_receive(self) -> Tensor | None:
        if self.sync_queue.empty():
            return None
        
        if self.dp_replicate_group is None:
            return self.sync_queue.get()
        else:
            tensor_list, handle = self.sync_queue.get()
            handle.wait()
            return torch.concatenate(tensor_list, dim=-2)
    
    # def _synchronize(self, input: torch.Tensor) -> torch.Tensor:
    #     if self.dp_replicate_group is None:
    #         return input
        
    #     tensor_list = [torch.zeros_like(input) for _ in range(self.dp_replicate_group.size())]
    #     dist.all_gather(tensor_list, input, group=self.dp_replicate_group)
    #     return torch.concatenate(tensor_list, dim=-2)
    
    
    
# Code modified and sourced from https://github.com/zh217/torch-dct
def _dct_fft_impl(v):
    return torch.view_as_real(torch.fft.fft(v, dim=1))


def _idct_irfft_impl(V):
    return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)


def _dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = _dct_fft_impl(v)

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * math.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == "ortho":
        V[:, 0] /= math.sqrt(N) * 2
        V[:, 1:] /= math.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def _idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == "ortho":
        X_v[:, 0] *= math.sqrt(N) * 2
        X_v[:, 1:] *= math.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * math.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = _idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, : N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, : N // 2]

    return x.view(*x_shape)
