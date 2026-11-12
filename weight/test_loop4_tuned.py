"""Precomputed sequential CDF; vector-core-bounded binary search dispatch."""
from functools import lru_cache
import torch
import triton
import triton.language as tl


def prepare(prob, mapping):
    from test_loop4 import _prepare_tables
    weights, sums = _prepare_tables(prob, mapping)
    cdf = torch.empty_like(weights)
    _prepare_cdf[(triton.cdiv(weights.shape[0], 64),)](
        weights, cdf, weights.shape[0], MC=weights.shape[1], BLOCK=64)
    return cdf, sums


@triton.jit
def _prepare_cdf(weights, cdf, L, MC: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    for c in range(MC):
        acc += tl.load(weights + rows * MC + c, rows < L, 0)
        tl.store(cdf + rows * MC + c, acc, rows < L)


@triton.autotune(configs=[triton.Config({'BLOCK': b}) for b in (64,128,256,512)],
                 key=['N', 'L', 'MC'])
@triton.jit
def kernel(ids, cdf, sums, mapping, random, out, N: tl.constexpr, L: tl.constexpr,
           MC: tl.constexpr, BLOCK: tl.constexpr):
    ot = tl.arange(0, triton.next_power_of_2(L * MC))
    cdf_tab = tl.load(cdf + ot, ot < L * MC, float('inf'))
    map_tab = tl.load(mapping + ot, ot < L * MC, -1).to(tl.float32)
    ol = tl.arange(0, triton.next_power_of_2(L))
    sums_tab = tl.load(sums + ol, ol < L, 0)
    for tile in tl.range(tl.program_id(0), tl.cdiv(N, BLOCK), tl.num_programs(0)):
        rows = tile * BLOCK + tl.arange(0, BLOCK)
        active = rows < N
        lid = tl.load(ids + rows, active, 0).to(tl.int32)
        r = tl.load(random + rows, active, 0)
        s = tl.gather(sums_tab, lid, 0)
        u = r * s
        lo = tl.full((BLOCK,), 0, tl.int32)
        hi = tl.full((BLOCK,), MC, tl.int32)
        for step in tl.static_range(0, (MC + 1).bit_length()):
            mid = (lo + hi) // 2
            v = tl.gather(cdf_tab, lid * MC + tl.minimum(mid, MC - 1), 0)
            v = tl.where(mid < MC, v, float('inf'))
            right = v <= u
            lo = tl.where((lo < hi) & right, mid + 1, lo)
            hi = tl.where((lo < hi) & ~right, mid, hi)
        chosen = tl.minimum(lo, MC - 1)
        result = tl.gather(map_tab, lid * MC + chosen, 0).to(out.dtype.element_ty)
        tl.store(out + rows, result, active)


@lru_cache(None)
def _vector_num(device):
    from triton.runtime import driver
    return int(driver.active.utils.get_device_properties(device.index)['num_vectorcore'])


_configs = {}


def dispatch(ids, prob, mapping, random, prepared=None):
    if prepared is None:
        prepared = prepare(prob, mapping)
    cdf, sums = prepared
    flat = ids.contiguous().view(-1)
    n = flat.numel()
    out = torch.empty_like(flat)
    if n:
        l, mc = prob.shape
        vector_num = _vector_num(ids.device)
        key = (ids.device, flat.dtype, cdf.dtype, mapping.dtype, random.dtype, n, l, mc)
        args = (flat, cdf, sums, mapping.contiguous(), random.contiguous(), out,n,l,mc)
        config = _configs.get(key)
        if config is None:
            kernel[lambda meta: (min(vector_num, triton.cdiv(n, meta['BLOCK'])),)](*args)
            config = kernel.best_config
            block = config.kwargs['BLOCK']
            grid = (min(vector_num, triton.cdiv(n, block)), 1, 1)
            compiled = kernel.fn[grid](*args, **config.kwargs,
                num_warps=config.num_warps, num_stages=config.num_stages)
            _configs[key] = (compiled, grid, block)
        else:
            compiled, grid, block = config
            compiled[grid](*args, block)
    return out.view(ids.shape)
