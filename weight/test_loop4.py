import torch
import triton
import triton.language as tl

_FALLBACK_VEC_CORES = 40

_BLOCK = 256

# Row widths up to this are fully unrolled; beyond it a runtime loop with a
# bounded unroll factor keeps bishengir's auto-blockify pass linear.
_UNROLL_MAX: tl.constexpr = 8
_CHUNK_UNROLL: tl.constexpr = 4

# True: 用一个微 kernel 按 c 顺序累加求 row_sum，与旧 kernel 逐位一致；
# False: 用 torch.sum（pairwise），与 torch reference 的 sum 语义一致。
# 两者只在最后一个 ulp 上可能不同，仅影响恰好落在 CDF 边界上的 token。
_SEQ_ROWSUM = False
_PREP_BLK = 128


def _num_vectorcore(device_index):
    """Number of AIV cores for the current device (host-side metadata only)."""
    try:
        from triton.runtime import driver

        props = driver.active.utils.get_device_properties(device_index)
        n = props["num_vectorcore"]
        if n > 0:
            return n
    except Exception:
        pass
    return _FALLBACK_VEC_CORES


def _next_pow2(n, floor=16):
    p = floor
    while p < n:
        p *= 2
    return p


# ---------------------------------------------------------------------------
# prep: 每个逻辑专家一行，把 "有概率走概率 / 全零走有效副本数" 折叠成
# 统一权重表 w[L, MC] 和统一总和表 s[L]。
# ---------------------------------------------------------------------------

@triton.jit(do_not_specialize=["L"])
def _row_sum_kernel(prob_ptr, rs_ptr, L, MC: tl.constexpr, BLK: tl.constexpr):
    """按 c 顺序累加的 row_sum，与旧 kernel 的求和顺序逐位一致。

    展开不改变 acc 的串行依赖链，所以 loop_unroll_factor 只影响编译时间。
    """
    pid = tl.program_id(0).to(tl.int32)
    rows = pid * BLK + tl.arange(0, BLK)
    rm = rows < L
    acc = tl.zeros([BLK], tl.float32)
    for c in tl.range(0, MC, loop_unroll_factor=4):
        acc += tl.load(prob_ptr + rows * MC + c, mask=rm, other=0.0)
    tl.store(rs_ptr + rows, acc, mask=rm)


def _prepare_tables(prob: torch.Tensor, mp: torch.Tensor):
    """prob: [L, MC] fp32; mp: [L, MC] 整型。返回 (w[L, MC] fp32, s[L] fp32)。

    w[row] = prob[row]            若 row_sum > 0
           = (map[row] >= 0)      否则（fallback：有效副本 0/1 权重）
    s[row] = row_sum 或 nvalid，与 w 对应。
    成本是 L*MC 规模的几个小算子，与 token 数 N 无关。
    """
    pf = prob.to(torch.float32)          # fp32 输入时为 no-op
    valid = (mp >= 0).to(torch.float32)
    if _SEQ_ROWSUM:
        L, MC = pf.shape
        row_sum = torch.empty(L, dtype=torch.float32, device=pf.device)
        _row_sum_kernel[(triton.cdiv(L, _PREP_BLK),)](
            pf, row_sum, L, MC=MC, BLK=_PREP_BLK
        )
    else:
        row_sum = pf.sum(dim=1)
    nvalid = valid.sum(dim=1)            # 0/1 计数，fp32 下任意顺序都精确
    has_p = row_sum > 0.0
    w = torch.where(has_p.unsqueeze(1), pf, valid).contiguous()
    s = torch.where(has_p, row_sum, nvalid).contiguous()
    return w, s


# ---------------------------------------------------------------------------
# 融合 kernel：单循环 inverse-CDF
# ---------------------------------------------------------------------------

@triton.jit
def _w_at(w_src, w_ptr, idx, msk, W_UB: tl.constexpr):
    """One weight of a row, from the UB table or from GM."""
    if W_UB:
        return tl.gather(w_src, idx, 0)
    else:
        return tl.load(w_ptr + idx, mask=msk, other=0.0)


@triton.jit(do_not_specialize=["N", "NC"])
def dispatch_probability_kernel(
    topk_ptr,
    w_ptr,
    s_ptr,
    map_ptr,
    rnd_ptr,
    out_ptr,
    N,
    NC,
    MC: tl.constexpr,
    TAB: tl.constexpr,
    PADT: tl.constexpr,
    L: tl.constexpr,
    PADL: tl.constexpr,
    BLOCK: tl.constexpr,
    W_UB: tl.constexpr,
    MAP_UB: tl.constexpr,
):
    """``out[i] = map[base + chosen]``，其中
    ``chosen = Σ_c 1[cum_c <= r0 * s[row]]``，``cum`` 为统一权重 ``w`` 的前缀和。

    row_sum / nvalid / has_p 都是每行常量，已在 host 侧折叠进 w/s，
    因此这里只需一个循环：每 c 一次 gather + 一次加法 + 一次比较。
    每 token 的 gather 次数从 4*MC+1 降到 MC+2。

    ``N``/``NC`` 是 runtime 标量（do_not_specialize），batch 变化不触发重编译；
    所有 tile 几何（MC/TAB/PADT/L/PADL/BLOCK）和路径开关（W_UB/MAP_UB）
    由 expert 布局固定，保持 constexpr。
    """
    pid = tl.program_id(0).to(tl.int32)

    # --- stage 0: tables into UB, once per program --------------------------
    if W_UB or MAP_UB:
        ot = tl.arange(0, PADT)
        tm = ot < TAB
    if W_UB:
        w_tab = tl.load(w_ptr + ot, mask=tm, other=0.0)
    else:
        w_tab = w_ptr  # placeholder: only read when W_UB
    if MAP_UB:
        m_tab = tl.load(map_ptr + ot, mask=tm, other=-1).to(tl.float32)
    else:
        m_tab = map_ptr  # placeholder: only read when MAP_UB
    # s 表只有 L 个 fp32（如 256 行 = 1 KiB），无条件常驻 UB
    ol = tl.arange(0, PADL)
    s_tab = tl.load(s_ptr + ol, mask=ol < L, other=0.0)

    o = tl.arange(0, BLOCK)
    n_blocks = tl.cdiv(N, BLOCK)
    for i in tl.range(0, tl.cdiv(n_blocks, NC)):
        blk = pid + i * NC
        offs = blk * BLOCK + o
        msk = offs < N

        # int64 ids are truncated right after the load
        lid = tl.load(topk_ptr + offs, mask=msk, other=0).to(tl.int32)
        # random-value load up-front: no data dependency on the gather chain
        r0 = tl.load(rnd_ptr + offs, mask=msk, other=0.0)
        base = lid * MC

        # 全行和一次 gather 拿到；fallback 已折叠，u 对两种行统一成立
        u = r0 * tl.gather(s_tab, lid, 0)

        # --- 唯一的循环：顺序前缀和 + 计数（与 reference 同序） ---------------
        cum = tl.zeros([BLOCK], tl.float32)
        chosen = tl.zeros([BLOCK], tl.int32)
        if MC <= 8:
            for c in tl.static_range(0, MC):
                wv = _w_at(w_tab, w_ptr, base + c, msk, W_UB)
                cum += wv
                chosen += (cum <= u).to(tl.int32)
        else:
            for c in tl.range(0, MC, loop_unroll_factor=4):
                wv = _w_at(w_tab, w_ptr, base + c, msk, W_UB)
                cum += wv
                chosen += (cum <= u).to(tl.int32)
        chosen = tl.where(chosen > MC - 1, MC - 1, chosen)  # clamp(max=MC-1)

        # --- one lookup picks the physical id --------------------------------
        if MAP_UB:
            res = tl.gather(m_tab, base + chosen, 0).to(out_ptr.dtype.element_ty)
        else:
            res = tl.load(map_ptr + base + chosen, mask=msk, other=-1).to(
                out_ptr.dtype.element_ty
            )
        tl.store(out_ptr + offs, res, mask=msk)


def dispatch(
    topk_ids: torch.Tensor,
    log2phy_prob: torch.Tensor,
    log2phy_map: torch.Tensor,
    random_vals: torch.Tensor,
    prepared=None,
) -> torch.Tensor:
    original_shape = topk_ids.shape
    flat_ids = topk_ids.reshape(-1)
    n = flat_ids.numel()
    num_logical, max_copies = log2phy_prob.shape

    # Idle ranks can reach here with an empty tensor; a zero grid is invalid.
    if n == 0:
        return torch.empty(
            original_shape, dtype=topk_ids.dtype, device=topk_ids.device
        )
    nvec = 32

    prob = log2phy_prob.contiguous()
    mp = log2phy_map.contiguous()
    rv = random_vals.contiguous()
    out = torch.empty(n, dtype=topk_ids.dtype, device=topk_ids.device)

    # 按行预计算，成本 O(L*MC)，与 N 无关
    w, s = _prepare_tables(prob, mp) if prepared is None else prepared

    tab = num_logical * max_copies
    pad_tab = _next_pow2(tab)
    pad_l = _next_pow2(num_logical)
    block = _BLOCK
    w_ub = map_ub = True

    # Runtime launch geometry: never constexpr.
    nb = (n + block - 1) // block
    nc = min(nvec, nb)
    dispatch_probability_kernel[(nc,)](
        flat_ids,
        w,
        s,
        mp,
        rv,
        out,
        n,
        nc,
        MC=max_copies,
        TAB=tab,
        PADT=pad_tab,
        L=num_logical,
        PADL=pad_l,
        BLOCK=block,
        W_UB=w_ub,
        MAP_UB=map_ub,
    )
    return out.view(original_shape)

def bench(fn, warmup=20, iters=200):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    s = torch.npu.Event(enable_timing=True)
    e = torch.npu.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.npu.synchronize()
    return s.elapsed_time(e) / iters                 # ms per call

# ---- sweep write_pos = 0 .. L-1 -------------------------------------------
# Fresh state per case so repeated flushes don't drive state -> 0 / NaN.

if __name__ == "__main__":
    topk_ids = torch.load("./weight/topk_ids.pt")
    log2phy_prob = torch.load("./weight/log2phy_prob.pt")
    log2phy_map = torch.load("./weight/log2phy_map.pt")
    random_vals = torch.load("./weight/random_vals.pt")

    out = dispatch(topk_ids,log2phy_prob,log2phy_map,random_vals)
    print("**********topk_ids shape ",topk_ids.shape,flush=True)
    print("**********log2phy_prob shape ",log2phy_prob,flush=True)
    print("**********log2phy_map shape ",log2phy_map,flush=True)

    # '''
    print("**********topk_ids shape ",topk_ids.shape,flush=True)
    times_us = []
    print()
    print(f"{'write_pos':>10s}  {'kind':<26s}  {'latency':>10s}")
    for wp in range(10):
        t = bench(lambda: dispatch(topk_ids,log2phy_prob,log2phy_map,random_vals)) * 1000.0    
        times_us.append(t)
    print(times_us)
    # '''
    
