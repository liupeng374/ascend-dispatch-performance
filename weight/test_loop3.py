import torch
import torch.nn as nn
import triton
import triton.language as tl

_FALLBACK_VEC_CORES = 40

# Ascend 910_9362 UB 预算(沿用原值, 含 allocator/多缓冲余量)。
_UB_BUDGET_BYTES = 226 * 1024

# ---- 2D tile 版本的 UB 模型 -------------------------------------------------
# 旧模型按 "2*mc+10 条 [BLOCK] 活向量" 给展开的 gather 链计费; 新内核把整行
# 物化为 [BLOCK, MCP] 的 2D tile, 峰值活 tile 约为:
#   idx(int32) / pv / mv->valid / src / cum / hit, 外加 cumsum 的多缓冲临时,
#   保守按 _TILE_LIVE 张 tile 计费; 1D 向量(offs/lid/r0/base/has_p/thr/...)
#   折算 _VEC_LIVE 条。这是保守估计: 上板编译后按实际 UB 占用下调 _TILE_LIVE
#   即可换回更大的 BLOCK。
_TILE_LIVE = 6
_VEC_LIVE = 16

# 候选 tile 宽度, 从大到小。2D 版本整行常驻, 单 token 的 UB 成本更高,
# 因此阶梯向下扩到 16。BLOCK 仍是 constexpr, 只由表几何决定, 不随 batch 变化。
_BLOCK_LADDER = (1024, 512, 256, 128, 64, 32, 16)


def _ub_need(pad_tab, mcp, block, tables):
    """UB 字节数: tables 张 fp32 常驻表 + 2D tile 活值。"""
    return 4 * tables * pad_tab + 4 * block * (_TILE_LIVE * mcp + _VEC_LIVE)


def _pick_block(pad_tab, mcp, tables):
    """满足 UB 预算的最大阶梯宽度; 都不满足返回 None。只看表几何, 不看 n。"""
    for cand in _BLOCK_LADDER:
        if _ub_need(pad_tab, mcp, cand, tables) <= _UB_BUDGET_BYTES:
            return cand
    return None


def _num_vectorcore(device_index):
    """当前设备 AIV 核数(仅 host 侧元数据)。"""
    try:
        from triton.runtime import driver

        props = driver.active.utils.get_device_properties(device_index)
        n = props["num_vectorcore"]
        if n > 0:
            return n
    except Exception:
        pass
    return _FALLBACK_VEC_CORES


def _pad_table(tab):
    """UB 表 tile  extent: 向上取 32B(8 个 fp32)粒度, 不必取 2 的幂。
    auto 模式下 tl.arange 不强制 2 的幂(检查只在 simt_only 生效), 尾段
    [tab, pad_tab) 装载时被掩码, 且任何 gather 索引都 < tab, 不会触达。"""
    return max(16, (tab + 7) // 8 * 8)


def _pad_mc(mc):
    """行 tile 的列数 MCP: 向上取 2 的幂(scan/cumsum  lowering 对 2 幂最稳)。
    若上板验证非 2 幂 scan 可用, 可改成 max(2, (mc + 7) // 8 * 8):
    mc=68 时 72 vs 128, tile 项 UB 几乎减半, 能换更大的 BLOCK。"""
    return max(2, triton.next_power_of_2(mc))


@triton.jit(do_not_specialize=["N", "NC"])
def dispatch_probability_kernel(
    topk_ptr,
    prob_ptr,
    map_ptr,
    rnd_ptr,
    out_ptr,
    N,
    NC,
    MC: tl.constexpr,
    MCP: tl.constexpr,
    TAB: tl.constexpr,
    PADT: tl.constexpr,
    BLOCK: tl.constexpr,
    PROB_UB: tl.constexpr,
    MAP_UB: tl.constexpr,
):
    """``out[i] = log2phy_map[topk_ids[i], inverse_cdf(log2phy_prob[topk_ids[i]])]``.

    无 MC 维度循环: 每个 token block 一次性取 [BLOCK, MCP] 的整行 tile,
    行和/逆 CDF 用 tl.sum + 单次 tl.cumsum + 掩码计数完成。指令数与 MC 无关,
    既没有 static_range 全展开导致的 bishengir 编译超线性, 也没有逐 c 的
    小向量指令发射开销。N/NC 仍是运行期标量, 不随 batch 重编译。
    """
    pid = tl.program_id(0).to(tl.int32)

    # --- stage 0: 整表驻留 UB, 每 program 一次(被外层 block 循环摊销) --------
    if PROB_UB or MAP_UB:
        ot = tl.arange(0, PADT)
        tm = ot < TAB
    if PROB_UB:
        p_tab = tl.load(prob_ptr + ot, mask=tm, other=0.0)
    else:
        p_tab = prob_ptr  # 占位: 仅 PROB_UB 时读取
    if MAP_UB:
        m_tab = tl.load(map_ptr + ot, mask=tm, other=-1).to(tl.float32)
    else:
        m_tab = map_ptr  # 占位: 仅 MAP_UB 时读取

    o = tl.arange(0, BLOCK)
    cols = tl.arange(0, MCP)
    cmask = cols < MC  # [MCP] 尾列掩码(MCP > MC 的填充列)

    # 外层运行期循环保留: 它按网格步长走 N, 用来摊销 stage 0 的整表预载,
    # 不是被优化掉的 MC 循环; 去掉它会导致每个 block 都重新预载表。
    n_blocks = tl.cdiv(N, BLOCK)
    for i in tl.range(0, tl.cdiv(n_blocks, NC)):
        blk = pid + i * NC
        offs = blk * BLOCK + o
        msk = offs < N

        # int64 ids 装载后立即截断为 int32
        lid = tl.load(topk_ptr + offs, mask=msk, other=0).to(tl.int32)
        # 随机数提前发射, 与 gather 链无数据依赖, MTE2 可重叠
        r0 = tl.load(rnd_ptr + offs, mask=msk, other=0.0)
        base = lid * MC

        # --- stage 1: 一次取整行 [BLOCK, MCP], 替代两遍逐 c 读取 -------------
        idx2 = base[:, None] + cols[None, :]
        lmask = msk[:, None] & cmask[None, :]
        if PROB_UB or MAP_UB:
            # tl.gather 无 mask 参数: 填充列索引钳到 0, 取回后再置 0/-1。
            # 否则 base + MCP - 1 可能越过 PADT(如 MC=68, MCP=128)。
            # src/index 必须同秩, 故展平成 1D gather 再 reshape 回 2D。
            ig = tl.where(cmask[None, :], idx2, 0)
            igf = tl.reshape(ig, (BLOCK * MCP,))
        if PROB_UB:
            pv = tl.reshape(tl.gather(p_tab, igf, 0), (BLOCK, MCP))
            pv = tl.where(cmask[None, :], pv, 0.0)
        else:
            pv = tl.load(prob_ptr + idx2, mask=lmask, other=0.0)
        if MAP_UB:
            mv = tl.reshape(tl.gather(m_tab, igf, 0), (BLOCK, MCP))
            mv = tl.where(cmask[None, :], mv, -1.0)
        else:
            mv = tl.load(map_ptr + idx2, mask=lmask, other=-1).to(tl.float32)

        # --- stage 2: 行和 + 逆 CDF, 单次 cumsum ------------------------------
        valid = (mv >= 0.0).to(tl.float32)
        # 概率非负 => "和>0" 与求和顺序无关, 可安全用树形归约判零
        has_p = tl.sum(pv, axis=1) > 0.0
        # 零概率行回退为 "有效副本均匀采样": 把扫描源换成 0/1 valid,
        # 一次 cumsum 同时覆盖 概率CDF / 有效副本前缀计数 两种语义
        src = tl.where(has_p[:, None], pv, valid)
        cum = tl.cumsum(src, axis=1)
        # 阈值取扫描的最后一个有效列 => 保持 row_sum == cum_last 不变量,
        # 避免 tl.sum 与 cumsum 结合顺序不同造成的 1ulp 边界翻转;
        # 回退行该列恰为 nvalid(整数, fp32 精确)
        last = tl.sum(tl.where(cols[None, :] == MC - 1, cum, 0.0), axis=1)
        thr = r0 * last
        # chosen = #{c < MC : cum[c] <= thr}, 尾列必须掩掉
        hit = (cum <= thr[:, None]) & cmask[None, :]
        chosen = tl.sum(hit.to(tl.int32), axis=1)
        chosen = tl.minimum(chosen, MC - 1)  # clamp(max=MC-1)

        # --- stage 3: 一次查表得物理 id ---------------------------------------
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
) -> torch.Tensor:
    original_shape = topk_ids.shape
    flat_ids = topk_ids.reshape(-1)
    n = flat_ids.numel()
    num_logical, max_copies = log2phy_prob.shape

    # 空 rank 可能拿到空张量; 0 网格非法
    if n == 0:
        return torch.empty(
            original_shape, dtype=topk_ids.dtype, device=topk_ids.device
        )
    nvec = _num_vectorcore(0)

    prob = log2phy_prob.contiguous()
    mp = log2phy_map.contiguous()
    rv = random_vals.contiguous()
    out = torch.empty(n, dtype=topk_ids.dtype, device=topk_ids.device)

    # 预载策略不变: 优先双表驻留, 其次单表, 最后纯 GM。
    # 选择只依赖表几何, 与 n 无关 => 服务运行期只编译一次。
    tab = num_logical * max_copies
    pad_tab = _pad_table(tab)
    mcp = _pad_mc(max_copies)
    prob_ub = map_ub = False
    picked = _pick_block(pad_tab, mcp, 2)
    if picked is not None:
        block, prob_ub, map_ub = picked, True, True
    else:
        picked = _pick_block(pad_tab, mcp, 1)
        if picked is not None:
            block, prob_ub = picked, True
        else:
            # 纯 GM 路径: 没有表项, 但 2D tile 本身仍占 UB, 同样按模型定宽
            block = _pick_block(0, mcp, 0)
            if block is None:
                block = 1  # 病态大 MC 的保底, 实际 MoE 布局不会走到
    nb = (n + block - 1) // block
    nc = min(nvec, nb)
    dispatch_probability_kernel[(nc,)](
        flat_ids,
        prob,
        mp,
        rv,
        out,
        n,
        nc,
        MC=max_copies,
        MCP=mcp,
        TAB=tab,
        PADT=pad_tab,
        BLOCK=block,
        PROB_UB=prob_ub,
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
    times_us = []
    print()
    print(f"{'write_pos':>10s}  {'kind':<26s}  {'latency':>10s}")
    for wp in range(10):
        t = bench(lambda: dispatch(topk_ids,log2phy_prob,log2phy_map,random_vals)) * 1000.0    
        times_us.append(t)
    


    #out = dispatch(topk_ids,log2phy_prob,log2phy_map,random_vals)
    print(times_us)