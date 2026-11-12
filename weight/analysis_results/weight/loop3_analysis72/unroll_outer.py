import triton
import triton.language as tl
@triton.jit(do_not_specialize=["N", "NC"])
def ablate_unroll_outer(
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
    NC = tl.minimum(NC, tl.cdiv(N, BLOCK))
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
    for i in tl.range(0, tl.cdiv(n_blocks, NC), loop_unroll_factor=2):
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
