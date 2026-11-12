import triton
import triton.language as tl
@triton.jit(do_not_specialize=["N", "NC"])
def ablate_io_only(
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
    for i in tl.range(0, tl.cdiv(n_blocks, NC)):
        blk = pid + i * NC
        offs = blk * BLOCK + o
        msk = offs < N

        # int64 ids 装载后立即截断为 int32
        lid = tl.load(topk_ptr + offs, mask=msk, other=0).to(tl.int32)
        # 随机数提前发射, 与 gather 链无数据依赖, MTE2 可重叠
        r0 = tl.load(rnd_ptr + offs, mask=msk, other=0.0)
        base = lid * MC

        res = (lid.to(tl.float32) + r0).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + offs, res, mask=msk)
