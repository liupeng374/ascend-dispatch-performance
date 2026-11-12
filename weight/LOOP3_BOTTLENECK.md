# loop3 注释二分与 UB/循环分析

环境为容器 sglang-fby，Triton 3.6.0，编译目标 Ascend950DT_9582。当前数据 N=3168,L=64,MC=72，vector_num=64。此前代码中的 910/192 KiB 注释不能当作此机器的硬件参数。

主结论：当前热点是二维 CDF 扫描及其输入布局/中间量；末端输出 map 查询不是主热点。最有效的已测方向是减少列填充，并重新选择 BLOCK 和表驻留策略，非展开外层循环。

## 消融方法与限制

脚本 analyze_loop3.py 从当前 kernel 生成独立诊断版本，分阶段截断后把剩余上游值归约并写回，避免直接注释导致整条依赖被编译器消除。io_only 中无用表预载仍会被 DCE，因此它代表最小 IO 下界，不是完整内核删掉计算的精确替代。各阶段编译器的融合、分配和同步也会变化，时间差不能简单相加为逐行成本。

测量固定 BLOCK=32,PROB_UB=True,MAP_UB=False,MCP=128,PADT=8192。下表为 msprof 每配置最后 5 次 Task Duration 均值，而非 Python times_us。诊断版本用直接编译默认配置；生产版使用 autotune Config 的 num_warps=4,num_stages=3，最终性能另用完整 dispatch 比较，勿混用两种口径。

|保留的路径|设备耗时 us|
|---|---:|
|完整 kernel|35.44|
|只做 token/random IO，移除行读取与后续计算|3.24 左右（最后五次约 3.24）|
|行读取后归约写回，移除 fallback/scan/select|10.75|
|保留 fallback 权重构造，移除 scan/select|10.92|
|保留 scan，移除阈值/计数/最终 map|33.18|
|仅移除最终 map 查询，输出 chosen|37.03|

注：精确原始五次值见 analysis_results/profile_ablation 的 op_summary CSV。scan 的引入带来最大增量；最终 map 移除没有变快，故不能把该时间差解释为 map 为负成本。

## 循环

1. 显式外层循环：BLOCK=32 时 ceil(3168/32)=99 个 tile，64 个 program，每个按当前公共上界执行两轮。部分 program 的第二轮全部 mask。改成 for tile in range(pid,n_blocks,NC) 跳过空 tile，在紧凑72列版本约24.18–24.37 us，对照24.30–24.38 us，未见稳定收益。
2. 移除外层循环并启动99个program，128列设备约36.74 us，差于35.44 us；72列约24.42 us，对照24.91 us，收益小且违反固定vector_num上限的偏好。
3. loop_unroll_factor=2：128列编译 UB overflow；72列可编译但约25.87 us，慢于对照24.91 us。暂不采用。
4. 源码没有MC循环，但编译器会生成循环。bcmlir 中 GM map 路径存在逐行 scf.for 并分配32x128xi64临时块；cumsum调用也需lowering，不能把“源码无循环”当作“机器代码无循环”。
5. 紧凑版选中BLOCK=64后仅50个tile，grid=50（不超过64），外层只一轮。它通过布局和BLOCK改善自然降低循环轮数，无需强行展开。

## UB 是否合理

- 原MCP=128，真实MC=72：43.75%列为padding；tile元素数从32x128=4096可降到32x72=2304，下降43.75%。单个fp32 tile从16 KiB到9 KiB。int64 map tile从32 KiB到18 KiB；还不含其它活跃临时块与多缓冲。
- 原PADT=8192，实际表4608项。每张fp32逻辑表从32 KiB到18 KiB，但实际物理峰值不能仅据该数字推算。
- 只改PADT=4608、仍MCP=128，没有收益（设备37.29 us vs35.44 us）。同时改MCP=72则降到24.49 us，因此主要收益是tile/scan布局，不是单纯少预载几个表项。
- 128列BLOCK=32双UB编译溢出：需要2309120 bits=281.875 KiB，可用2031616 bits=248 KiB。BLOCK=64的多个组合亦溢出。外层展开2需要2059264 bits=251.375 KiB，可用1769472 bits=216 KiB。这是编译器给出的特定配置预算，不是声称硬件总UB容量就是这些值。
- 紧凑72列后，BLOCK=64双UB可以编译，并成为生产autotune最优。可见“MAP_UB=False最快”依赖旧布局；修复padding后结论改变。
- metadata.shared_mem_dynamic_size=221184是该后端元数据，不应直接标成实际活跃UB峰值；中间bcmlir也不是最终物理内存规划。此次未得到成功配置最终UB峰值，因此不声称测出了精确占用率。

## 完整 dispatch 的 times_us 验证

新增实验副本 test_loop3_compact.py，保留 test_loop3.py 和 test_loop3_tuned.py。
所有数据相关计算保留在计时中，只改变MCP到8列对齐、PADT到8元素对齐，并重新联合autotune。

同场：base大多44.2–44.8 us（一次48.46）；当前版35.18–35.37 us；紧凑版28.32–29.12 us。紧凑版选BLOCK=64,PROB_UB=True,MAP_UB=True，相比当前约降低19%，相比base典型值约降低36%。

复现：docker exec -w /tmp/weight-loop4-validation sglang-fby python weight/validate_loop3_compact.py

该紧凑非2幂scan只在当前Triton-Ascend与数据布局验证，不能推断其它backend同样支持。随机浮点CDF边界还需持续关注；源代码保留相同cumsum/阈值语义，但改变形状可能改变编译器结合顺序。

证据：analysis_results 下包含两轮msprof CSV、results.json、source/ttir/ttadapter/bcmlir。重跑分析：python weight/analyze_loop3.py，或增加 --compact。

最终复测：base43.56–44.58 us；当前版35.23–35.36 us（一次37.00）；紧凑版28.26–28.52 us。保存数据以及N=0/1/129/3168随机、零概率fallback、r=0/1与缓存启动路径均通过。所有完整调用测量均含数据相关处理。
