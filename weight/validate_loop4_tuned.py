import torch
from test_loop3 import dispatch as baseline, bench
from test_loop4_tuned import dispatch, kernel, prepare
ids, p, m, r = [torch.load('weight/' + name + '.pt') for name in ('topk_ids','log2phy_prob','log2phy_map','random_vals')]
a = baseline(ids,p,m,r); b = dispatch(ids,p,m,r)
print('mismatches', (a != b).sum().item(), 'config', kernel.best_config, flush=True)
assert torch.equal(a,b)
for name, fn in [('baseline_loop3',baseline), ('tuned',dispatch)]:
    times_us = [bench(lambda:fn(ids,p,m,r))*1000 for _ in range(10)]
    print(name, 'times_us:', times_us, flush=True)
assert torch.equal(baseline(ids,p,m,r), dispatch(ids,p,m,r))
print('PASS cached compiled launch', flush=True)
