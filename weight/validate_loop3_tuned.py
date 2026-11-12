import torch
from test_loop3 import dispatch as baseline, bench
from test_loop3_tuned import dispatch, dispatch_probability_kernel
args=[torch.load('weight/'+n+'.pt') for n in ('topk_ids','log2phy_prob','log2phy_map','random_vals')]
assert torch.equal(baseline(*args),dispatch(*args))
print('PASS; config:',dispatch_probability_kernel.best_config,flush=True)
for name,fn in [('base',baseline),('tuned',dispatch)]:
 print(name,[bench(lambda:fn(*args))*1000 for _ in range(10)],flush=True)

assert torch.equal(baseline(*args),dispatch(*args))
print('PASS cached compiled launch',flush=True)
