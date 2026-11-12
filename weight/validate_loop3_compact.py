import torch
from test_loop3 import dispatch as baseline, bench
from test_loop3_tuned import dispatch as current
from test_loop3_compact import dispatch, dispatch_probability_kernel
args=[torch.load('weight/'+n+'.pt') for n in ('topk_ids','log2phy_prob','log2phy_map','random_vals')]
for _ in range(2):
    assert torch.equal(baseline(*args),dispatch(*args))
print('PASS saved/cached launch; config:', dispatch_probability_kernel.best_config,flush=True)
for name,fn in [('base',baseline),('current',current),('compact',dispatch)]:
    print(name,[bench(lambda:fn(*args))*1000 for _ in range(10)],flush=True)
torch.manual_seed(42)
for n in (0,1,129,3168):
    p=torch.rand((64,72),device='npu');m=torch.arange(4608,device='npu').reshape(64,72)
    p[:,36:]=0;m[:,36:]=-1;p[::3]=0;m[0]=-1
    ids=torch.randint(0,64,(n,),device='npu');r=torch.rand(n,device='npu');r[::3]=0;r[1::3]=1
    for _ in range(2):
        assert torch.equal(baseline(ids,p,m,r),dispatch(ids,p,m,r)),n
    print('PASS random/fallback/boundary/cached N=',n,flush=True)
