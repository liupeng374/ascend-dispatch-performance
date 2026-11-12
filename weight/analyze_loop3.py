"""Controlled stage ablations. Diagnostic outputs are intentionally not dispatch results."""
from pathlib import Path
import importlib.util
import json
import argparse
import torch
import triton
from test_loop3 import dispatch as baseline
from test_loop3_tuned import dispatch_probability_kernel

parser = argparse.ArgumentParser()
parser.add_argument('--compact', action='store_true')
options = parser.parse_args()
root = Path('weight/loop3_analysis72' if options.compact else 'weight/loop3_analysis')
root.mkdir(exist_ok=True)
src = Path('weight/test_loop3_tuned.py').read_text()
a = src.index('@triton.jit('); b = src.index('\n\n_launch_cache', a)
source = src[a:b]
variants = {'full': source}
# Keep the removed stage's upstream values observable, avoiding accidental DCE.
a = source.index('        # --- stage 2:'); b = source.index('        # --- stage 3:',a)
variants['rows_only'] = source[:a] + '''        res = tl.sum(pv + mv, 1).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + offs, res, mask=msk)
'''
a = source.index('        cum = tl.cumsum'); b = source.index('        # --- stage 3:',a)
variants['no_scan_select'] = source[:a] + '''        res = tl.sum(src, 1).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + offs, res, mask=msk)
'''
a = source.index('        # --- stage 3:')
variants['no_final_map'] = source[:a] + '''        tl.store(out_ptr + offs, chosen.to(out_ptr.dtype.element_ty), mask=msk)
'''
a = source.index('        # --- stage 1:')
variants['io_only'] = source[:a]+'''        res = (lid.to(tl.float32) + r0).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + offs, res, mask=msk)
'''
variants['scan_only'] = variants['no_final_map'].replace('        last = tl.sum(tl.where(cols[None, :] == MC - 1, cum, 0.0), axis=1)', '        last = tl.sum(tl.where(cols[None, :] == MC - 1, cum, 0.0), axis=1)')
a = source.index('        last ='); variants['scan_only'] = source[:a]+'''        res = tl.sum(cum, 1).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + offs, res, mask=msk)
'''
variants['unroll_outer'] = source.replace('for i in tl.range(0, tl.cdiv(n_blocks, NC)):', 'for i in tl.range(0, tl.cdiv(n_blocks, NC), loop_unroll_factor=2):')
variants['no_outer'] = source.replace('    for i in tl.range(0, tl.cdiv(n_blocks, NC)):', '    for i in tl.static_range(0, 1):')
args = [torch.load('weight/'+n+'.pt') for n in ('topk_ids','log2phy_prob','log2phy_map','random_vals')]
ids,p,m,r=args; n=ids.numel(); l,mc=p.shape; out=torch.empty_like(ids)
expected=baseline(*args)
results=[]
for name, code in variants.items():
 path=root/(name+'.py'); path.write_text('import triton\nimport triton.language as tl\n'+code.replace('def dispatch_probability_kernel(', 'def ablate_'+name+'('))
 spec=importlib.util.spec_from_file_location('ablate_'+name,path); mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 fn=getattr(mod,'ablate_'+name)
 configs=[(32,True,False,72,4608)] if options.compact else [(32,True,False,128,8192)]
 if name=='full':
  configs=[(b,pub,mub,72 if options.compact else 128,4608 if options.compact else 8192) for b in (16,32,64) for pub in (False,True) for mub in (False,True)]
  if not options.compact: configs += [(32,True,False,128,4608),(32,True,False,72,4608)]
 for block,pub,mub,mcp,padt in configs:
  nc= min(64,triton.cdiv(n,block)) if name!='no_outer' else triton.cdiv(n,block)
  kw=dict(MC=mc,MCP=mcp,TAB=l*mc,PADT=padt,BLOCK=block,PROB_UB=pub,MAP_UB=mub)
  launch_args=(ids,p,m,r,out,n,nc)
  tag=f'{name}_b{block}_p{pub}_m{mub}_c{mcp}_t{padt}'
  try:
   compiled=fn[(nc,1,1)](*launch_args,**kw)
   if name in ('full','unroll_outer','no_outer'): assert torch.equal(expected,out),tag
   def run(): compiled[(nc,1,1)](*launch_args,mc,mcp,l*mc,padt,block,pub,mub)
   for _ in range(20): run()
   torch.npu.synchronize()
   start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
   start.record()
   for _ in range(200): run()
   end.record();torch.npu.synchronize()
   us=start.elapsed_time(end)*1000/200
   meta=str(compiled.metadata)
   for ext,txt in compiled.asm.items():
    if isinstance(txt,str): (root/(tag+'.'+ext)).write_text(txt)
   row=dict(tag=tag,times_us=us,metadata=meta,asm_keys=list(compiled.asm))
   print(json.dumps(row),flush=True);results.append(row)
  except Exception as e:
   print('FAILED',tag,str(e)[:700],flush=True);results.append(dict(tag=tag,error=str(e)))
(root/'results.json').write_text(json.dumps(results,indent=2))
