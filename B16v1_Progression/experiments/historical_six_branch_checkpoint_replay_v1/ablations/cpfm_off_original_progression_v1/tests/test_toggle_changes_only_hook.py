import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from torch import nn
from cpfm_toggle_adapter import temporary_cpfm_enabled,hook_count
class Block(nn.Module):
 def __init__(self):
  super().__init__(); self.resnets=nn.ModuleList([nn.Identity()])
class Net(nn.Module):
 def __init__(self):
  super().__init__(); self.down_blocks=nn.ModuleList([Block() for _ in range(4)])
class Hooks:
 def __init__(self,unet,cpfm,holder):
  self.holder=holder; self.handles=[b.resnets[0].register_forward_pre_hook(self._make_hook(i)) for i,b in enumerate(unet.down_blocks)]
 def _active(self,i): return self.holder['enabled']
 def _make_hook(self,i):
  def f(m,args): return (args[0]+self.holder['condition'],) if self._active(i) else args
  return f
 def close(self):
  for h in self.handles: h.remove()
class ToggleTest(unittest.TestCase):
 def test_gate_only(self):
  u=Net(); x=torch.ones(2,3); c=torch.tensor(2.); holder={'condition':c}
  with temporary_cpfm_enabled(u,None,holder,Hooks,False): off=u.down_blocks[0].resnets[0](x)
  with temporary_cpfm_enabled(u,None,holder,Hooks,True): on=u.down_blocks[0].resnets[0](x)
  self.assertTrue(torch.equal(off,x)); self.assertTrue(torch.equal(on,x+c)); self.assertTrue(torch.equal(holder['condition'],c)); self.assertEqual(hook_count(u),0)
 def test_exception_removal(self):
  u=Net(); holder={'condition':torch.tensor(1.)}
  with self.assertRaises(ValueError):
   with temporary_cpfm_enabled(u,None,holder,Hooks,True): raise ValueError('intentional anonymous failure')
  self.assertEqual(hook_count(u),0)
