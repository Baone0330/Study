import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from branch_contract import validate_seed_pairs
class SeedTest(unittest.TestCase):
 def test_exact(self):
  r=[dict(node_key='ANON',image_index=i,on_seed=i+20,off_seed=i+20) for i in range(6)]; self.assertTrue(validate_seed_pairs(r))
 def test_duplicate(self):
  r=[dict(node_key='ANON',image_index=0,on_seed=20,off_seed=20)]*6
  with self.assertRaises(ValueError): validate_seed_pairs(r)
