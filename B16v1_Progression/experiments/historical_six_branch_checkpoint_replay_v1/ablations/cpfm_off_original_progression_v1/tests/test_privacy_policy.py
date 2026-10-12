import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
class PrivacyTest(unittest.TestCase):
 def test_no_private_binary(self):
  root=Path(__file__).resolve().parents[1]; self.assertFalse(any(p.suffix in {'.pt','.pth','.ckpt','.safetensors','.parquet','.sqlite'} for p in root.rglob('*')))
 def test_only_nonmedical_png(self):
  root=Path(__file__).resolve().parents[1]; self.assertTrue(all(p.name.startswith('Fig0') and 'PRIVATE' not in p.name for p in root.rglob('*.png')))
