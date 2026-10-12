import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from downstream_metric_utils import paired_patient_bootstrap
class BootstrapTest(unittest.TestCase):
 def test_identical_zero(self):
  y=np.array([0,0,1,1]); p=np.array([.1,.3,.6,.9]); r=paired_patient_bootstrap(y,p,p,20); self.assertTrue(all(x['ci_lower_95']==x['ci_upper_95']==x['point_difference']==0 for x in r))
 def test_same_seed(self):
  y=np.array([0,0,1,1]); p=np.array([.1,.3,.6,.9]); self.assertEqual(paired_patient_bootstrap(y,p,1-p,20),paired_patient_bootstrap(y,p,1-p,20))
