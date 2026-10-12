import runpy, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
runpy.run_module('aggregate_paired_val_results',run_name='__main__')
