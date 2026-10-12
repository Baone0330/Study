import runpy, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
runpy.run_module('freeze_route',run_name='__main__')
