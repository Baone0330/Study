import ast,sys
from pathlib import Path
p=Path(__file__).resolve().parents[1]
[ast.parse(f.read_text(encoding='utf-8')) for f in p.rglob('*.py')]
assert not any(f.suffix in {'.pt','.pth','.ckpt','.safetensors','.parquet','.sqlite'} for f in p.rglob('*'))
print('Public source syntax and forbidden binary extensions PASS')
