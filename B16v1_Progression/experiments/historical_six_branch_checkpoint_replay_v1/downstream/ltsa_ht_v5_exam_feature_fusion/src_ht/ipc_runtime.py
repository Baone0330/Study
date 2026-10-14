"""Non-numerical Linux IPC adaptation: relative path inside the long V5 root."""
import multiprocessing
import multiprocessing.util
import os
from pathlib import Path
def ensure_short_ipc(root):
    root=Path(root).resolve();assert Path.cwd().resolve()==root
    relative=Path('cache/ipc')/('p'+str(os.getpid()));absolute=(root/relative).resolve()
    assert absolute.is_relative_to(root/'cache/ipc')
    absolute.mkdir(parents=True,exist_ok=True,mode=0o700)
    # Python3.10 get_temp_dir returns this config unchanged. Workers inherit both config and cwd.
    multiprocessing.current_process()._config['tempdir']=str(relative)
    actual=multiprocessing.util.get_temp_dir();assert Path(actual).resolve()==absolute and len(actual)<40
