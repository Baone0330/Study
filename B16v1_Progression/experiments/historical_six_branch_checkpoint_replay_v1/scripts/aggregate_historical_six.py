import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.replay_workflow import main

if __name__ == "__main__":
    if "--help" not in sys.argv and "-h" not in sys.argv and "--aggregate" not in sys.argv:
        sys.argv.append("--aggregate")
    raise SystemExit(main())
