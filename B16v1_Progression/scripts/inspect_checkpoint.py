"""Print checkpoint metadata keys only, never tensor contents."""
from __future__ import annotations
import argparse
import json
import torch

def main():
    p=argparse.ArgumentParser()
    p.add_argument("checkpoint")
    a=p.parse_args()
    value=torch.load(a.checkpoint,map_location="cpu",weights_only=True)
    print(json.dumps({"top_level_keys":list(value) if isinstance(value,dict) else [],
                      "model_tensor_count":len(value.get("model_state_dict",{})) if isinstance(value,dict) else 0}))

if __name__ == "__main__":
    main()
