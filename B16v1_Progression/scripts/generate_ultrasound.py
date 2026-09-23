from __future__ import annotations
import argparse
import json
from pathlib import Path
from b16v1_progression.pipeline import load_runtime_stack, make_condition, generate_one

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--medical_lora", required=True)
    p.add_argument("--longitudinal_ckpt", required=True)
    p.add_argument("--cpfm_ckpt", required=True)
    p.add_argument("--progression_ckpt", required=True)
    p.add_argument("--history_json", type=Path, required=True)
    p.add_argument("--query_date", required=True)
    p.add_argument("--annual_index", type=int, required=True)
    p.add_argument("--output_png", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    history = json.loads(args.history_json.read_text(encoding="utf-8"))["history"]
    stack = load_runtime_stack(args.base_model, args.medical_lora, args.longitudinal_ckpt,
                               args.cpfm_ckpt, args.progression_ckpt)
    try:
        condition = make_condition(stack, history, args.query_date, args.annual_index)
        generate_one(stack, condition, args.output_png, args.seed)
    finally:
        stack["hooks"].close()

if __name__ == "__main__":
    main()
