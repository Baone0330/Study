def safe_checkpoint(path, expected):
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Unsupported historical checkpoint schema; no unsafe fallback")
    if (int(checkpoint["seed"]) != int(expected["seed"])
            or int(checkpoint["epoch"]) != int(expected["best_epoch"])
            or checkpoint["config"]["branch"] != expected["original_branch_name"]):
        raise ValueError("Checkpoint identity differs from original selected run")
    state = checkpoint["model_state_dict"]
    if not any(key.startswith("core.feature_extractor.") for key in state):
        raise ValueError("Incomplete backbone state")
    for tensor in state.values():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError("Non-finite checkpoint state")
    return checkpoint
