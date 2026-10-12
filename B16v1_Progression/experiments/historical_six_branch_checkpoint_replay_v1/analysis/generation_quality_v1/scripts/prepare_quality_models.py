"""Task-local public metric weights; no shared-environment update or image transfer."""
from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import urllib.request

from b16_quality_common import atomic_json, now, read_json, setup, stat_record


def offline_lpips(root, device):
    import torch
    import torchvision.models as tv
    import lpips
    weights = Path(torch.hub.get_dir()) / "checkpoints/alexnet-owt-7be5be79.pth"
    assert weights.is_file(), "Offline official AlexNet state missing"
    model_state = torch.load(weights, map_location="cpu", weights_only=True)
    original = tv.alexnet
    def builder(*args, **kwargs):
        kwargs.pop("pretrained", None); kwargs.pop("weights", None)
        model = original(weights=None, **kwargs)
        model.load_state_dict(model_state, strict=True)
        return model
    tv.alexnet = builder
    calibration = root / "vendor/lpips/weights/v0.1/alex.pth"
    assert calibration.is_file()
    try:
        model = lpips.LPIPS(net="alex", version="0.1", model_path=str(calibration),
                            pnet_rand=False, pnet_tune=False, pretrained=True, verbose=False)
    finally:
        tv.alexnet = original
    return model.to(device).eval().requires_grad_(False), weights, calibration


def main(output_root):
    root = setup(output_root)
    import torch
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3, URL_INCEPTION_V3
    assert (root / "protocol/FROZEN_QUALITY_EVALUATION_CONTRACT_V1.json").exists()
    target = root / "models/inception_v3_fidelity_compat.pth"
    if not target.exists():
        part = target.with_suffix(".part")
        with urllib.request.urlopen(URL_INCEPTION_V3, timeout=60) as response, part.open("wb") as f:
            while True:
                chunk = response.read(1024*1024)
                if not chunk: break
                f.write(chunk)
        # Metadata and strict tensor loading only; no file digests/checksums.
        state = torch.load(part, map_location="cpu", weights_only=True)
        assert isinstance(state, dict) and all(torch.isfinite(t).all() for t in state.values())
        os.replace(part, target)
    model = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"],
                                       feature_extractor_weights_path=str(target), feature_extractor_internal_dtype="float32")
    assert all(not p.requires_grad for p in model.parameters())
    perceptual, alex_path, calibration = offline_lpips(root, "cpu")
    assert all(not p.requires_grad for p in perceptual.parameters())
    contract = {"status": "WEIGHTS_READY_CPU_STRICT_LOAD_PASS", "encoder": "torch-fidelity inception-v3-compat",
                "torch_fidelity_version": importlib.metadata.version("torch-fidelity"), "feature_dim": 2048,
                "feature_dtype": "float32", "feature_extractor_weights_path": str(target), "weights_source_url": URL_INCEPTION_V3,
                "inception_weight_metadata": stat_record(target), "alexnet_weight_metadata": stat_record(alex_path),
                "lpips_calibration_metadata": stat_record(calibration), "lpips_package_version": importlib.metadata.version("lpips"),
                "lpips_network": "AlexNet", "lpips_model_version": "0.1", "lpips_input": "RGB native512 float32, uint8/127.5-1",
                "SSIM_package_version": importlib.metadata.version("scikit-image"), "new_training_steps": 0,
                "new_SD_forward": 0, "clinical_pixels_read_at_preparation": 0, "global_environment_changed": False,
                "created_utc": now()}
    atomic_json(root / "image_features/encoder_contract.json", contract)
    print({k: contract[k] for k in ("status", "torch_fidelity_version", "feature_dim", "lpips_package_version", "SSIM_package_version")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    a = parser.parse_args()
    main(a.output_root)
