"""Public, identity-free structural contract checks."""


def validate_seed_pairs(rows, expected_count=6):
    if len(rows) != expected_count:
        raise ValueError("Incomplete seed support")
    keys = {(r["node_key"], int(r["image_index"])) for r in rows}
    if len(keys) != len(rows):
        raise ValueError("Duplicate node/image index")
    if sorted(int(r["image_index"]) for r in rows) != list(range(expected_count)):
        raise ValueError("Image index coverage differs")
    if any(int(r["on_seed"]) != int(r["off_seed"]) for r in rows):
        raise ValueError("ON/OFF seeds differ")
    return True


def validate_runtime_contract(config):
    expected = {"requested_inference_steps": 30, "guidance_scale": 7.0, "resolution": [512, 512], "images_per_generated_node": 6,
                "dtype": "fp16", "lora_override": "NONE", "original_generation_batch": 6}
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError("Frozen sampling contract differs: " + name)
    return True
