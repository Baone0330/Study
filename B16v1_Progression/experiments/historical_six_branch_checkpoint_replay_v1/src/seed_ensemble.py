import numpy as np


def align_and_average(frames, expected_seeds):
    """Join by unique patient key; never assume stored row positions align."""
    if set(frames) != set(expected_seeds) or len(expected_seeds) != 3:
        raise ValueError("Incomplete original three-seed ensemble")
    aligned, labels, ids = [], None, None
    for seed in expected_seeds:
        frame = frames[seed]
        if frame.patient_id.duplicated().any():
            raise ValueError("Duplicate patient rows in seed output")
        frame = frame.assign(patient_id=frame.patient_id.astype(str)).sort_values("patient_id").reset_index(drop=True)
        if ids is None:
            ids, labels = frame.patient_id.tolist(), frame.label.to_numpy(int)
        elif frame.patient_id.tolist() != ids or not np.array_equal(frame.label.to_numpy(int), labels):
            raise ValueError("Seed patient identity/labels disagree")
        probability = frame.prob_ht1.to_numpy(float)
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise ValueError("Invalid seed probability")
        aligned.append(probability)
    return ids, labels, np.mean(np.stack(aligned), axis=0)
