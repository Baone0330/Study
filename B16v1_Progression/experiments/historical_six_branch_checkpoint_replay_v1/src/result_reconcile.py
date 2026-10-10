METRICS = ("ROC_AUC", "PR_AUC", "Accuracy", "Sensitivity", "Specificity", "F1", "Val_N")


def reconcile(branch, actual, reference):
    output = dict(branch_id=branch, actual_val_N=actual["Val_N"])
    matches = []
    for metric in METRICS:
        value, old = float(actual[metric]), float(reference[metric])
        output["actual_" + metric] = value
        output["historical_" + metric] = old
        output["diff_" + metric] = value - old
        match = f"{value:.6f}" == f"{old:.6f}"
        output["display_match_" + metric] = match
        matches.append(match)
    output["all_display_fields_match"] = all(matches)
    output["replay_status"] = "CHECKPOINT_FORWARD_DISPLAY_MATCHED" if all(matches) else "CHECKPOINT_FORWARD_NUMERIC_DIFFERENCE"
    return output
