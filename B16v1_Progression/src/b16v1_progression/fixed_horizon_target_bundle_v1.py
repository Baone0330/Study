"""Experimental fixed-horizon endpoint construction; the default remains H1.

This is tensor-only model/training integration, not a training launcher.
The caller provides a screened historical prefix and a supervision-only endpoint.
No file access, patient identifiers, labels, or clinical paths are accepted.
"""
from datetime import date
import torch


@torch.no_grad()
def construct_endpoint_bundle(frozen, *, before, prefix, prefix_dates,
                              endpoint_feature, endpoint_date,
                              horizon=1, padded_deltas=5):
    """Rebuild query, teacher, compatibility and delta for ONE horizon.

    Prefix is F_0..F_k, never F_(k+1) in an H2 student's inputs.
    Endpoint feature is supervision-only. Both teacher and delta use it.
    The existing query encoder uses the zero-based endpoint index as its
    annual_index metadata; this preserves the actual H1 training semantics.
    Compatibility uses the last visible feature, not the future endpoint.
    """
    if horizon not in (1, 2):
        raise ValueError('Only fixed H1 or experimental H2 is supported')
    prefix = torch.as_tensor(prefix, dtype=torch.float32)
    before = torch.as_tensor(before, dtype=torch.float32)
    endpoint_feature = torch.as_tensor(endpoint_feature, dtype=torch.float32)
    if prefix.ndim != 2 or prefix.shape[1] != 16 or len(prefix) != len(prefix_dates):
        raise ValueError('Expected ordered prefix [n,16] with matching dates')
    if before.shape != (16,) or endpoint_feature.shape != (16,):
        raise ValueError('Expected before and supervision endpoint [16]')
    dates = [date.fromisoformat(str(d)[:10]) for d in prefix_dates]
    endpoint = date.fromisoformat(str(endpoint_date)[:10])
    if not dates or any(a >= b for a, b in zip(dates, dates[1:])) or endpoint <= dates[-1]:
        raise ValueError('Endpoint must follow the strictly ordered visible prefix')
    k = len(prefix)-1
    if k > padded_deltas:
        raise ValueError('Visible deltas exceed the frozen padding contract')
    task = dict(query_date=str(endpoint_date), anchor_date=str(prefix_dates[0]),
                before_exam_date=str(prefix_dates[k-1 if k else k]),
                after_exam_date=str(prefix_dates[k]), annual_index=k+horizon)
    query = frozen.query_encoder(task)[0]
    after = prefix[-1]
    compatibility = frozen.bridge(before[None], after[None], query[None])['condition_vector'][0]
    teacher = frozen.bridge(before[None], endpoint_feature[None], query[None])['condition_vector'][0]
    deltas = torch.zeros(padded_deltas, 16, dtype=torch.float32)
    mask = torch.zeros(padded_deltas, dtype=torch.float32)
    if k:
        deltas[:k] = torch.diff(prefix, dim=0)
        mask[:k] = 1
    return dict(before=before, after=after, temporal=query, deltas=deltas, mask=mask,
                target=endpoint_feature-after, compat=compatibility, teacher=teacher)


def balanced_h12_values(progression, h1_target, h2_target):
    """Neutral, non-training selection values; aggregate within patient first.

    Numerical zero-P cosine is zero, consistent with the original protocol.
    This helper neither backpropagates nor introduces a multi-horizon loss.
    """
    with torch.no_grad():
        return .5*(torch.nn.functional.cosine_similarity(progression, h1_target, dim=-1)
                   +torch.nn.functional.cosine_similarity(progression, h2_target, dim=-1))
