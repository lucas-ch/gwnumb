from typing import Mapping


from numb_project.constants import *

def get_from_dict_or_val(
    val: int | Mapping[str, int], key: str, log: str
) -> int:
    """
    If val is int, return val, otherwise return val[key]
    """
    if isinstance(val, int):
        return val

    assert key in val, f"{key} should be defined in {log}."
    return val[key]

def merge_metrics(metrics: dict[str, torch.Tensor], other_metrics: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    merged = dict(metrics)
    for key, value in other_metrics.items():
        merged[key] = merged[key] + value if key in merged else value
    return merged
