from typing import Mapping

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

