from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List

import numpy as np


def round_float(f: float, r: float = 0.000001) -> float:
    return float(Decimal(str(f)).quantize(Decimal(str(r)), rounding=ROUND_HALF_UP))


def round_list(l: List[float], r: float = 0.000001) -> List[float]:
    return [round_float(f, r) for f in l]


def round_dict(d: Dict[Any, Any], r: float = 0.000001) -> Dict[Any, Any]:
    return {key: round(d[key], r) for key in d.keys()}


def round(arg: Any, r: float = 0.000001) -> Any:
    if type(arg) == float or type(arg) == np.float64 or type(arg) == np.float32:
        return round_float(arg, r)
    elif type(arg) == list or type(arg) == np.ndarray:
        return round_list(arg, r)
    elif type(arg) == dict:
        return round_dict(arg, r)
    else:
        raise ValueError(f"Arg type {type(arg)} is not supported")
