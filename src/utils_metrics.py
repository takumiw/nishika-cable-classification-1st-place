from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss


def calc_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None, metrics: List[str] = ["loss"]
) -> Dict[str, float]:
    result = {}
    for metric in metrics:
        if metric == "loss":
            result["loss"] = log_loss(y_true, y_prob)
        elif metric == "accuracy":
            result["accuracy"] = accuracy_score(y_true, y_pred)
        elif metric == "f1":
            result["f1"] = f1_score(y_true, y_pred, average="micro")
        else:
            raise NameError(f"metric {metric} is not defined")
    return result
