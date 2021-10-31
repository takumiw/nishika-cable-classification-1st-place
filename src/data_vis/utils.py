from pathlib import PosixPath
from typing import List, Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def read_cv_image(path: Union[str, PosixPath]) -> np.ndarray:
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def plot_confusion_matrix(
    trues: List[np.ndarray],
    preds: List[np.ndarray],
    phases: List[str] = ["train"],
    labels: Optional[List[str]] = None,
    path: str = "path/to/confusion_matrix.png",
    normalize: Optional[str] = None,
) -> None:
    """
    Args:
        phases (List[str]): phases of input, e.g. "train", "valid", "oof"
    """
    assert len(trues) == len(preds) == len(phases), "length does not match"

    fig, ax = plt.subplots(ncols=len(phases), figsize=(9 * len(phases), 10))
    if not isinstance(ax, np.ndarray):
        ax = [ax]
    for true, pred, phase, ax_ in zip(trues, preds, phases, ax):
        cm = confusion_matrix(true, pred, normalize=normalize)
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap="Blues",
            square=True,
            vmin=0 if normalize else None,
            vmax=1.0 if normalize else None,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax_,
        )
        ax_.set_xlabel("Predicted label")
        ax_.set_xticklabels(labels, rotation=70)
        ax_.set_ylabel("True label")
        ax_.set_title(f"Confusion matrix - {phase}")

    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close()
