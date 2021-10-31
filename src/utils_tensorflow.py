import os

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Model


def plot_model(model: Model, path: str) -> None:
    if not os.path.isfile(path):
        keras.utils.plot_model(model, to_file=path, show_shapes=True)


def plot_learning_history(fit, metric: str = "accuracy", path: str = "history.png") -> None:
    """Plot learning curve
    Args:
        fit (Any): History object
        path (str, default="history.png")
    """
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 4))
    axL.plot(fit.history["loss"], label="train")
    axL.plot(fit.history["val_loss"], label="validation")
    axL.set_title("Loss")
    axL.set_xlabel("epoch")
    axL.set_ylabel("loss")
    axL.legend(loc="upper right")

    axR.plot(fit.history[metric], label="train")
    axR.plot(fit.history[f"val_{metric}"], label="validation")
    axR.set_title(metric.capitalize())
    axR.set_xlabel("epoch")
    axR.set_ylabel(metric)
    axR.legend(loc="best")

    fig.savefig(path)
    plt.close()
