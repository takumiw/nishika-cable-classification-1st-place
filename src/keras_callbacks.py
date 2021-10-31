"""Collections of callbacks for TensorFlow/Keras"""
import os
from logging import getLogger
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss
from tensorflow.keras.callbacks import (Callback, EarlyStopping, LearningRateScheduler, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.models import Model

logger = getLogger(__name__)


class MetricCallback(Callback):
    """Plot logloss, accuracy, f1-score every epoch"""

    def __init__(
        self,
        model: Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        n_classes: int,
        save_csv: bool,
        save_png: bool,
        path_history: str = "path/to/history",
    ) -> None:
        self.history = pd.DataFrame(
            columns=[
                "epoch",
                "loss",
                "val_loss",
                "acc",
                "val_acc",
                "f1",
                "val_f1",
            ]
        )
        self.path_history = path_history
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.n_classes = n_classes
        self.save_csv = save_csv
        self.save_png = save_png
        self.best_valid_acc = 0.0
        self.best_acc_epoch = 0
        self.best_valid_fscore = 0.0
        self.best_fscore_epoch = 0
        self.best_valid_loss = 10.0 ** 10
        self.best_loss_epoch = 0

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
        # [Note] model.evaluate induces a bug in TF 2.3 when labels are sparse
        # https://github.com/tensorflow/tensorflow/issues/42045#issuecomment-674232499
        train_pred = self.model.predict(self.X_train)
        valid_pred = self.model.predict(self.X_valid)

        train_acc = accuracy_score(self.y_train, train_pred.argmax(axis=1))
        valid_acc = accuracy_score(self.y_valid, valid_pred.argmax(axis=1))

        arg_labels = [i for i in range(self.n_classes)]
        train_loss = log_loss(self.y_train, train_pred, labels=arg_labels)
        valid_loss = log_loss(self.y_valid, valid_pred, labels=arg_labels)

        train_f1 = f1_score(self.y_train, train_pred.argmax(axis=1), average="macro")
        valid_f1 = f1_score(self.y_valid, valid_pred.argmax(axis=1), average="macro")

        self.history.loc[epoch] = [
            epoch,
            train_loss,
            valid_loss,
            train_acc,
            valid_acc,
            train_f1,
            valid_f1,
        ]

        if valid_acc > self.best_valid_acc:
            self.best_valid_acc = valid_acc
            self.best_acc_epoch = epoch
            self.model.save(os.path.join(self.path_history, "best_acc_model.h5"))

        if valid_f1 > self.best_valid_fscore:
            self.best_valid_fscore = valid_f1
            self.best_fscore_epoch = epoch
            self.model.save(os.path.join(self.path_history, "best_f1_model.h5"))

        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            self.best_loss_epoch = epoch
            # self.model.save(os.path.join(self.path_history, "best_loss_model.h5"))

        if self.save_png:
            fig, (axL, axM, axR) = plt.subplots(ncols=3, figsize=(13, 4))
            axL.plot(self.history["loss"], label="train")
            axL.plot(self.history["val_loss"], label="validation")
            axL.set_title("Loss")
            axL.set_xlabel("epoch")
            axL.set_ylabel("loss")
            axL.legend(loc="upper right")

            axM.plot(self.history["acc"], label="train")
            axM.plot(self.history["val_acc"], label="validation")
            axM.set_title("Accuracy")
            axM.set_xlabel("epoch")
            axM.set_ylabel("accuracy")
            axM.legend(loc="lower right")

            axR.plot(self.history["f1"], label="train")
            axR.plot(self.history["val_f1"], label="validation")
            axR.set_title("F1-score")
            axR.set_xlabel("epoch")
            axR.set_ylabel("f1-score")
            axR.legend(loc="lower right")

            plt.tight_layout()
            fig.savefig(os.path.join(self.path_history, "history.png"))
            plt.close()

    def on_train_end(self, logs: Optional[Dict[str, float]] = None) -> None:
        if self.save_csv:
            self.history.to_csv(os.path.join(self.path_history, "history.csv"), header=True, index=True)
        logger.debug(f"Best accuracy on validation is {self.best_valid_acc} on Epoch {self.best_acc_epoch}")
        logger.debug(f"Best f1-score on validation is {self.best_valid_fscore} on Epoch {self.best_fscore_epoch}")
        logger.debug(f"Best logloss on validation is {self.best_valid_loss} on Epoch {self.best_loss_epoch}")


class PeriodicLogger(Callback):
    """Logging history every n epochs"""

    def __init__(self, metric: str = "accuracy", verbose: int = 1, epochs: Optional[int] = None) -> None:
        self.metric = metric
        self.verbose = verbose
        self.epochs = epochs

    def on_epoch_end(self, epoch: int, logs: Dict[str, float]) -> None:
        epoch += 1
        if epoch % self.verbose == 0:
            msg = " - ".join(
                [
                    f"Epoch {epoch}/{self.epochs}",
                    f"loss: {logs['loss']:.06f}",
                    f"{self.metric}: {logs[self.metric]:.06f}",
                    f"val_loss: {logs['val_loss']:.06f}",
                    f"val_{self.metric}: {logs[f'val_{self.metric}']:.06f}",
                ]
            )
            logger.debug(msg)


def lr_scheduler(epoch):
    lr = 1e-3
    if epoch >= 15:
        lr = 4e-4
    if epoch >= 30:
        lr = 1e-4
    if epoch >= 45:
        lr = 4e-5
    if epoch >= 60:
        lr = 1e-5
    return lr


def create_callback(
    model: Model,
    X_train: Union[np.ndarray, List[np.ndarray]],
    y_train: np.ndarray,
    X_valid: Union[np.ndarray, List[np.ndarray]],
    y_valid: np.ndarray,
    save_best_only: bool = True,
    patience: int = 30,
    metric: str = "accuracy",
    verbose: int = 10,
    epochs: Optional[int] = None,
    path_history: str = "path/to/history",
    n_classes: int = 4,
    save_csv: bool = True,
    save_png: bool = True,
) -> List[Any]:
    """callback settings
    Args:
        model (Model): built model
    Returns:
        callbacks (List[Any]): List of Callback
    """
    callbacks = []
    callbacks.append(
        MetricCallback(
            model=model,
            path_history=path_history,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            n_classes=n_classes,
            save_csv=save_csv,
            save_png=save_png,
        )
    )
    callbacks.append(EarlyStopping(monitor="val_loss", min_delta=0, patience=patience, verbose=1, mode="min"))
    callbacks.append(
        ModelCheckpoint(filepath=os.path.join(path_history, "best_loss_model.h5"), save_best_only=save_best_only)
    )
    callbacks.append(PeriodicLogger(metric=metric, verbose=verbose, epochs=epochs))
    callbacks.append(ReduceLROnPlateau(factor=0.5, patience=10, verbose=1, min_lr=0.00001))
    # callbacks.append(LearningRateScheduler(lr_scheduler, verbose=1))
    return callbacks
