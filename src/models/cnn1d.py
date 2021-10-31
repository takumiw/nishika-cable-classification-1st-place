import os
from logging import getLogger
from typing import Any, Dict, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Conv1D, Dense, Dropout, Flatten
from tensorflow.keras.models import Model, Sequential

from src.keras_callbacks import create_callback
from src.utils_tensorflow import plot_learning_history, plot_model

logger = getLogger(__name__)


def train_and_predict(
    output_dir: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    hparams: Dict[str, Union[int, Dict[str, Union[str, float]]]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Model]:
    """Train 1D-CNN
    Args:
    Returns:
        prob_train (np.ndarray): probability of train prediction
        prob_valid (np.ndarray): probability of valid prediction
        prob_test (np.ndarray): probability of test prediction
        model (Model): trained model
    """
    model = build_model(hparams=hparams, input_shape=X_train.shape[1:])
    plot_model(model, path=os.path.join(output_dir, "model.png"))

    callbacks = create_callback(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        epochs=hparams["epochs"],
        path_history=output_dir,
        n_classes=hparams["num_class"],
    )

    fit = model.fit(
        X_train,
        y_train,
        batch_size=hparams["batch_size"],
        epochs=hparams["epochs"],
        verbose=hparams["verbose"],
        validation_data=(X_valid, y_valid),
        callbacks=callbacks,
    )

    plot_learning_history(fit=fit, path=os.path.join(output_dir, "history.png"))
    model = keras.models.load_model(os.path.join(output_dir, "best_loss_model.h5"))

    prob_train = model.predict(X_train)
    prob_valid = model.predict(X_valid)
    prob_test = model.predict(X_test)

    K.clear_session()

    return prob_train, prob_valid, prob_test, model


def build_model(hparams: Dict[str, Any], input_shape: Tuple[int, int] = (5, 15)) -> Model:
    """
    Args:
        input_shape (tuple[int, int], default=(5, 15)): (n_models, n_classes)
    Returns:
        model (Model)
    """
    model = Sequential()
    model.add(Conv1D(512, kernel_size=3, strides=1, padding="valid", input_shape=input_shape))  # (3, 512)
    model.add(Activation(tf.nn.silu))

    model.add(Conv1D(1024, kernel_size=3, strides=1, padding="valid"))  # (1, 1024)
    model.add(Activation(tf.nn.silu))

    model.add(Flatten())  # (1024,)
    model.add(Dense(512))  # (512,)
    model.add(Activation(tf.nn.silu))
    model.add(Dropout(0.3, seed=0))
    model.add(Dense(hparams["num_class"]))  # (15,)
    model.add(Activation("softmax"))

    model.compile(
        loss="SparseCategoricalCrossentropy",
        optimizer=tfa.optimizers.AdamW(
            weight_decay=hparams["weight_decay"],
            learning_rate=hparams["learning_rate"],
            epsilon=1e-8,
        ),
        metrics=["accuracy"],
    )
    return model
