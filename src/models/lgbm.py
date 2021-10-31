import os
import pickle
from logging import getLogger
from typing import Dict, Tuple, Union

import lightgbm as lgb
import numpy as np
from lightgbm import Booster, early_stopping

from src import log_evaluation

logger = getLogger(__name__)


def train_and_predict(
    output_dir: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    hparams: Dict[str, Union[int, Dict[str, Union[str, float]]]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Booster]:
    """Train LightGBM
    Args:
    Returns:
        pred_train (np.ndarray): probability of train prediction
        pred_valid (np.ndarray): probability of valid prediction
        pred_test (np.ndarray): probability of test prediction
        model (Booster): trained model
    """
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
    callbacks = [
        early_stopping(stopping_rounds=hparams["early_stopping_rounds"]),
        log_evaluation(logger, period=hparams["verbose_periods"]),
    ]

    model = lgb.train(
        dict(hparams["params"]),
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        feval=eval_accuracy if hparams["metric"] == "accuracy" else None,
        num_boost_round=50000,
        callbacks=callbacks,
    )

    with open(os.path.join(output_dir, "model_best_loss.pickle"), mode="wb") as f:
        pickle.dump(model, f)

    logger.debug(f"best iteration: {model.best_iteration}")
    logger.debug(f'training best score: {model.best_score["training"].items()}')
    logger.debug(f'valid_1 best score: {model.best_score["valid_1"].items()}')

    prob_train = model.predict(X_train)
    prob_valid = model.predict(X_valid)
    prob_test = model.predict(X_test)

    return prob_train, prob_valid, prob_test, model


def eval_accuracy(preds: np.ndarray, data: lgb.Dataset) -> Tuple[str, float, bool]:
    n_classes = 15
    y_true = data.get_label()
    reshaped_preds = preds.reshape(n_classes, len(preds) // n_classes)
    y_pred = np.argmax(reshaped_preds, axis=0)
    acc = np.mean(y_true == y_pred)
    return "accuracy", acc, True
