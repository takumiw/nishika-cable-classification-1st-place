import os
from logging import getLogger
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import KFold
from tqdm import tqdm

from src.utils import round

logger = getLogger(__name__)


class WeightOptimization:
    def __init__(
        self,
        output_dir: str,
        y_true: np.ndarray,
        prob_oofs: np.ndarray,
        prob_tests: np.ndarray,
        method: str = "Nelder-Mead",
        n_folds: int = 5,
        n_loops: int = 100,
    ):
        """
        Args:
            y_true (np.ndarray): (n_samples,)
            prob_oofs (np.ndarray): (n_samples, n_classes, n_models)
            method (str): chosen from Nelder-Mead, L-BFGS-B, SLSQP
        """
        self.output_dir = output_dir
        self.y_true = y_true
        self.prob_oof = prob_oofs
        self.prob_test = prob_tests
        self.method = method
        self.n_models = prob_oofs.shape[2]
        self.n_classes = prob_oofs.shape[1]
        self.n_folds = n_folds
        self.n_loops = n_loops

    def _objective(self, weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        n_models = X.shape[2]
        n_classes = X.shape[1]
        n_samples = X.shape[0]
        prob_wo = np.zeros((n_samples, n_classes))
        for i in range(n_models):
            prob_wo += X[:, :, i].reshape(n_samples, n_classes) * weights[i]
        return log_loss(y, prob_wo)

    def _evaluate(
        self, train_idxs: np.ndarray, valid_idxs: np.ndarray, weights: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """
        Returns:
            tuple[float, float, float, float]: train_loss, train_accuracy, valid_loss, valid_accuracy
        """
        prob_train = self.prob_oof[train_idxs, :, :]
        prob_valid = self.prob_oof[valid_idxs, :, :]
        prob_train = np.average(prob_train, axis=-1, weights=weights)
        prob_valid = np.average(prob_valid, axis=-1, weights=weights)

        y_train = self.y_true[train_idxs]
        y_valid = self.y_true[valid_idxs]

        train_loss = log_loss(y_train, prob_train)
        valid_loss = log_loss(y_valid, prob_valid)
        train_acc = accuracy_score(y_train, prob_train.argmax(axis=-1))
        valid_acc = accuracy_score(y_valid, prob_valid.argmax(axis=-1))
        return train_loss, train_acc, valid_loss, valid_acc

    def _run_kfold(self, seed: int = 0) -> List[np.ndarray]:
        """
        Returns:
            list[np.ndarray]: weights (n_folds, n_classes)
        """
        weights = []
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=seed)
        for fold, (train_idxs, valid_idxs) in enumerate(kf.split(self.prob_oof)):
            # w0 = np.ones(self.n_models) / self.n_models
            np.random.seed(int(f"{seed}{fold}"))
            w0 = np.random.random_sample(self.n_models)
            w0 = w0 / sum(w0)
            # logger.info(f"initial weights: {round(w0)}")
            bounds = [(0, 1)] * self.n_models

            res = minimize(
                fun=self._objective,
                x0=w0,
                args=(self.prob_oof[train_idxs], self.y_true[train_idxs]),
                method=self.method,
                bounds=bounds,
                tol=1e-6,
                options={"maxiter": 1e6},
            )
            w_opt = res.x / sum(res.x)
            logger.info(f"optimized weights: {round(w_opt)}")
            train_loss, train_acc, valid_loss, valid_acc = self._evaluate(train_idxs, valid_idxs, w_opt)
            logger.info(f"{train_loss=:.06f} -- {train_acc=:.06f} -- {valid_loss=:.06f} -- {valid_acc=:.06f}")
            weights.append(w_opt)

        return weights

    def run_loop(self):
        weights = []
        for i in tqdm(range(self.n_loops)):
            weights.extend(self._run_kfold(i))

        weights = np.array(weights)
        weight = weights.mean(axis=0)
        weight = weight / sum(weight)
        logger.info(f"FINAL Optimized Weights: {weight}")

        pd.DataFrame(weights, columns=[f"model{i}" for i in range(self.n_models)]).to_csv(
            os.path.join(self.output_dir, "weights.csv"), index=False
        )
        np.save(os.path.join(self.output_dir, "best_weight.npy"), weight)
        self.opt_weight = weight

    def calc_weight_averaging(self):
        prob_oof = np.average(self.prob_oof, axis=-1, weights=self.opt_weight)
        prob_test = np.average(self.prob_test, axis=-1, weights=self.opt_weight)
        return prob_oof, prob_test
