import os
from logging import getLogger
from typing import Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.models import cnn1d, cnn2d, lgbm
from src.models.weight_optimization import WeightOptimization

logger = getLogger(__name__)

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(CUR_DIR, "../")
LOG_DIR = os.path.join(ROOT_DIR, "logs/")


class Ensemble:

    n_classes = 15

    def __init__(self, output_dir, cfg: DictConfig):
        self.output_dir = output_dir
        self.method = cfg.method
        self.inputs = cfg.inputs
        self.cfg = cfg

        assert self.method in ["averaging", "weight_optimization", "stacking"], "Method not implemented"

        self.oof = pd.read_csv(os.path.join(ROOT_DIR, "logs", self.inputs[0], "oof.csv"))
        self.oof["pred"] = -1
        self.sub = pd.read_csv(os.path.join(ROOT_DIR, "input", "sample_submission.csv"))

    def averaging(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """averaging probabilities"""

        prob_oofs = []
        prob_tests = []
        # モデルごとに実行
        for input_id in self.inputs:
            # load oof
            prob_valid = np.load(os.path.join(LOG_DIR, input_id, "prob_oof.npy"))
            assert prob_valid.shape[0] == len(self.oof), "Length does not match"
            assert prob_valid.shape[1] == self.__class__.n_classes, "Length does not match"
            prob_oofs.append(prob_valid)

            # load test
            prob_test = np.load(os.path.join(LOG_DIR, input_id, "prob_test_avg.npy"))
            assert prob_test.shape[0] == len(self.sub), "Length does not match"
            assert prob_test.shape[1] == self.__class__.n_classes, "Length does not match"
            prob_tests.append(prob_test)

        prob_oofs = np.array(prob_oofs)
        prob_tests = np.array(prob_tests)

        # averaging
        prob_oof_ave = np.mean(prob_oofs, axis=0)
        pred_oof = prob_oof_ave.argmax(axis=-1)

        prob_test_ave = np.mean(prob_tests, axis=0)
        pred_test = prob_test_ave.argmax(axis=-1)

        return prob_oof_ave, pred_oof, prob_test_ave, pred_test

    def weight_optimize(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        prob_oofs = []
        prob_tests = []
        # モデルごとに実行
        for input_id in self.inputs:
            # load oof
            prob_valid = np.load(os.path.join(LOG_DIR, input_id, "prob_oof.npy"))
            assert prob_valid.shape[0] == len(self.oof), "Length does not match"
            assert prob_valid.shape[1] == self.__class__.n_classes, "Length does not match"
            prob_oofs.append(prob_valid)

            # load test
            prob_test = np.load(os.path.join(LOG_DIR, input_id, "prob_test_avg.npy"))
            assert prob_test.shape[0] == len(self.sub), "Length does not match"
            assert prob_test.shape[1] == self.__class__.n_classes, "Length does not match"
            prob_tests.append(prob_test)

        prob_oofs = np.array(prob_oofs)
        prob_tests = np.array(prob_tests)
        logger.info(f"{prob_oofs.shape=}, {prob_tests.shape=}")

        # Transpose (n_models, n_samples, n_classes) -> (n_samples, n_classes, n_models)
        prob_oofs = prob_oofs.transpose((1, 2, 0))
        prob_tests = prob_tests.transpose((1, 2, 0))
        logger.info(f"Transposed to {prob_oofs.shape=}, {prob_tests.shape=}")

        # weight optimization
        wgt_opt = WeightOptimization(
            output_dir=self.output_dir,
            y_true=self.oof.target.values.flatten(),
            prob_oofs=prob_oofs,
            prob_tests=prob_tests,
            method=self.cfg.weight_opt.method,
            n_folds=self.cfg.weight_opt.n_folds,
            n_loops=self.cfg.weight_opt.n_loops,
        )
        wgt_opt.run_loop()
        prob_oof, prob_test = wgt_opt.calc_weight_averaging()

        pred_oof = prob_oof.argmax(axis=-1)
        pred_test = prob_test.argmax(axis=-1)

        return prob_oof, pred_oof, prob_test, pred_test

    def stacking(self, fold: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """stacking probabilities
        Returns:

        """

        prob_oofs = []
        prob_tests = []
        # 各モデルの出力の確率分布をサンプルごとに結合
        for input_id in self.inputs:
            # load oof (n_samples, n_classes)
            prob_valid = np.load(os.path.join(LOG_DIR, input_id, "prob_oof.npy"))
            assert prob_valid.shape[0] == len(self.oof), "Length does not match"
            assert prob_valid.shape[1] == self.__class__.n_classes, "Length does not match"
            prob_oofs.append(prob_valid)

            # load test (n_samples, n_classes)
            prob_test = np.load(os.path.join(LOG_DIR, input_id, "prob_test_avg.npy"))
            assert prob_test.shape[0] == len(self.sub), "Length does not match"
            assert prob_test.shape[1] == self.__class__.n_classes, "Length does not match"
            prob_tests.append(prob_test)

        prob_oofs = np.array(prob_oofs)  # (n_models, n_samples, n_classes)
        prob_tests = np.array(prob_tests)  # (n_models, n_samples, n_classes)
        logger.debug(f"{prob_oofs.shape=}, {prob_tests.shape=}")

        # Transpose (n_models, n_samples, n_classes) -> (n_samples, n_models, n_classes)
        prob_oofs = prob_oofs.transpose((1, 0, 2))
        prob_tests = prob_tests.transpose((1, 0, 2))
        logger.info(f"Transposed to {prob_oofs.shape=}, {prob_tests.shape=}")

        # Preprocessing
        if self.cfg.stacking.model == "lgbm":
            logger.info("Run stacking using LightGBM")
            # Concatenate probabilitties (n_samples, n_models, n_classes) -> (n_samples, n_models x n_classes)
            prob_oofs = prob_oofs.reshape((prob_oofs.shape[0], prob_oofs.shape[1] * prob_oofs.shape[2]))
            prob_tests = prob_tests.reshape((prob_tests.shape[0], prob_tests.shape[1] * prob_tests.shape[2]))
            logger.info(f"Reshaped to {prob_oofs.shape=}, {prob_tests.shape=}")

        elif self.cfg.stacking.model == "cnn1d":
            logger.info("Run stacking using 1D-CNN")
            logger.info(f"Input shape remains {prob_oofs.shape=}, {prob_tests.shape=}")

        elif self.cfg.stacking.model == "cnn2d":
            logger.info("Run stacking using 2D-CNN")
            # Concatenate probabilitties (n_samples, n_models, n_classes) -> (n_samples, n_models, n_classes, 1)
            prob_oofs = prob_oofs.reshape((prob_oofs.shape[0], prob_oofs.shape[1], prob_oofs.shape[2], 1))
            prob_tests = prob_tests.reshape((prob_tests.shape[0], prob_tests.shape[1], prob_tests.shape[2], 1))
            logger.info(f"Reshaped to {prob_oofs.shape=}, {prob_tests.shape=}")

        else:
            raise ValueError(f"{self.cfg.stacking.model=} is not supported!")

        # Split train-validation
        train_idxs = self.oof.query(f"fold != {fold}").index
        valid_idxs = self.oof.query(f"fold == {fold}").index

        X_train = prob_oofs[train_idxs]
        X_valid = prob_oofs[valid_idxs]
        X_test = prob_tests.copy()

        y_train = self.oof.query(f"fold != {fold}").target.values.flatten()
        y_valid = self.oof.query(f"fold == {fold}").target.values.flatten()

        assert X_train.shape[1:] == X_valid.shape[1:] == X_test.shape[1:], "Shape does not match"
        logger.debug(f"{X_train.shape=}, {X_valid.shape=}, {X_test.shape=}")
        logger.debug(f"{y_train.shape=}, {y_valid.shape=}")

        # Train model
        if self.cfg.stacking.model == "lgbm":
            prob_train, prob_valid, prob_test, model = lgbm.train_and_predict(
                output_dir=self.output_dir,
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
                X_test=X_test,
                hparams=dict(self.cfg.lgbm),
            )

        elif self.cfg.stacking.model == "cnn1d":
            # shuffle model order (n_samples, n_models, n_classes)
            seed = self.cfg.stacking.seed
            if seed >= 0:
                logger.info("Shuffle model order")
                model_order = np.arange(len(self.inputs))
                np.random.seed(seed)
                np.random.shuffle(model_order)
                logger.info(f"{model_order=}")

                X_train = X_train[:, model_order, :]
                X_valid = X_valid[:, model_order, :]
                X_test = X_test[:, model_order, :]

                logger.debug(f"{X_train.shape=}, {X_valid.shape=}, {X_test.shape=}")

            prob_train, prob_valid, prob_test, model = cnn1d.train_and_predict(
                output_dir=self.output_dir,
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
                X_test=X_test,
                hparams=dict(self.cfg.training),
            )

        elif self.cfg.stacking.model == "cnn2d":
            # shuffle model order (n_samples, n_models, n_classes, 1)
            seed = self.cfg.stacking.seed
            if seed >= 0:
                logger.info("Shuffle model order")
                model_order = np.arange(len(self.inputs))
                np.random.seed(seed)
                np.random.shuffle(model_order)
                logger.info(f"{model_order=}")
                X_train = X_train[:, model_order, :, :]
                X_valid = X_valid[:, model_order, :, :]
                X_test = X_test[:, model_order, :, :]

                logger.debug(f"{X_train.shape=}, {X_valid.shape=}, {X_test.shape=}")

            prob_train, prob_valid, prob_test, model = cnn2d.train_and_predict(
                output_dir=self.output_dir,
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
                X_test=X_test,
                hparams=dict(self.cfg.training),
            )

        else:
            raise ValueError(f"{self.cfg.stacking.model=} is not supported!")

        return prob_train, prob_valid, prob_test
