import json
import os
import shutil
from logging import Logger
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from src import NishikaDataset, calc_metrics, get_logger
from src.data_vis import plot_confusion_matrix
from src.ensemble import Ensemble

CUR_DIR = Path().resolve()  # Path to current directory
TARGET2CLASS = NishikaDataset.target2class


def run(output_dir: str, fold: int, hparams: DictConfig, logger: Logger) -> None:
    assert hparams.method == "stacking", "Thin module only supports stacking ensemble"
    ens = Ensemble(output_dir, hparams)
    prob_train, prob_valid, prob_test = ens.stacking(fold=fold)
    oof = ens.oof

    pred_train = prob_train.argmax(axis=1)
    pred_valid = prob_valid.argmax(axis=1)
    # pred_test = prob_test.argmax(axis=1)

    # Save predictions
    with open(os.path.join(output_dir, "prob_train.npy"), "wb") as f:
        np.save(f, prob_train)
    with open(os.path.join(output_dir, "prob_valid.npy"), "wb") as f:
        np.save(f, prob_valid)
    with open(os.path.join(output_dir, "prob_test.npy"), "wb") as f:
        np.save(f, prob_test)

    oof.loc[oof["fold"] == fold, "pred"] = pred_valid
    oof.to_csv(os.path.join(output_dir, "oof.csv"), index=False, header=True)

    # Plot confusion matrix
    y_train = oof.query(f"fold != {fold}")["target"].values
    y_valid = oof.query(f"fold == {fold}")["target"].values

    plot_confusion_matrix(
        trues=[y_train, y_valid],
        preds=[pred_train, pred_valid],
        phases=["train", "valid"],
        labels=NishikaDataset.labels,
        path=os.path.join(output_dir, "confusion_matrix.png"),
    )
    plot_confusion_matrix(
        trues=[y_train, y_valid],
        preds=[pred_train, pred_valid],
        phases=["train", "valid"],
        labels=NishikaDataset.labels,
        path=os.path.join(output_dir, "normalize_confusion_matrix.png"),
        normalize="true",
    )

    # Log best metrics
    phases = ["train", "valid"]
    metrics = ["loss", "accuracy", "f1"]
    result = {phase: {metric: {} for metric in metrics} for phase in phases}

    valid_logs = calc_metrics(y_true=y_valid, y_pred=pred_valid, y_prob=prob_valid, metrics=metrics)
    result["valid"]["loss"][f"cv{fold}"] = valid_logs["loss"]
    result["valid"]["accuracy"][f"cv{fold}"] = valid_logs["accuracy"]
    result["valid"]["f1"][f"cv{fold}"] = valid_logs["f1"]

    train_logs = calc_metrics(y_true=y_train, y_pred=pred_train, y_prob=prob_train, metrics=metrics)
    result["train"]["loss"][f"cv{fold}"] = train_logs["loss"]
    result["train"]["accuracy"][f"cv{fold}"] = train_logs["accuracy"]
    result["train"]["f1"][f"cv{fold}"] = train_logs["f1"]

    with open(os.path.join(output_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=4)


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger, log_dir = get_logger(
        fn_args=[cfg.exp.name],
        dir=os.path.join(CUR_DIR, "logs/"),
        fold=cfg.fold,
        exec_time=cfg.exec_time,
    )

    # move hydra logs to logging directory and reset working directory
    if os.path.exists(os.path.join(log_dir, ".hydra")):
        shutil.rmtree(os.path.join(log_dir, ".hydra"))
    shutil.move(os.path.join(os.getcwd(), ".hydra"), log_dir)
    os.remove(os.path.join(os.getcwd(), os.path.basename(__file__).replace(".py", ".log")))
    os.rmdir(os.getcwd())
    os.chdir(hydra.utils.get_original_cwd())

    run(output_dir=log_dir, fold=cfg.fold, hparams=cfg.exp, logger=logger)


if __name__ == "__main__":
    main()
