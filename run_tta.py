import json
import os
import shutil
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src import NishikaDataset, calc_metrics, get_logger
from src.data_vis import plot_confusion_matrix
from src.tta import TestTimeAugmentation

CUR_DIR = Path().resolve()  # Path to current directory
DATASET_DIR = CUR_DIR.joinpath("input")


def run(cfg: DictConfig, logger, log_dir) -> None:
    # create log directory
    log_id = "_".join([cfg.exp.name, cfg.exec_time])
    base_output_dir = os.path.join(CUR_DIR, "logs", log_id)

    # Inference with Test Time Augmentation
    tta = TestTimeAugmentation(cfg=cfg)
    prob_valid, prob_test = tta.infer()
    assert prob_valid.shape[1] == prob_test.shape[1] == 15, "Length does not match"

    with open(os.path.join(log_dir, "prob_valid.npy"), "wb") as f:
        np.save(f, prob_valid)
    with open(os.path.join(log_dir, "prob_test.npy"), "wb") as f:
        np.save(f, prob_test)

    pred_valid = prob_valid.argmax(axis=1)
    # pred_test = prob_test.argmax(axis=1)

    oof = pd.read_csv(os.path.join(base_output_dir, "oof.csv"))
    oof.loc[oof["fold"] == cfg.fold, "pred"] = pred_valid
    oof.to_csv(os.path.join(log_dir, "oof.csv"), index=False, header=True)

    # Plot confusion matrix
    y_valid = oof.query(f"fold == {cfg.fold}")["target"].values

    plot_confusion_matrix(
        trues=[y_valid],
        preds=[pred_valid],
        phases=["valid"],
        labels=NishikaDataset.labels,
        path=os.path.join(log_dir, "confusion_matrix.png"),
    )
    plot_confusion_matrix(
        trues=[y_valid],
        preds=[pred_valid],
        phases=["valid"],
        labels=NishikaDataset.labels,
        path=os.path.join(log_dir, "normalize_confusion_matrix.png"),
        normalize="true",
    )

    # Log metrics
    phases = ["valid"]
    metrics = ["loss", "accuracy", "f1"]
    result = {phase: {metric: {} for metric in metrics} for phase in phases}

    valid_logs = calc_metrics(y_true=y_valid, y_pred=pred_valid, y_prob=prob_valid, metrics=metrics)
    result["valid"]["loss"][f"cv{cfg.fold}"] = valid_logs["loss"]
    result["valid"]["accuracy"][f"cv{cfg.fold}"] = valid_logs["accuracy"]
    result["valid"]["f1"][f"cv{cfg.fold}"] = valid_logs["f1"]

    with open(os.path.join(log_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=4)


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger, log_dir = get_logger(
        fn_args=[cfg.exp.name], dir=os.path.join(CUR_DIR, "logs/"), fold=cfg.fold, exec_time=cfg.exec_time, tta=True
    )

    # move hydra logs to logging directory and reset working directory
    shutil.move(os.path.join(os.getcwd(), ".hydra"), os.path.join(log_dir, ".hydra"))
    os.remove(os.path.join(os.getcwd(), os.path.basename(__file__).replace(".py", ".log")))
    os.rmdir(os.getcwd())
    os.chdir(hydra.utils.get_original_cwd())

    run(cfg, logger, log_dir)


if __name__ == "__main__":
    main()
