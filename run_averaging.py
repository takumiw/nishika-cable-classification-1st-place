import os
import shutil
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from src import NishikaDataset, calc_metrics, get_logger
from src.data_vis import plot_confusion_matrix
from src.ensemble import Ensemble

CUR_DIR = Path().resolve()  # Path to current directory
TARGET2CLASS = NishikaDataset.target2class


def run(output_dir, cfg, logger):
    ens = Ensemble(output_dir, cfg)
    prob_oof, pred_oof, prob_test, pred_test = ens.averaging()

    with open(os.path.join(output_dir, "prob_oof.npy"), "wb") as f:
        np.save(f, prob_oof)
    with open(os.path.join(output_dir, "prob_test.npy"), "wb") as f:
        np.save(f, prob_test)

    oof = ens.oof
    oof["pred"] = pred_oof
    oof.to_csv(os.path.join(output_dir, "oof.csv"), index=False, header=True)

    # Create submission
    sub = ens.sub
    sub["class"] = pred_test
    sub["class"] = sub["class"].map(lambda x: TARGET2CLASS[x])
    sub.to_csv(os.path.join(output_dir, "submission.csv"), index=False)

    # Plot confusion matrix
    y_valid = oof.target.values.flatten()
    plot_confusion_matrix(
        trues=[y_valid],
        preds=[pred_oof],
        phases=["oof"],
        labels=NishikaDataset.labels,
        path=os.path.join(output_dir, "confusion_matrix.png"),
    )
    plot_confusion_matrix(
        trues=[y_valid],
        preds=[pred_oof],
        phases=["oof"],
        labels=NishikaDataset.labels,
        path=os.path.join(output_dir, "normalize_confusion_matrix.png"),
        normalize="true",
    )

    # Log metrics
    metrics = ["loss", "accuracy", "f1"]

    oof_logs = calc_metrics(y_true=y_valid, y_pred=pred_oof, y_prob=prob_oof, metrics=metrics)
    for metric in metrics:
        logger.debug(f"{metric}: {oof_logs[metric]:.06f}")


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger, log_dir = get_logger(
        fn_args=[cfg.exp.name],
        dir=os.path.join(CUR_DIR, "logs/"),
    )

    # move hydra logs to logging directory and reset working directory
    shutil.move(os.path.join(os.getcwd(), ".hydra"), log_dir)
    os.remove(os.path.join(os.getcwd(), os.path.basename(__file__).replace(".py", ".log")))
    os.rmdir(os.getcwd())
    os.chdir(hydra.utils.get_original_cwd())

    run(output_dir=log_dir, cfg=cfg.exp, logger=logger)


if __name__ == "__main__":
    main()
