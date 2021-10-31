import argparse
import json
import os
import sys
from logging import DEBUG, WARNING, StreamHandler, basicConfig, getLogger
from pathlib import Path

import numpy as np
import pandas as pd

from src import NishikaDataset
from src.data_vis import plot_confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument("lod_id", type=str, default="exp20_effb4_380_2021-1031-1000")
parser.add_argument("--tta", action="store_true")
args = parser.parse_args()
print(args)

CUR_DIR = Path().resolve()  # Path to current directory
LOG_DIR = CUR_DIR.joinpath("logs", args.lod_id)
if args.tta:
    LOG_DIR = LOG_DIR / "tta"
DATASET_DIR = CUR_DIR.joinpath("input")
LABELS = NishikaDataset.labels
TARGET2CLASS = NishikaDataset.target2class

formatter = "%(levelname)s: %(asctime)s: %(filename)s: %(funcName)s: %(message)s"
basicConfig(filename=os.path.join(LOG_DIR, "summary.log"), level=DEBUG, format=formatter, force=True)
getLogger("matplotlib").setLevel(WARNING)  # Suppress matplotlib logging
getLogger().addHandler(StreamHandler(sys.stdout))

logger = getLogger(__name__)
logger.setLevel(DEBUG)


def main():
    cv_dirs = sorted(LOG_DIR.glob("cv[0-9]"))
    if args.tta:
        phases = ["valid"]
    else:
        phases = ["train", "valid"]
    metrics = ["loss", "accuracy", "f1"]

    # Summarize metrics
    result = {phase: {metric: {} for metric in metrics} for phase in phases}
    for cv_dir in cv_dirs:
        cv = cv_dir.stem
        with open(os.path.join(cv_dir, "result.json")) as f:
            result_cv = json.load(f)
        for phase in phases:
            for metric in metrics:
                result[phase][metric][cv] = result_cv[phase][metric][cv]

    for phase in phases:
        for metric in metrics:
            logger.debug(
                f"{phase} {metric}: {np.mean(list(result[phase][metric].values())):.06f} -- {result[phase][metric]}"
            )

    with open(os.path.join(LOG_DIR, "result.json"), "w") as f:
        json.dump(result, f, indent=4)

    # Summarize oof inference
    oof = pd.read_csv(os.path.join(cv_dirs[0], "oof.csv"))
    for cv_dir in cv_dirs[1:]:
        fold = int(cv_dir.stem.lstrip("cv"))
        oof_cv = pd.read_csv(os.path.join(cv_dir, "oof.csv"))
        oof.loc[oof.fold == fold, "pred"] = oof_cv.loc[oof_cv.fold == fold, "pred"]

    oof.to_csv(os.path.join(LOG_DIR, "oof.csv"), index=False, header=True)

    prob_oof = np.empty((len(oof), len(LABELS)))
    for cv_dir in cv_dirs:
        fold = int(cv_dir.stem.lstrip("cv"))
        prob_cv = np.load(os.path.join(cv_dir, "prob_valid.npy"))
        valid_idxs = oof[oof.fold == fold].index.tolist()
        prob_oof[valid_idxs] = prob_cv

    with open(os.path.join(LOG_DIR, "prob_oof.npy"), "wb") as f:
        np.save(f, prob_oof)

    # Summarize test inference
    prob_test = []
    for cv_dir in cv_dirs:
        prob_test.append(np.load(os.path.join(cv_dir, "prob_test.npy")))

    prob_test = np.array(prob_test)
    prob_test_avg = np.mean(prob_test, axis=0)  # Averaging
    pred_test = prob_test_avg.argmax(axis=1)

    with open(os.path.join(LOG_DIR, "prob_test.npy"), "wb") as f:
        np.save(f, prob_test)
    with open(os.path.join(LOG_DIR, "prob_test_avg.npy"), "wb") as f:
        np.save(f, prob_test_avg)
    with open(os.path.join(LOG_DIR, "pred_test.npy"), "wb") as f:
        np.save(f, pred_test)

    # Create submission
    sub = pd.read_csv(os.path.join(DATASET_DIR, "sample_submission.csv"))
    sub["class"] = pred_test
    sub["class"] = sub["class"].map(lambda x: TARGET2CLASS[x])
    sub.to_csv(os.path.join(LOG_DIR, "submission.csv"), index=False)

    # Plot oof results
    plot_confusion_matrix(
        trues=[oof["target"].values],
        preds=[prob_oof.argmax(axis=1)],
        phases=["oof"],
        labels=LABELS,
        path=os.path.join(LOG_DIR, "confusion_matrix.png"),
    )
    plot_confusion_matrix(
        trues=[oof["target"].values],
        preds=[prob_oof.argmax(axis=1)],
        phases=["oof"],
        labels=LABELS,
        path=os.path.join(LOG_DIR, "normalize_confusion_matrix.png"),
        normalize="true",
    )


if __name__ == "__main__":
    main()
