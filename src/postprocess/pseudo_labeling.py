from pathlib import Path

import numpy as np
import pandas as pd

from src.pl_dataset import NishikaDataset


EXP_ID = "exp12_averaging_2021-1031-0800"
CUR_DIR = Path().resolve()
ROOT_DIR = CUR_DIR.parent.parent
LOG_DIR = ROOT_DIR / "logs" / EXP_ID
DATASET_DIR = ROOT_DIR / "input" / "preprocessed"

target2class = NishikaDataset.target2class


def main(thre: float = 0.9):
    test_df = pd.read_csv(ROOT_DIR / "input" / "sample_submission.csv")
    test_prob = np.load(LOG_DIR / "prob_test.npy")

    test_df.loc[test_prob.max(axis=-1) > thre, "target_pseudo"] = test_prob[test_prob.max(axis=-1) > thre].argmax(
        axis=-1
    )
    test_df["has_pseudo"] = False
    test_df.loc[~test_df.target_pseudo.isnull(), "has_pseudo"] = True
    test_df.loc[test_df.has_pseudo is True, "class_pseudo"] = test_df.loc[
        test_df.has_pseudo is True, "target_pseudo"
    ].map(lambda x: target2class[x])

    test_pseudo_df = test_df[test_df["has_pseudo"] is True][["filename", "class_pseudo", "target_pseudo"]]
    test_pseudo_df.columns = ["filename", "class", "target"]
    test_pseudo_df.reset_index(drop=True, inplace=True)
    test_pseudo_df["target"] = test_pseudo_df["target"].astype(int)

    print(test_pseudo_df.shape, test_pseudo_df.shape[0] / test_df.shape[0])

    test_pseudo_df.to_csv(ROOT_DIR / "input" / "test_pseudo.csv", index=False, header=True)


if __name__ == "__main__":
    main()
