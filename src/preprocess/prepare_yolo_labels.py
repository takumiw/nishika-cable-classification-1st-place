import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

CUR_DIR = Path().resolve()
ROOT_DIR = CUR_DIR.parent.parent
DATASET_DIR = ROOT_DIR / "input"
OUTPUT_DIR = DATASET_DIR / "preprocessed" / "train_all_clean_1280" / "labels"

parser = argparse.ArgumentParser()
parser.add_argument("target", type=str, default="Lightning_T_edge")
args = parser.parse_args()
print(f"{args=}")


def main():
    if args.target == "Lightning_T_edge":
        train_all_df = pd.read_csv(DATASET_DIR / "train_all_clean.csv")
    elif args.target == "Lightning_T_all":
        train_all_df = pd.read_csv(DATASET_DIR / "train_all_clean_Lightning_T_all.csv")
    print(f"{train_all_df.shape}=")

    for _, (filename, _, target, x_center, y_center, width, height) in tqdm(train_all_df.iterrows()):
        body = f"{target} {x_center} {y_center} {width} {height} "
        with open(OUTPUT_DIR / filename.replace(".jpg", ".txt"), "w") as f:
            f.write(body)
            f.write("\n")


if __name__ == "__main__":
    main()
