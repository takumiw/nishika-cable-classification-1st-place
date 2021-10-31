from pathlib import Path

import albumentations as A
import cv2
import joblib
import pandas as pd
from tqdm import tqdm

CUR_DIR = Path().resolve()
ROOT_DIR = CUR_DIR.parent.parent
DATASET_DIR = ROOT_DIR / "input"


class YoloPreprocessor:
    def __init__(self, size: int = 1280, phase: str = "train_all_clean", border_mode: int = 1):
        self.size = size
        self.phase = phase
        self.border_mode = border_mode

    def __call__(self, row):
        fn = row.filename
        path = row.path
        img = cv2.imread(str(path))

        # Padding
        H, W = img.shape[0:2]
        composer = A.Compose(
            [
                A.PadIfNeeded(
                    always_apply=True, min_height=max(H, W), min_width=max(H, W), border_mode=self.border_mode
                ),
            ]
        )
        img = composer(image=img)["image"]

        # Resize
        composer = A.Compose([A.Resize(self.size, self.size)])
        img = composer(image=img)["image"]

        # Save image in jpg format
        cv2.imwrite(str(DATASET_DIR.joinpath("preprocessed", f"{self.phase}_{self.size}", "images", fn)), img)


def main():
    # load dataset
    train_all_df = pd.read_csv(DATASET_DIR / "train_all_clean.csv")
    train_all_df["path"] = DATASET_DIR / "train_all" / train_all_df["filename"]

    test_df = pd.read_csv(DATASET_DIR / "sample_submission.csv")
    test_df["path"] = DATASET_DIR / "test" / test_df["filename"]

    print(f"{train_all_df.shape=} {test_df.shape=}")

    # preprocess train images
    pool = joblib.Parallel(4)
    converter = YoloPreprocessor(size=1280, phase="train_all_clean")
    mapper = joblib.delayed(converter)
    tasks = [mapper(row) for row in train_all_df.itertuples()]
    pool(tqdm(tasks))

    # preprocess test images
    pool = joblib.Parallel(4)
    converter = YoloPreprocessor(size=1280, phase="test")
    mapper = joblib.delayed(converter)
    tasks = [mapper(row) for row in test_df.itertuples()]
    pool(tqdm(tasks))


if __name__ == "__main__":
    main()
