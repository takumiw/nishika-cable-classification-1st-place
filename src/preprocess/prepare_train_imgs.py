import os
import pickle
from pathlib import Path

import albumentations as A
import cv2
import pandas as pd
from tqdm import tqdm

from src.data_vis import read_cv_image

CUR_DIR = Path().resolve()
ROOT_DIR = CUR_DIR.parent.parent
DATASET_DIR = ROOT_DIR / "input"

border_mode = 1


def padding_and_cropping(img, x_center, y_center, width, height):
    # padding
    h = w = max(img.shape[:2])
    composer = A.Compose([A.PadIfNeeded(always_apply=True, min_height=h, min_width=w, border_mode=1)])
    img = composer(image=img)["image"]

    # cropping
    x_center_pos = x_center * w
    y_center_pos = y_center * h
    x_min = x_center_pos - width * w / 2
    x_max = x_center_pos + width * w / 2
    y_min = y_center_pos - height * h / 2
    y_max = y_center_pos + height * h / 2
    composer = A.Compose(
        [A.Crop(always_apply=True, x_min=int(x_min), y_min=int(y_min), x_max=int(x_max), y_max=int(y_max))]
    )
    img = composer(image=img)["image"]
    return img


def run_cropping():
    train_all_df = pd.read_csv(DATASET_DIR / "train_all_clean.csv")
    train_all_df["path"] = DATASET_DIR / "train_all" / train_all_df["filename"]
    print(f"{train_all_df.shape=}")

    dir_ = os.path.join(DATASET_DIR, "preprocessed", "train_all_clean_cropped")
    for row in tqdm(train_all_df.iterrows()):
        filename, _, _, x_center, y_center, width, height, path = row[1]
        img = cv2.imread(str(path))
        img = padding_and_cropping(img, x_center, y_center, width, height)
        cv2.imwrite(os.path.join(dir_, filename), img)


def run_padding():
    train_df = pd.read_csv(DATASET_DIR / "train_all_clean.csv")
    train_df["path"] = DATASET_DIR / "preprocessed" / "train_all_clean_cropped" / train_df["filename"]

    for filename, path in tqdm(train_df[["filename", "path"]].values):
        img = read_cv_image(str(path))
        size = max(img.shape[0:2])
        composer = A.Compose(
            [
                A.PadIfNeeded(always_apply=True, min_height=size, min_width=size, border_mode=border_mode),
            ]
        )
        img = composer(image=img)["image"]

        save_path = DATASET_DIR / "preprocessed" / "train_all_clean_cropped_pad1" / filename.replace(".jpg", ".pickle")
        with open(save_path, "wb") as f:
            pickle.dump(img, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    run_cropping()
    run_padding()


if __name__ == "__main__":
    main()
