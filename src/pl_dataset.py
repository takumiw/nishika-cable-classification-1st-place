"""Dataset Modules"""
import os
import pickle
from logging import getLogger
from typing import Tuple

import albumentations as A
import pandas as pd
import pytorch_lightning as pl
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

from src.data_vis import read_cv_image

logger = getLogger(__name__)


class NishikaDataset(Dataset):
    object_path_key = "path"
    target_key = "target"
    target2class = {
        0: "USB_Micro_B",
        1: "USB_Micro_B_3.1",
        2: "USB_Micro_B_W",
        3: "USB_Type_A",
        4: "USB_Type_B",
        5: "USB_Type_C",
        6: "USB_Mini",
        7: "Lightning",
        8: "Lightning_T",
        9: "DisplayPort",
        10: "Mini_DisplayPort",
        11: "RJ_45",
        12: "HDMI",
        13: "VGA",
        14: "Dock",
    }
    class2target = {
        "USB_Micro_B": 0,
        "USB_Micro_B_3.1": 1,
        "USB_Micro_B_W": 2,
        "USB_Type_A": 3,
        "USB_Type_B": 4,
        "USB_Type_C": 5,
        "USB_Mini": 6,
        "Lightning": 7,
        "Lightning_T": 8,
        "DisplayPort": 9,
        "Mini_DisplayPort": 10,
        "RJ_45": 11,
        "HDMI": 12,
        "VGA": 13,
        "Dock": 14,
    }
    labels = [
        "USB_Micro_B",
        "USB_Micro_B_3.1",
        "USB_Micro_B_W",
        "USB_Type_A",
        "USB_Type_B",
        "USB_Type_C",
        "USB_Mini",
        "Lightning",
        "Lightning_T",
        "DisplayPort",
        "Mini_DisplayPort",
        "RJ_45",
        "HDMI",
        "VGA",
        "Dock",
    ]

    def __init__(self, meta_df: pd.DataFrame, dataset_dir: str, phase: str = "train", transform=None):
        """
        Args:
            meta_df (pd.DataFrame): meta dataframe with columns ["filename", "class"]
            dataset_dir (str): path to dataset directory
            phase (str): chosen from ["train", "valid", "test"]
            transform: transformation applied to an image
        """
        self.phase = phase
        if phase == "test":
            meta_df["path"] = meta_df["filename"].map(
                lambda x: os.path.join(
                    dataset_dir, "preprocessed", "test_all_clean_cropped_pad1", x.replace(".jpg", ".pickle")
                )
            )
        else:
            meta_df.loc[meta_df["pseudo"] is False, "path"] = meta_df.loc[meta_df["pseudo"] is False, "filename"].map(
                lambda x: os.path.join(
                    dataset_dir, "preprocessed", "train_all_clean_cropped_pad1", x.replace(".jpg", ".pickle")
                )
            )
            meta_df.loc[meta_df["pseudo"] is True, "path"] = meta_df.loc[meta_df["pseudo"] is True, "filename"].map(
                lambda x: os.path.join(
                    dataset_dir, "preprocessed", "test_all_clean_cropped_pad1", x.replace(".jpg", ".pickle")
                )
            )

        meta_df["target"] = meta_df["class"].map(lambda x: self.__class__.class2target[x])
        meta_df = meta_df[["path", "target"]]
        self.meta_df = meta_df.reset_index(drop=True)
        self.index_to_data = self.meta_df.to_dict(orient="index")
        self.transform = transform

    def __getitem__(self, index):
        data = self.index_to_data[index]
        with open(data.get(self.object_path_key), "rb") as f:
            img = pickle.load(f)
        # img = read_cv_image(data.get(self.object_path_key))
        img = self.transform(image=img)["image"]
        target = data.get(self.target_key, -1)

        if self.phase in ["train", "valid"]:
            return img, target
        else:
            return img

    def __len__(self):
        return len(self.meta_df)


class NishikaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str = "path/to/dataset/directory",
        output_dir: str = "path/to/output/directory",
        use_pseudo: bool = False,
        batch_size: int = 32,
        fold: int = 0,
        n_splits: int = 5,
        size: Tuple[int, int] = (224, 224),
        norm_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        norm_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        num_workers: int = 10,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.use_pseudo = use_pseudo
        self.batch_size = batch_size
        self.fold = fold
        self.n_splits = n_splits
        self.size = size
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.num_workers = num_workers

    def get_train_valid_df(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        use_cols = ["filename", "class", "target"]
        df = pd.read_csv(os.path.join(self.dataset_dir, "train_all_clean.csv"))[use_cols]
        df["pseudo"] = False

        if self.use_pseudo:
            df_pseudo = pd.read_csv(os.path.join(self.dataset_dir, "test_pseudo.csv"))[use_cols]
            df_pseudo["pseudo"] = True
            df = pd.concat([df, df_pseudo]).reset_index(drop=True)
            logger.debug(f"Load pseudo labeled test dataset: {len(df_pseudo)} samples")

        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=71)
        for fold_, (train_idxs, valid_idxs) in enumerate(cv.split(df, y=df["class"])):
            df.loc[valid_idxs, "fold"] = fold_

        df["fold"] = df["fold"].astype(int)
        path = os.path.join(self.output_dir, "oof.csv")
        if not os.path.exists(path):
            df_oof = df.copy()
            df_oof["pred"] = -1
            df_oof["target"] = df_oof["class"].map(lambda x: NishikaDataset.class2target[x])
            df_oof.to_csv(path, index=False, header=True)
        train_df = df.query(f"fold != {self.fold}").reset_index(drop=True)
        valid_df = df.query(f"fold == {self.fold}").reset_index(drop=True)
        return train_df, valid_df

    def get_test_df(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.dataset_dir, "sample_submission.csv"))

    def get_transform(self, phase: str):
        additional_items = (
            [A.Resize(self.size[0], self.size[1])]
            if phase != "train"
            else [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Blur(blur_limit=(3, 13), p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
                A.HueSaturationValue(sat_shift_limit=15, hue_shift_limit=10, p=0.5),
                A.Resize(self.size[0], self.size[1]),
            ]
        )
        transform = A.Compose(
            [
                *additional_items,
                A.Normalize(mean=self.norm_mean, std=self.norm_std, always_apply=True),
                ToTensorV2(always_apply=True),
            ]
        )
        return transform

    def setup(self, stage=None):
        train_df, valid_df = self.get_train_valid_df()
        test_df = self.get_test_df()
        logger.debug(f"{train_df.shape=}, {valid_df.shape=}, {test_df.shape=}")
        self.train_dataset = NishikaDataset(
            meta_df=train_df,
            dataset_dir=self.dataset_dir,
            transform=self.get_transform(phase="train"),
            phase="train",
        )
        self.train_inference_dataset = NishikaDataset(
            meta_df=train_df,
            dataset_dir=self.dataset_dir,
            transform=self.get_transform(phase="inference"),
            phase="inference",
        )
        self.valid_dataset = NishikaDataset(
            meta_df=valid_df,
            dataset_dir=self.dataset_dir,
            transform=self.get_transform(phase="valid"),
            phase="valid",
        )
        self.valid_inference_dataset = NishikaDataset(
            meta_df=valid_df,
            dataset_dir=self.dataset_dir,
            transform=self.get_transform(phase="inference"),
            phase="inference",
        )
        self.test_dataset = NishikaDataset(
            meta_df=test_df,
            dataset_dir=self.dataset_dir,
            transform=self.get_transform(phase="test"),
            phase="test",
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def train_inference_dataloader(self):
        return DataLoader(
            self.train_inference_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def val_inference_dataloader(self):
        return DataLoader(
            self.valid_inference_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
