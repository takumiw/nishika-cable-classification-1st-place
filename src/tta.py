import os
from logging import getLogger
from typing import Tuple

import my_ttach as tta
import numpy as np
import torch
from omegaconf import DictConfig
from torch.nn import functional as F
from tqdm import tqdm

from src import LitModule, NishikaDataModule

logger = getLogger(__name__)

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_WORKERS = min(os.cpu_count(), 10)
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


class TestTimeAugmentation:

    # 2 x 3 x 3 = 12
    # transforms = tta.Compose(
    #     [
    #         tta.HorizontalFlip(),
    #         tta.VerticalFlip(),
    #         tta.Scale(scales=[1, 0.83, 0.67]),
    #     ]
    # )
    root_dir = os.path.join(CUR_DIR, "../")
    dataset_dir = os.path.join(root_dir, "input")
    # n_tta = 12
    n_classes = 15
    default_scales = [1, 0.83, 0.67]
    default_angles = [0, 90, 270]
    default_factors = [0.9, 1.0, 1.1]

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.cfg_tta = cfg.exp.tta
        self.log_id = "_".join([cfg.exp.name, cfg.exec_time])
        self.fold = cfg.fold
        self.tta_size = cfg.exp.tta.size
        self.batch_size = cfg.exp.tta.batch_size
        self.base_output_dir = os.path.join(self.__class__.root_dir, "logs", self.log_id)
        self.transforms = self.get_transforms()

    def get_transforms(self):
        transforms = []
        for transform in self.cfg_tta.transforms:
            if transform == "HorizontalFlip":
                transforms.append(tta.HorizontalFlip())
                logger.info("Add transform: HorizontalFlip")
            elif transform == "VerticalFlip":
                transforms.append(tta.VerticalFlip())
                logger.info("Add transform: VerticalFlip")
            elif transform == "Scale":
                if "scales" in self.cfg_tta.settings:
                    transforms.append(tta.Scale(scales=self.cfg_tta.settings.scales))
                    logger.info(f"Add transform: Scale ({self.cfg_tta.settings.scales})")
                else:
                    transforms.append(tta.Scale(scales=self.__class__.default_scales))
                    logger.info(f"Add transform: Scale ({self.__class__.default_scales})")
            elif transform == "Rotate90":
                if "angles" in self.cfg_tta.settings:
                    transforms.append(tta.Rotate90(angles=self.cfg_tta.settings.angles))
                    logger.info(f"Add transform: Rotate90 ({self.cfg_tta.settings.angles})")
                else:
                    transforms.append(tta.Rotate90(angles=self.__class__.default_angles))
                    logger.info(f"Add transform: Rotate90 ({self.__class__.default_angles})")
            elif transform == "Multiply":
                if "factors" in self.cfg_tta.settings:
                    transforms.append(tta.Multiply(factors=self.cfg_tta.settings.factors))
                    logger.info(f"Add transform: Multiply ({self.cfg_tta.settings.factors})")
                else:
                    transforms.append(tta.Multiply(angles=self.__class__.default_factors))
                    logger.info(f"Add transform: Multiply ({self.__class__.default_factors})")
            else:
                raise ValueError(f"{transform} is not supported")

        logger.info(f"Set {len(transforms)} transforms")
        return tta.Compose(transforms)

    def infer(self) -> Tuple[np.ndarray, np.ndarray]:
        output_dir = os.path.join(self.base_output_dir, "tta", f"cv{self.fold}")
        os.makedirs(output_dir, exist_ok=True)

        # Get DataLoader
        datamodule = NishikaDataModule(
            dataset_dir=self.__class__.dataset_dir,
            output_dir=output_dir,
            use_pseudo=self.cfg.exp.input.use_pseudo,
            batch_size=self.batch_size,
            fold=self.fold,
            size=(self.tta_size, self.tta_size),
            num_workers=NUM_WORKERS,
        )
        datamodule.setup()
        val_inference_dataloader = datamodule.val_inference_dataloader()
        test_dataloader = datamodule.predict_dataloader()

        # Load trained model
        logger.debug("Load model from checkpoint ...")
        path_model = os.path.join(self.base_output_dir, f"cv{self.fold}", "model_best_loss.ckpt")
        model = LitModule(hparams=self.cfg.exp).load_from_checkpoint(path_model)
        model.to(DEVICE)
        model.eval()

        tta_model = tta.ClassificationTTAWrapper(model, self.transforms, merge_mode="raw")

        # Predict validation dataset
        logger.debug("Predict validation set ...")
        preds_valid = []
        with torch.no_grad():
            for batch in tqdm(val_inference_dataloader):
                x = batch.to(DEVICE)
                y = tta_model(x)  # (n_tta, batch_size, n_classes)

                # forward softmax
                prob = np.array([F.softmax(p, dim=-1).detach().cpu().numpy() for p in y])
                prob = prob.mean(axis=0)  # (batch_size, n_classes)
                preds_valid.append(prob)

        preds_valid = np.concatenate(preds_valid)

        # Predict test dataset
        logger.debug("Predict test set ...")
        preds_test = []
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                x = batch.to(DEVICE)
                y = tta_model(x)

                # forward softmax
                prob = np.array([F.softmax(p, dim=-1).detach().cpu().numpy() for p in y])
                prob = prob.mean(axis=0)  # (batch_size, n_classes)
                preds_test.append(prob)

        preds_test = np.concatenate(preds_test)

        return preds_valid, preds_test
