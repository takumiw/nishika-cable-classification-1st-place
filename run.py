import json
import os
import shutil
from pathlib import Path
from typing import Tuple

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from configs import settings
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
# from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.csv_logs import CSVLogger
from torch.nn import functional as F

from src import LitModule, NishikaDataModule, NishikaDataset, calc_metrics, get_logger, seed_everything, summary_model
from src.data_vis import plot_confusion_matrix

CUR_DIR = Path().resolve()  # Path to current directory
DATASET_DIR = CUR_DIR.joinpath("input")
NUM_WORKERS = min(os.cpu_count(), 10)


def train(output_dir: str, fold: int, hparams: DictConfig, logger) -> Tuple[str, float]:
    seed_everything(hparams.model.seed, deterministic=True, benchmark=False)
    logger.debug(f"Set seed with {hparams.model.seed}")

    csv_logger = CSVLogger(save_dir=output_dir, name="history")
    # wandb_logger = WandbLogger(
    #     name=f"{hparams.name}_cv{fold}",
    #     save_dir=output_dir,
    #     offline=False,
    #     project="Nishika19",
    #     tags=[hparams.model.backbone, f"cv{fold}"],
    # )
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="model_best_loss",
        save_weights_only=True,
        monitor="valid/loss",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        default_root_dir=output_dir,
        max_epochs=hparams.training.epochs,
        gpus=1,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=[
            csv_logger,
            # wandb_logger,
        ],
        deterministic=True,
        log_every_n_steps=hparams.training.log_every_n_steps,
    )

    model = LitModule(hparams=hparams)
    summary_model(
        model.model,
        input_size=(
            hparams.training.batch_size,
            3,
            hparams.input.size[0],
            hparams.input.size[1],
        ),
        path=os.path.join(output_dir, "model.txt"),
    )
    # wandb_logger.watch(model, log="gradients", log_freq=100)  # Watch weights distributions
    datamodule = NishikaDataModule(
        dataset_dir=DATASET_DIR,
        output_dir=output_dir,
        use_pseudo=hparams.input.use_pseudo,
        batch_size=hparams.training.batch_size,
        fold=fold,
        size=hparams.input.size,
        num_workers=NUM_WORKERS,
    )
    trainer.fit(model, datamodule=datamodule)

    # Inference
    logits_test = trainer.predict(datamodule=datamodule, return_predictions=True, ckpt_path="best")
    logits_test = torch.cat(logits_test)
    prob_test = F.softmax(logits_test, dim=1).cpu().numpy()
    with open(os.path.join(output_dir, "prob_test.npy"), "wb") as f:
        np.save(f, prob_test)

    train_inference_dataloader = datamodule.train_inference_dataloader()
    logits_train = trainer.predict(
        dataloaders=train_inference_dataloader,
        return_predictions=True,
        ckpt_path="best",
    )
    logits_train = torch.cat(logits_train)
    prob_train = F.softmax(logits_train, dim=1).cpu().numpy()
    with open(os.path.join(output_dir, "prob_train.npy"), "wb") as f:
        np.save(f, prob_train)
    pred_train = prob_train.argmax(axis=1)

    val_inference_dataloader = datamodule.val_inference_dataloader()
    logits_valid = trainer.predict(dataloaders=val_inference_dataloader, return_predictions=True, ckpt_path="best")
    logits_valid = torch.cat(logits_valid)
    prob_valid = F.softmax(logits_valid, dim=1).cpu().numpy()
    with open(os.path.join(output_dir, "prob_valid.npy"), "wb") as f:
        np.save(f, prob_valid)
    pred_valid = prob_valid.argmax(axis=1)
    oof = pd.read_csv(os.path.join(output_dir, "oof.csv"))
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

    return (
        checkpoint_callback.best_model_path,
        checkpoint_callback.best_model_score.item(),
    )


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger, log_dir = get_logger(
        fn_args=[cfg.exp.name],
        dir=os.path.join(CUR_DIR, "logs/"),
        fold=cfg.fold,
        exec_time=cfg.exec_time,
    )

    # move hydra logs to logging directory and reset working directory
    shutil.move(os.path.join(os.getcwd(), ".hydra"), log_dir)
    os.remove(os.path.join(os.getcwd(), os.path.basename(__file__).replace(".py", ".log")))
    os.rmdir(os.getcwd())
    os.chdir(hydra.utils.get_original_cwd())

    logger.debug(f"{NUM_WORKERS=}")
    best_model_path, best_model_score = train(output_dir=log_dir, fold=cfg.fold, hparams=cfg.exp, logger=logger)
    logger.debug(f"Best model path: {best_model_path}")
    logger.debug(f"Best validation loss: {best_model_score:.04f}")


if __name__ == "__main__":
    main()
