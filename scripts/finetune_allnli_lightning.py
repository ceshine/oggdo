import os
import json
from pathlib import Path
from dataclasses import asdict
from typing import Tuple, Dict, Optional, Any, List, Iterable

import typer
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import pytorch_lightning_spells as pls

from oggdo.dataset import XnliDfDataset, SBertDataset
# from oggdo.components import TransformerWrapper, PoolingLayer
# from oggdo.encoder import SentenceEncoder
# from oggdo.models import SentencePairNliClassification
from oggdo.lightning_modules import BaseConfig, NliModule, SBertSentencePairDataModule

from finetune_xnli_lightning import load_model


CACHE_DIR = Path('./cache/')
CACHE_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR = Path('./cache/models/')
MODEL_DIR.mkdir(exist_ok=True, parents=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

NO_DECAY = [
    'LayerNorm.weight', 'LayerNorm.bias'
]


def main(
    model_path: str,
    sample_train: float = -1,
    data_folder: str = "data/",
    batch_size: int = 32,
    lr: float = 3e-5,
    epochs: int = 3,
    wd: float = 0,
    grad_accu: int = 1,
    layerwise_decay: bool = False,
    model_type: Optional[str] = None,
    lowercase: bool = False,
    use_amp: bool = False,
    workers: int = 2
):
    pl.seed_everything(int(os.environ.get("SEED", 42)))
    model = load_model(model_path, model_type, do_lower_case=lowercase)

    config = BaseConfig(
        model_path=model_path,
        data_path=data_folder,
        sample_train=sample_train,
        batch_size=batch_size, grad_accu=grad_accu,
        learning_rate=lr, fp16=use_amp,
        epochs=epochs, loss_fn=nn.CrossEntropyLoss(),
        t2s=False,
        # optimizer_cls=pls.optimizers.RAdam,
        optimizer_cls=torch.optim.AdamW,
        weight_decay=wd,
        layerwise_decay=layerwise_decay
    )

    pl_module = NliModule(
        config, model, metrics=(("accuracy", pl.metrics.Accuracy()),),
        layerwise_decay=config.layerwise_decay
    )

    data_module = SBertSentencePairDataModule(
        SBertDataset.AllNLI, model.encoder[0], config,
        dataset_cls=XnliDfDataset, workers=workers,
        name="allnli"
    )

    checkpoints = pl.callbacks.ModelCheckpoint(
        dirpath=str(CACHE_DIR / "model_checkpoints"),
        monitor='val_loss',
        mode="min",
        filename='{step:06d}-{val_loss:.4f}',
        save_top_k=1,
        save_last=False
    )

    callbacks = [
        checkpoints,
        pl.callbacks.LearningRateMonitor(logging_interval='step'),
    ]

    trainer = pl.Trainer(
        # accelerator='dp' if num_gpus > 1 else None,
        # amp_backend="apex", amp_level='O2',
        precision=16 if config.fp16 else 32,
        gpus=1,
        val_check_interval=0.25,
        gradient_clip_val=10,
        max_epochs=epochs,
        # max_steps=steps,
        callbacks=callbacks,
        accumulate_grad_batches=config.grad_accu,
        # auto_scale_batch_size='power' if batch_size is None else None,
        logger=[
            pl.loggers.TensorBoardLogger(str(CACHE_DIR / "tb_logs"), name=""),
            pls.loggers.ScreenLogger(),
            # pl.loggers.WandbLogger(project="news-similarity")
        ],
        log_every_n_steps=100
    )

    trainer.fit(pl_module, datamodule=data_module)

    output_folder = MODEL_DIR / f"allnli_{Path(model_path).name}"
    model.save(str(output_folder))

    pl_module.load_state_dict(torch.load(checkpoints.best_model_path)["state_dict"])
    trainer.test(datamodule=data_module)

    config_dict = asdict(config)
    del config_dict["loss_fn"]
    del config_dict["optimizer_cls"]
    (output_folder / 'params.json').write_text(
        json.dumps(config_dict, indent=4, sort_keys=True))

if __name__ == '__main__':
    typer.run(main)
