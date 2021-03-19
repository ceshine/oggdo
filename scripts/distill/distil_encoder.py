import os
from pathlib import Path
from functools import partial

import typer
import torch
import joblib
from torch import nn
import pytorch_lightning as pl
import pytorch_lightning_spells as pls
from torch.utils.data import DataLoader
from oggdo.dataset import DistillDataset
from oggdo.dataloading import SortSampler, SortishSampler, collate_singles
from oggdo.lightning_modules import BaseConfig, SentenceEncoderModule

from common import load_encoder


CACHE_DIR = Path("cache/")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class EncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, **batch):
        return self.encoder(batch)["sentence_embeddings"]


def main(
    model_path: str = "nreimers/TinyBERT_L-4_H-312_v2",
    cache_folder: str = "cache/teacher_embs/", batch_size: int = 32,
    fp16: bool = False, workers: int = 2, grad_accu: int = 1,
    lr: float = 3e-5, epochs: int = 2, wd: float = 0,
    layerwise_decay: bool = False
):
    pl.seed_everything(int(os.environ.get("SEED", 42)))

    config = BaseConfig(
        model_path=model_path,
        data_path=cache_folder,
        batch_size=batch_size,
        grad_accu=grad_accu,
        learning_rate=lr, fp16=fp16,
        epochs=epochs, loss_fn=nn.MSELoss(),
        # optimizer_cls=pls.optimizers.RAdam,
        optimizer_cls=torch.optim.AdamW,
        weight_decay=wd,
        layerwise_decay=layerwise_decay
    )

    sents, embs = joblib.load(Path(cache_folder) / "allnli_train.jbl")
    encoder = load_encoder(
        model_path, None, 256, do_lower_case=True,
        mean_pooling=True, expand_to_dimension=embs.shape[1])
    # print(encoder)
    tokenizer = encoder[0].tokenizer
    train_ds = DistillDataset(tokenizer, sents, embs)
    sents, embs = joblib.load(Path(cache_folder) / "allnli_valid.jbl")
    valid_ds = DistillDataset(tokenizer, sents, embs)
    del sents
    del embs
    train_loader = DataLoader(
        train_ds,
        sampler=SortishSampler(
            train_ds,
            key=lambda x: len(train_ds.text[x]),
            bs=batch_size
        ),
        collate_fn=partial(
            collate_singles,
            pad=0,
            opening_id=encoder[0].cls_token_id,
            closing_id=encoder[0].sep_token_id,
            truncate_length=encoder[0].max_seq_length
        ),
        batch_size=batch_size,
        num_workers=workers
    )
    valid_loader = DataLoader(
        valid_ds,
        sampler=SortSampler(
            valid_ds,
            key=lambda x: len(valid_ds.text[x])
        ),
        collate_fn=partial(
            collate_singles,
            pad=0,
            opening_id=encoder[0].cls_token_id,
            closing_id=encoder[0].sep_token_id,
            truncate_length=encoder[0].max_seq_length
        ),
        batch_size=batch_size,
        num_workers=1
    )

    pl_module = SentenceEncoderModule(
        config, EncoderWrapper(encoder), metrics=(),
        layerwise_decay=config.layerwise_decay
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
        val_check_interval=0.5,
        gradient_clip_val=10,
        max_epochs=epochs,
        # max_steps=steps,
        callbacks=callbacks,
        accumulate_grad_batches=config.grad_accu,
        # auto_scale_batch_size='power' if batch_size is None else None,
        logger=[
            # pl.loggers.TensorBoardLogger(str(CACHE_DIR / "tb_logs"), name=""),
            pls.loggers.ScreenLogger(),
            # pl.loggers.WandbLogger(project="news-similarity")
        ],
        log_every_n_steps=100
    )

    trainer.fit(pl_module, train_dataloader=train_loader, val_dataloaders=valid_loader)

    pl_module.load_state_dict(torch.load(checkpoints.best_model_path)["state_dict"])

    output_folder = CACHE_DIR / f"{encoder[0].__class__.__name__}_distilled"
    encoder.save(str(output_folder))


if __name__ == "__main__":
    typer.run(main)
