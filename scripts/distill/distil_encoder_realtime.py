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
from oggdo.dataset import SBertDataset, DistillSentenceDataset
from oggdo.dataloading import SortSampler, SortishSampler, collate_distill
from oggdo.lightning_modules import DistillConfig, DistillModule
from common import load_encoder, get_splits


CACHE_DIR = Path("cache/")
MEMORY = joblib.Memory("cache/joblib_memory", verbose=0)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class EncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, **batch):
        return self.encoder(batch)["sentence_embeddings"]


@MEMORY.cache
def get_datasets(teacher_model_path, student_model_path, dataset):
    train_sents, valid_sents, _ = get_splits("data/", dataset)
    teacher_encoder = load_encoder(
        teacher_model_path, None, 256, do_lower_case=True,
        mean_pooling=True).eval()
    teacher_config_dict = teacher_encoder[0].transformer.config.to_dict()
    dim_expand = (
        teacher_config_dict.get("dim") or
        teacher_config_dict.get("hidden_size")
    )
    student_encoder = load_encoder(
        student_model_path, None, 256, do_lower_case=True,
        mean_pooling=True,
        expand_to_dimension=dim_expand)
    student_tokenizer = student_encoder[0].tokenizer
    teacher_tokenizer = teacher_encoder[0].tokenizer
    train_ds = DistillSentenceDataset(
        student_tokenizer, teacher_tokenizer, train_sents)
    valid_ds = DistillSentenceDataset(
        student_tokenizer, teacher_tokenizer, valid_sents)
    return teacher_encoder, student_encoder, train_ds, valid_ds


def main(
    teacher_model_path: str,
    student_model_path: str = "nreimers/TinyBERT_L-4_H-312_v2",
    dataset: SBertDataset = "allnli",
    batch_size: int = 32,
    fp16: bool = False, workers: int = 2, grad_accu: int = 1,
    lr: float = 3e-5, epochs: int = 2, wd: float = 0,
    layerwise_decay: bool = False, attn_loss_weight: float = 1.
):
    pl.seed_everything(int(os.environ.get("SEED", 42)))

    config = DistillConfig(
        model_path=student_model_path,
        teacher_model_path=teacher_model_path,
        dataset=dataset,
        data_path="",
        batch_size=batch_size,
        grad_accu=grad_accu,
        learning_rate=lr, fp16=fp16,
        epochs=epochs, loss_fn=nn.MSELoss(),
        # optimizer_cls=pls.optimizers.RAdam,
        optimizer_cls=torch.optim.AdamW,
        weight_decay=wd,
        layerwise_decay=layerwise_decay,
        attn_loss_weight=attn_loss_weight
    )

    teacher_encoder, student_encoder, train_ds, valid_ds = get_datasets(
        teacher_model_path, student_model_path, dataset=dataset
    )
    teacher_encoder.eval()
    pls.utils.set_trainable(teacher_encoder, False)
    teacher_encoder[0].attentions = True
    student_encoder[0].attentions = True
    print(len(train_ds), len(valid_ds))
    train_loader = DataLoader(
        train_ds,
        sampler=SortishSampler(
            train_ds,
            key=lambda x: len(train_ds.text_1[x]),
            bs=batch_size
        ),
        collate_fn=partial(
            collate_distill,
            pad=0,
            opening_id=teacher_encoder[0].cls_token_id,
            closing_id=teacher_encoder[0].sep_token_id,
            truncate_length=teacher_encoder[0].max_seq_length
        ),
        batch_size=batch_size,
        num_workers=workers
    )
    valid_loader = DataLoader(
        valid_ds,
        sampler=SortSampler(
            valid_ds,
            key=lambda x: len(valid_ds.text_1[x])
        ),
        collate_fn=partial(
            collate_distill,
            pad=0,
            opening_id=teacher_encoder[0].cls_token_id,
            closing_id=teacher_encoder[0].sep_token_id,
            truncate_length=teacher_encoder[0].max_seq_length
        ),
        batch_size=batch_size,
        num_workers=1
    )

    pl_module = DistillModule(
        config, teacher_encoder, student_encoder, metrics=()
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

    loggers = [
        pl.loggers.TensorBoardLogger(str(CACHE_DIR / "tb_logs_distill"), name=""),
        pls.loggers.ScreenLogger(),
    ]
    if os.environ.get("WANDB_PROJ"):
        loggers.append(pl.loggers.WandbLogger(project=os.environ["WANDB_PROJ"]))
    trainer = pl.Trainer(
        # amp_backend="apex", amp_level='O2',
        precision=16 if config.fp16 else 32,
        gpus=1,
        val_check_interval=0.5 if dataset is SBertDataset.AllNLI else 1.,
        gradient_clip_val=10,
        max_epochs=epochs,
        # max_steps=steps,
        callbacks=callbacks,
        accumulate_grad_batches=config.grad_accu,
        # auto_scale_batch_size='power' if batch_size is None else None,
        logger=loggers,
        log_every_n_steps=100
    )

    trainer.fit(pl_module, train_dataloader=train_loader, val_dataloaders=valid_loader)

    pl_module.load_state_dict(torch.load(checkpoints.best_model_path)["state_dict"])

    output_folder = CACHE_DIR / f"{student_encoder[0].transformer.__class__.__name__}_distilled"
    student_encoder.save(str(output_folder))


if __name__ == "__main__":
    typer.run(main)
