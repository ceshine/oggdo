import os
import json
from pathlib import Path
from typing import Optional
from dataclasses import asdict

import torch
import typer
import pytorch_lightning as pl
import pytorch_lightning_spells as pls

from oggdo.dataset import NewsSimilarityDataset
from oggdo.components import TransformerWrapper, PoolingLayer
from oggdo.encoder import SentenceEncoder
from oggdo.models import SentencePairCosineSimilarity
from oggdo.lightning_modules import CosineSimilarityConfig, SentenceEncodingModule, SentencePairDataModule

CACHE_DIR = Path('./cache/')
CACHE_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR = Path('./cache/models/')
MODEL_DIR.mkdir(exist_ok=True, parents=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_model(model_path: str, linear_transform, model_type):
    model_path_ = Path(model_path)
    if (model_path_ / "modules.json").exists():
        encoder = SentenceEncoder(str(model_path))
        encoder[1].pooling_mode_mean_tokens = False
        encoder[1].pooling_mode_cls_token = True
        print(encoder[1].get_config_dict())
    else:
        embedder = TransformerWrapper(
            model_path,
            max_seq_length=256,
            model_type=model_type
        )
        pooler = PoolingLayer(
            embedder.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=False,
            pooling_mode_cls_token=True,
            pooling_mode_max_tokens=False,
            layer_to_use=-1
        )
        encoder = SentenceEncoder(modules=[
            embedder, pooler
        ])
    model = SentencePairCosineSimilarity(
        encoder, linear_transform=linear_transform
    )
    if linear_transform:
        model.scaler.data = torch.tensor([0.6]).to(model.encoder.device)
        model.shift.data = torch.tensor([0.3]).to(model.encoder.device)
    return model


def main(
    model_path: str = typer.Argument("pretrained_models/bert_wwm_ext/"),
    data_path: str = typer.Argument("data/annotated.csv"),
    sample_train: float = -1,
    batch_size: int = 16, grad_accu: int = 2,
    lr: float = 3e-5, workers: int = 4, t2s: bool = False,
    epochs: int = 3, linear_transform: bool = False,
    use_amp: bool = False, wd: float = 0,
    layerwise_decay: float = 0,
    model_type: Optional[str] = None
):
    pl.seed_everything(int(os.environ.get("SEED", 42)))
    model = load_model(model_path, linear_transform, model_type)

    config = CosineSimilarityConfig(
        model_path=model_path,
        data_path=data_path,
        sample_train=sample_train,
        batch_size=batch_size, grad_accu=grad_accu,
        learning_rate=lr, fp16=use_amp,
        epochs=epochs,
        loss_fn=torch.nn.MSELoss(),
        # loss_fn=torch.nn.L1Loss(),
        t2s=t2s, linear_transform=linear_transform,
        optimizer_cls=pls.optimizers.RAdam,
        weight_decay=wd,
        layerwise_decay=layerwise_decay
    )

    pl_module = SentenceEncodingModule(
        config, model,
        layerwise_decay=config.layerwise_decay)

    data_module = SentencePairDataModule(
        model.encoder[0], config,
        dataset_cls=NewsSimilarityDataset, workers=workers,
        name="news"
    )

    checkpoints = pl.callbacks.ModelCheckpoint(
        dirpath=str(CACHE_DIR / "model_checkpoints"),
        monitor='val_spearman',
        mode="max",
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
        # val_check_interval=0.5,
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

    pl_module.load_state_dict(torch.load(checkpoints.best_model_path)["state_dict"])

    trainer.test(datamodule=data_module)

    output_folder = MODEL_DIR / f"news_{Path(model_path).name}"
    model.save(str(output_folder))

    config_dict = asdict(config)
    del config_dict["loss_fn"]
    del config_dict["optimizer_cls"]
    (output_folder / 'params.json').write_text(
        json.dumps(config_dict, indent=4, sort_keys=True))


if __name__ == '__main__':
    typer.run(main)
