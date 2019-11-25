import json
import argparse
from pathlib import Path
from functools import partial
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from helperbot import (
    BaseBot, LearningRateSchedulerCallback,
    MovingAverageStatsTrackerCallback,
    CheckpointCallback,
    MultiStageScheduler, LinearLR,
    AdamW
)
from helperbot.metrics import BinaryAccuracy, AUC
from helperbot.callbacks import Callback

from encoder.dataset import LcqmcDataset
from encoder.dataloading import SortSampler, SortishSampler, collate_pairs
from encoder.components import BertWrapper, PoolingLayer
from encoder.encoder import SentenceEncoder
from encoder.models import SentencePairCosineSimilarity

CACHE_DIR = Path('./cache/')
CACHE_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR = Path('./cache/models/')
MODEL_DIR.mkdir(exist_ok=True, parents=True)


NO_DECAY = [
    'LayerNorm.weight', 'LayerNorm.bias'
]


@dataclass
class CosineSimilarityBot(BaseBot):
    log_dir: Path = MODEL_DIR / "logs/"

    def __post_init__(self):
        super().__post_init__()
        self.loss_format = "%.4f"
        self.metrics = (
            BinaryAccuracy(
                threshold=(0.05, 1.0, 0.05)
            ),
            AUC()
        )

    @staticmethod
    def extract_prediction(output):
        return output


class ScalerDebugCallback(Callback):
    def on_step_ends(self, bot, train_loss, train_weight):
        bot.model.scaler.data.clamp_(0.5, 2.)
        bot.model.shift.data.clamp_(-0.5, 0.5)

    def on_eval_ends(self, bot: BaseBot, metrics: Dict[str, Tuple[float, str]]):
        bot.logger.info(
            "Transformations: %.4f %.4f",
            bot.model.scaler, bot.model.shift)


def get_optimizer(model, lr):
    params = [
        {
            'params': [p for n, p in model.encoder.named_parameters()
                       if not any(nd in n for nd in NO_DECAY)],
            'weight_decay': 0.1
        },
        {
            'params': [p for n, p in model.encoder.named_parameters()
                       if any(nd in n for nd in NO_DECAY)],
            'weight_decay': 0
        }
    ]
    if model.linear_transform:
        params.append(
            {
                'params': [model.scaler, model.shift],
                'weight_decay': 0,
                'lr': 2e-3
            }
        )
    return AdamW(params, lr=lr)


def finetune(args, model, train_loader, valid_loader, criterion):
    total_steps = len(train_loader) * args.epochs
    optimizer = get_optimizer(model, args.lr)
    if args.debug:
        print(
            "No decay:",
            [n for n, p in model.named_parameters()
             if any(nd in n for nd in NO_DECAY)]
        )

    checkpoints = CheckpointCallback(
        keep_n_checkpoints=1,
        checkpoint_dir=CACHE_DIR / "model_cache/",
        monitor_metric="loss"
    )
    lr_durations = [
        int(total_steps*0.2),
        int(np.ceil(total_steps*0.8))
    ]
    break_points = [0] + list(np.cumsum(lr_durations))[:-1]
    callbacks = [
        MovingAverageStatsTrackerCallback(
            avg_window=len(train_loader) // 20,
            log_interval=len(train_loader) // 16
        ),
        LearningRateSchedulerCallback(
            MultiStageScheduler(
                [
                    LinearLR(optimizer, 0.01, lr_durations[0]),
                    CosineAnnealingLR(optimizer, lr_durations[1])
                ],
                start_at_epochs=break_points
            )
        ),
        checkpoints
    ]
    if model.linear_transform:
        callbacks.append(ScalerDebugCallback())
    bot = CosineSimilarityBot(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        clip_grad=10.,
        optimizer=optimizer,
        echo=True,
        criterion=criterion,
        callbacks=callbacks,
        pbar=True,
        use_tensorboard=False,
        use_amp=False
    )
    bot.logger.info("train batch size: %d", train_loader.batch_size)
    bot.train(
        total_steps=total_steps,
        checkpoint_interval=len(train_loader) // 4
    )
    bot.load_model(checkpoints.best_performers[0][1])
    checkpoints.remove_checkpoints(keep=0)
    return bot.model


def load_model(model_path, linear_transform):
    embedder = BertWrapper(
        model_path,
        max_seq_length=128
    )
    pooler = PoolingLayer(
        embedder.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
        layer_to_use=-1
    )
    encoder = SentenceEncoder(modules=[
        embedder, pooler
    ])
    model = SentencePairCosineSimilarity(
        encoder, linear_transform=linear_transform
    )
    return model


def pair_max_len(dataset):
    def pair_max_len_(idx):
        return max(
            len(dataset.text_1[idx]),
            len(dataset.text_2[idx])
        )
    return pair_max_len_


def get_loaders(embedder, args) -> Tuple[DataLoader, DataLoader]:
    ds_train = LcqmcDataset(
        embedder.tokenizer, filename="train.txt", cache_dir=CACHE_DIR)
    train_loader = DataLoader(
        ds_train,
        sampler=SortishSampler(
            ds_train,
            key=pair_max_len(ds_train),
            bs=args.batch_size
        ),
        collate_fn=partial(
            collate_pairs,
            pad=0,
            opening_id=embedder.cls_token_id,
            closing_id=embedder.sep_token_id,
            truncate_length=embedder.max_seq_length
        ),
        batch_size=args.batch_size,
        num_workers=args.workers
    )
    ds_valid = LcqmcDataset(
        embedder.tokenizer, filename="dev.txt", cache_dir=CACHE_DIR)
    valid_loader = DataLoader(
        ds_valid,
        sampler=SortSampler(
            ds_valid,
            key=pair_max_len(ds_valid)
        ),
        collate_fn=partial(
            collate_pairs,
            pad=0,
            opening_id=embedder.cls_token_id,
            closing_id=embedder.sep_token_id,
            truncate_length=embedder.max_seq_length
        ),
        batch_size=args.batch_size * 2,
        num_workers=args.workers
    )
    return train_loader, valid_loader


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('model_path', type=str)
    arg('--batch-size', type=int, default=16)
    arg('--lr', type=float, default=3e-5)
    arg('--workers', type=int, default=2)
    arg('--epochs', type=int, default=3)
    arg('--linear-transform', action='store_true')
    arg('--debug', action='store_true')
    args = parser.parse_args()

    # criterion = nn.L1Loss()
    criterion = nn.MSELoss()
    model = load_model(args.model_path, args.linear_transform)

    train_loader, valid_loader = get_loaders(model.encoder[0], args)
    print(f'{len(train_loader.dataset):,} items in train, '
          f'{len(valid_loader.dataset):,} in valid')

    model = finetune(
        args, model, train_loader,
        valid_loader, criterion
    )

    model.save(str(MODEL_DIR / "tmp"))
    (MODEL_DIR / "tmp" / 'params.json').write_text(
        json.dumps(vars(args), indent=4, sort_keys=True))


if __name__ == '__main__':
    main()
