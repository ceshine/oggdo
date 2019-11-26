import json
import argparse
from pathlib import Path
from functools import partial
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import ShuffleSplit
from scipy.stats import spearmanr
from helperbot import (
    BaseBot, LearningRateSchedulerCallback,
    MovingAverageStatsTrackerCallback,
    CheckpointCallback,
    MultiStageScheduler, LinearLR,
    AdamW
)
from helperbot.metrics import Metric

from oggdo.dataset import NewsSimilarityDataset
from oggdo.dataloading import SortSampler, SortishSampler, collate_pairs
from oggdo.components import BertWrapper, PoolingLayer
from oggdo.encoder import SentenceEncoder
from oggdo.models import SentencePairCosineSimilarity
from .finetune_lcqmc import CosineSimilarityBot, ScalerDebugCallback, pair_max_len
from scripts.bert_eval_custom import convert_t2s

CACHE_DIR = Path('./cache/')
CACHE_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR = Path('./cache/models/')
MODEL_DIR.mkdir(exist_ok=True, parents=True)
DATA_PATH = Path("data/annotated.csv")

NO_DECAY = [
    'LayerNorm.weight', 'LayerNorm.bias'
]


class SpearmanCorrelation(Metric):
    """Spearman Correlation"""
    name = "spearman"

    def __call__(self, truth: torch.Tensor, pred: torch.Tensor) -> Tuple[float, str]:
        spearman_score = spearmanr(truth.numpy(), pred.numpy()).correlation
        return spearman_score * -1, f"{spearman_score:.4f}"


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
                'lr': 3e-2
            }
        )
    return AdamW(params, lr=lr)


def finetune(args, model, train_loader, valid_loader, criterion) -> CosineSimilarityBot:
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
            avg_window=len(train_loader) // 10,
            log_interval=len(train_loader) // 7
        ),
        LearningRateSchedulerCallback(
            MultiStageScheduler(
                [
                    LinearLR(optimizer, 0.01, lr_durations[0]),
                    LinearLR(optimizer, 0.001, lr_durations[1], upward=False)
                    # CosineAnnealingLR(optimizer, lr_durations[1])
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
        use_amp=False,
        gradient_accumulation_steps=args.grad_accu,
        metrics=(SpearmanCorrelation(),)
    )
    bot.logger.info("train batch size: %d", train_loader.batch_size)
    bot.train(
        total_steps=total_steps,
        checkpoint_interval=len(train_loader) // 2
    )
    bot.load_model(checkpoints.best_performers[0][1])
    checkpoints.remove_checkpoints(keep=0)
    return bot


def load_model(model_path: str, linear_transform):
    model_path = Path(model_path)
    if (model_path / "modules.json").exists():
        encoder = SentenceEncoder(str(model_path))
        encoder[1].pooling_mode_mean_tokens = True
        encoder[1].pooling_mode_cls_token = False
        print(encoder[1].get_config_dict())
    else:
        embedder = BertWrapper(
            model_path,
            max_seq_length=256
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
    if linear_transform:
        model.scaler.data = torch.tensor([1.]).to(model.encoder.device)
        model.shift.data = torch.tensor([-0.1]).to(model.encoder.device)
    return model


def get_splitted_data(args):
    df = pd.read_csv(DATA_PATH)
    if args.t2s:
        df["text_1"] = df["text_1"].apply(convert_t2s)
        df["text_2"] = df["text_2"].apply(convert_t2s)
    sss = ShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, rest_idx = next(sss.split(df))
    df_train = df.iloc[train_idx]
    df_rest = df.iloc[rest_idx]
    sss = ShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    valid_idx, test_idx = next(sss.split(df_rest))
    df_valid = df_rest.iloc[valid_idx]
    df_test = df_rest.iloc[test_idx]
    return df_train, df_valid, df_test


def get_loaders(embedder, args) -> Tuple[DataLoader, DataLoader]:
    df_train, df_valid, df_test = get_splitted_data(args)
    ds_train = NewsSimilarityDataset(
        embedder.tokenizer, df_train)
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
    ds_valid = NewsSimilarityDataset(
        embedder.tokenizer, df_valid)
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
    ds_test = NewsSimilarityDataset(
        embedder.tokenizer, df_test)
    test_loader = DataLoader(
        ds_test,
        sampler=SortSampler(
            ds_test,
            key=pair_max_len(ds_test)
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
    return train_loader, valid_loader, test_loader


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('mode', type=str)
    arg('model_path', type=str)
    arg('--batch-size', type=int, default=16)
    arg('--grad-accu', type=int, default=2)
    arg('--lr', type=float, default=3e-5)
    arg('--workers', type=int, default=2)
    arg('--t2s', action="store_true")
    arg('--epochs', type=int, default=3)
    arg('--linear-transform', action='store_true')
    arg('--debug', action='store_true')
    args = parser.parse_args()

    if args.mode == "train":
        # criterion = nn.L1Loss()
        criterion = nn.MSELoss()
        model = load_model(args.model_path, args.linear_transform)

        train_loader, valid_loader, test_loader = get_loaders(
            model.encoder[0], args)

        print(f'{len(train_loader.dataset):,} items in train, '
              f'{len(valid_loader.dataset):,} in valid, '
              f'{len(test_loader.dataset):,} in test.')

        bot = finetune(
            args, model, train_loader,
            valid_loader, criterion
        )

        test_metrics = bot.eval(test_loader)
        print("Test metrics:", test_metrics)

        model.save(str(MODEL_DIR / "tmp_news_sim"))
        (MODEL_DIR / "tmp_news_sim" / 'params.json').write_text(
            json.dumps(vars(args), indent=4, sort_keys=True))
    elif args.mode == "eval":
        model = SentencePairCosineSimilarity.load(args.model_path)
        _, valid_loader, test_loader = get_loaders(
            model.encoder[0], args)
        print(f'{len(valid_loader.dataset):,} in valid, '
              f'{len(test_loader.dataset):,} in test.')
        bot = CosineSimilarityBot(
            model=model,
            train_loader=valid_loader,
            valid_loader=valid_loader,
            optimizer=None,
            echo=True,
            criterion=None,
            callbacks=None,
            pbar=False,
            use_tensorboard=False,
            use_amp=False
        )
        print("=" * 20)
        print("Validation")
        print("=" * 20)
        preds_val, y_val = bot.predict(
            valid_loader, return_y=True
        )
        spearman_score = spearmanr(
            preds_val, y_val
        )
        print(f"Spearman: {spearman_score.correlation:.4f}")
        print("=" * 20)
        print("Test")
        print("=" * 20)
        preds_test, y_test = bot.predict(
            test_loader, return_y=True
        )
        spearman_score = spearmanr(
            preds_test, y_test
        )
        print(f"Spearman: {spearman_score.correlation:.4f}")
    else:
        raise ValueError("Unrecognized mode!")


if __name__ == '__main__':
    main()
