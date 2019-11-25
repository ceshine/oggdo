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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from helperbot import (
    BaseBot, LearningRateSchedulerCallback,
    MovingAverageStatsTrackerCallback,
    CheckpointCallback,
    MultiStageScheduler, LinearLR,
    AdamW
)
from helperbot.metrics import Top1Accuracy
from helperbot.callbacks import Callback

from encoder.dataset import NewsClassificationDataset
from encoder.dataloading import SortSampler, SortishSampler, collate_singles
from encoder.components import BertWrapper, PoolingLayer
from encoder.encoder import SentenceEncoder
from encoder.models import SentenceClassification
from scripts.bert_eval_custom import convert_t2s

CACHE_DIR = Path('./cache/')
CACHE_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR = Path('./cache/models/')
MODEL_DIR.mkdir(exist_ok=True, parents=True)
DATA_PATH = Path("data/classification_dataset.csv")

NO_DECAY = [
    'LayerNorm.weight', 'LayerNorm.bias'
]


@dataclass
class ClassificationBot(BaseBot):
    log_dir: Path = MODEL_DIR / "logs/"

    def __post_init__(self):
        super().__post_init__()
        self.loss_format = "%.4f"
        self.metrics = (
            Top1Accuracy(),
        )

    @staticmethod
    def extract_prediction(output):
        return output


def get_optimizer(model, lr) -> AdamW:
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
        },
        {
            'params': model.classifier.parameters(),
            'weight_decay': 0.1,
            'lr': lr * 10
        }
    ]
    return AdamW(params, lr=lr)


def finetune(args, model, train_loader, valid_loader, criterion) -> ClassificationBot:
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
        monitor_metric="accuracy"
    )
    lr_durations = [
        int(total_steps*0.2),
        int(np.ceil(total_steps*0.8))
    ]
    break_points = [0] + list(np.cumsum(lr_durations))[:-1]
    callbacks = [
        MovingAverageStatsTrackerCallback(
            avg_window=len(train_loader) // 10,
            log_interval=len(train_loader) // 8
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
    bot = ClassificationBot(
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
        gradient_accumulation_steps=args.grad_accu
    )
    bot.logger.info("train batch size: %d", train_loader.batch_size)
    bot.train(
        total_steps=total_steps,
        checkpoint_interval=len(train_loader) // 2
    )
    bot.load_model(checkpoints.best_performers[0][1])
    checkpoints.remove_checkpoints(keep=0)
    return bot


def load_model(model_path: str, dropout: float, n_classes: int = 4) -> SentenceClassification:
    embedder = BertWrapper(
        model_path,
        max_seq_length=256,
        do_lower_case=False
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
    model = SentenceClassification(
        encoder, n_classes=n_classes, dropout=dropout
    )
    return model


def get_splitted_data(args):
    df = pd.read_csv(DATA_PATH)
    if args.t2s:
        df["text"] = df["text"].apply(convert_t2s)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
    train_idx, rest_idx = next(sss.split(df, df.label))
    df_train = df.iloc[train_idx]
    df_rest = df.iloc[rest_idx]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    valid_idx, test_idx = next(sss.split(df_rest, df_rest.label))
    df_valid = df_rest.iloc[valid_idx]
    df_test = df_rest.iloc[test_idx]
    return df_train, df_valid, df_test


def get_loaders(embedder, args) -> Tuple[DataLoader, DataLoader]:
    df_train, df_valid, df_test = get_splitted_data(args)
    ds_train = NewsClassificationDataset(
        embedder.tokenizer, df_train)
    train_loader = DataLoader(
        ds_train,
        sampler=SortishSampler(
            ds_train,
            key=lambda x: len(ds_train.text[x]),
            bs=args.batch_size
        ),
        collate_fn=partial(
            collate_singles,
            pad=0,
            opening_id=embedder.cls_token_id,
            closing_id=embedder.sep_token_id,
            truncate_length=embedder.max_seq_length
        ),
        batch_size=args.batch_size,
        num_workers=args.workers
    )
    ds_valid = NewsClassificationDataset(
        embedder.tokenizer, df_valid)
    valid_loader = DataLoader(
        ds_valid,
        sampler=SortSampler(
            ds_valid,
            key=lambda x: len(ds_valid.text[x])
        ),
        collate_fn=partial(
            collate_singles,
            pad=0,
            opening_id=embedder.cls_token_id,
            closing_id=embedder.sep_token_id,
            truncate_length=embedder.max_seq_length
        ),
        batch_size=args.batch_size * 2,
        num_workers=args.workers
    )
    ds_test = NewsClassificationDataset(
        embedder.tokenizer, df_test)
    test_loader = DataLoader(
        ds_test,
        sampler=SortSampler(
            ds_test,
            key=lambda x: len(ds_test.text[x])
        ),
        collate_fn=partial(
            collate_singles,
            pad=0,
            opening_id=embedder.cls_token_id,
            closing_id=embedder.sep_token_id,
            truncate_length=embedder.max_seq_length
        ),
        batch_size=args.batch_size * 2,
        num_workers=args.workers
    )
    return train_loader, valid_loader, test_loader


def eval_preds(preds, truths):
    class_preds = torch.argmax(preds, axis=1)
    df = pd.DataFrame({
        "pred": class_preds,
        "truth": truths
    })
    df["correct"] = (df.pred == df.truth)
    print(
        df.groupby("truth")["correct"].agg([
            "mean", "count"
        ])
    )
    print(confusion_matrix(df.pred, df.truth))
    print(f"Overall: {df['correct'].mean() * 100 :.2f}%")
    print("Merging international and politics...")
    df.loc[df["pred"] == 2, "pred"] = 0
    df.loc[df["truth"] == 2, "truth"] = 0
    df["correct"] = (df.pred == df.truth)
    print(
        df.groupby("truth")["correct"].agg([
            "mean", "count"
        ])
    )
    print(confusion_matrix(df.pred, df.truth))
    print(f"Overall: {df['correct'].mean() * 100 :.2f}%")


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('mode', type=str)
    arg('model_path', type=str)
    arg('--t2s', action="store_true")
    arg('--grad-accu', type=int, default=2)
    arg('--batch-size', type=int, default=8)
    arg('--lr', type=float, default=2e-5)
    arg('--workers', type=int, default=2)
    arg('--epochs', type=int, default=3)
    arg('--debug', action='store_true')
    arg('--dropout', type=float, default=0.5)
    args = parser.parse_args()

    if args.mode == "train":
        criterion = nn.CrossEntropyLoss()
        model = load_model(args.model_path, dropout=args.dropout)
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

        bot.model.save(str(MODEL_DIR / "tmp_news_cls"))
        (MODEL_DIR / "tmp_news_cls" / 'params.json').write_text(
            json.dumps(vars(args), indent=4, sort_keys=True))
    elif args.mode == "eval":
        model = SentenceClassification.load(args.model_path)
        _, valid_loader, test_loader = get_loaders(
            model.encoder[0], args)
        print(f'{len(valid_loader.dataset):,} in valid, '
              f'{len(test_loader.dataset):,} in test.')
        bot = ClassificationBot(
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
        eval_preds(preds_val, y_val)
        print("=" * 20)
        print("Test")
        print("=" * 20)
        preds_test, y_test = bot.predict(
            test_loader, return_y=True
        )
        eval_preds(preds_test, y_test)
    else:
        raise ValueError("Unrecognized mode!")


if __name__ == '__main__':
    main()
