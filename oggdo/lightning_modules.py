import math
from functools import partial
from dataclasses import dataclass, asdict
from typing import Sequence, Callable, Type, Tuple, Dict

import torch
import numpy as np
import pandas as pd
from opencc import OpenCC
import pytorch_lightning as pl
import pytorch_lightning_spells as pls
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedShuffleSplit

from .components import TransformerWrapper
from .dataloading import collate_pairs

T2S = OpenCC('t2s')
NO_DECAY = [
    'LayerNorm.weight', 'LayerNorm.bias'
]


def convert_t2s(text: str) -> str:
    return T2S.convert(text)


def pair_max_len(dataset):
    def pair_max_len_(idx):
        return max(
            len(dataset.text_1[idx]),
            len(dataset.text_2[idx])
        )
    return pair_max_len_


@dataclass
class BaseConfig:
    model_path: str
    data_path: str
    batch_size: int
    fp16: bool
    learning_rate: float
    weight_decay: float
    epochs: int
    # max_len: int
    loss_fn: Callable
    # num_gpus: int = 1
    grad_accu: int = 1
    sample_train: float = -1
    t2s: bool = False
    optimizer_cls: torch.optim.Optimizer = pls.optimizers.RAdam


@dataclass
class CosineSimilarityConfig(BaseConfig):
    linear_transform: bool = False


class SimilarityModule(pls.BaseModule):
    def __init__(
        self, config: BaseConfig, model: torch.nn.Module,
        metrics: Sequence[Tuple[str, pl.metrics.Metric]] = (
            ("spearman", pls.metrics.SpearmanCorrelation(sigmoid=False)),
        )
    ):
        super().__init__()
        self.config = config
        self.save_hyperparameters(asdict(config))
        self.model = model
        self.metrics = metrics

    def forward(self, features: Dict[str, torch.Tensor]):
        return self.model(**features)

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch[0])
        loss = self.config.loss_fn(
            output,
            batch[1]
        )
        return {
            'loss': loss,
            'pred': output,
            'target': batch[1]
        }

    def training_step(self, batch, batch_idx):
        loss = self.config.loss_fn(
            self.forward(batch[0]),
            batch[1]
        )
        return {"loss": loss, "log": batch_idx % self.trainer.accumulate_grad_batches == 0}

    def configure_optimizers(self):
        params = [
            {
                'params': [p for n, p in self.model.encoder.named_parameters()
                           if not any(nd in n for nd in NO_DECAY)],
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in self.model.encoder.named_parameters()
                           if any(nd in n for nd in NO_DECAY)],
                'weight_decay': 0
            }
        ]
        optimizer = self.config.optimizer_cls(
            params, lr=self.config.learning_rate
        )
        steps_per_epochs = math.floor(
            len(self.train_dataloader().dataset) / self.config.batch_size /
            self.config.grad_accu  # / self.num_gpus # dpp mode
        )
        print("Steps per epochs:", steps_per_epochs)
        n_steps = steps_per_epochs * self.config.epochs
        lr_durations = [
            int(n_steps*0.05),
            int(np.ceil(n_steps*0.95)) + 1
        ]
        break_points = [0] + list(np.cumsum(lr_durations))[:-1]
        scheduler = {
            'scheduler': pls.lr_schedulers.MultiStageScheduler(
                [
                    pls.lr_schedulers.LinearLR(
                        optimizer, 0.01, lr_durations[0]),
                    CosineAnnealingLR(optimizer, lr_durations[1])
                ],
                start_at_epochs=break_points
            ),
            'interval': 'step',
            'frequency': 1,
            'strict': True,
        }
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }


class SentencePairDataModule(pl.LightningDataModule):
    def __init__(
        self,
        embedder: TransformerWrapper,
        config: BaseConfig,
        dataset_cls: Type[torch.utils.data.Dataset],
        workers: int = 4
    ):
        super().__init__()
        self.config = config
        self.data_path = self.config.data_path
        self.batch_size = self.config.batch_size
        self.embedder = embedder
        self.t2s = self.config.t2s
        self.dataset_cls = dataset_cls
        self.workers = workers
        self.sample_train = self.config.sample_train

    def setup(self, stage=None):
        df_train, df_valid, df_test = self._get_splitted_data()
        # Tokenize:
        if stage == "fit" or stage is None:
            self.ds_train = self.dataset_cls(
                self.embedder.tokenizer, df_train)
            if self.sample_train > 0 and self.sample_train < 1:
                df_train = df_train.sample(frac=self.sample_train)
            self.ds_valid = self.dataset_cls(
                self.embedder.tokenizer, df_valid)
            print(f'{len(self.ds_train):,} items in train, '
                  f'{len(self.ds_valid):,} in valid.')
        elif stage == "test" or stage is None:
            self.ds_test = self.dataset_cls(
                self.embedder.tokenizer, df_test)
            print(f'{len(self.ds_test):,} in test.')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.ds_train,
            sampler=pls.samplers.SortishSampler(
                self.ds_train,
                key=pair_max_len(self.ds_train),
                bs=self.batch_size
            ),
            collate_fn=partial(
                collate_pairs,
                pad=0,
                opening_id=self.embedder.cls_token_id,
                closing_id=self.embedder.sep_token_id,
                truncate_length=self.embedder.max_seq_length
            ),
            batch_size=self.batch_size,
            num_workers=self.workers
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.ds_valid,
            sampler=pls.samplers.SortSampler(
                self.ds_valid,
                key=pair_max_len(self.ds_valid)
            ),
            collate_fn=partial(
                collate_pairs,
                pad=0,
                opening_id=self.embedder.cls_token_id,
                closing_id=self.embedder.sep_token_id,
                truncate_length=self.embedder.max_seq_length
            ),
            batch_size=self.batch_size * 2,
            num_workers=self.workers
        )
        # print(next(iter(loader)))
        # return loader

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.ds_test,
            sampler=pls.samplers.SortSampler(
                self.ds_test,
                key=pair_max_len(self.ds_test)
            ),
            collate_fn=partial(
                collate_pairs,
                pad=0,
                opening_id=self.embedder.cls_token_id,
                closing_id=self.embedder.sep_token_id,
                truncate_length=self.embedder.max_seq_length
            ),
            batch_size=self.batch_size * 2,
            num_workers=self.workers
        )

    def _get_splitted_data(self):
        df = pd.read_csv(self.data_path)
        if self.t2s:
            df["text"] = df["text"].apply(convert_t2s)
        if "similarity" in df:
            df["labels"] = df.similarity.astype("category")
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=0.3, random_state=666)
        train_idx, rest_idx = next(sss.split(df, df.labels))
        df_train = df.iloc[train_idx]
        df_rest = df.iloc[rest_idx]
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=0.5, random_state=666)
        valid_idx, test_idx = next(sss.split(df_rest, df_rest.labels))
        df_valid = df_rest.iloc[valid_idx]
        df_test = df_rest.iloc[test_idx]
        return df_train, df_valid, df_test

    # To make linters happy
    def prepare_data(self):
        pass

    def transfer_batch_to_device(self, batch, device):
        return super().transfer_batch_to_device(batch, device)
