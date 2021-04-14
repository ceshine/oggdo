import math
import warnings
from pathlib import Path
from functools import partial
from dataclasses import dataclass, asdict
from typing import Sequence, Callable, Type, Tuple, Dict, Optional

import torch
import joblib
import numpy as np
import pandas as pd
from opencc import OpenCC
import pytorch_lightning as pl
import pytorch_lightning_spells as pls
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedShuffleSplit

from .components import TransformerWrapper
from .dataloading import collate_pairs
from .dataset import SBertDataset

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
    layerwise_decay: float
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


@dataclass
class DistillConfig(BaseConfig):
    teacher_model_path: str = ""
    dataset: SBertDataset = SBertDataset.AllNLI
    attn_loss_weight: float = 1.


class SentenceEncodingModule(pls.BaseModule):
    def __init__(
        self, config: BaseConfig, model: torch.nn.Module,
        metrics: Sequence[Tuple[str, pl.metrics.Metric]] = (
            ("spearman", pls.metrics.SpearmanCorrelation(sigmoid=False)),
        ), layerwise_decay: float = 0
    ):
        warnings.warn(
            '"layerwise_decay" is deprecated. Use config.layerwise_decay instead.',
            DeprecationWarning
        )
        print("layerwise_decay parameter ")
        super().__init__()
        self.config = config
        self.save_hyperparameters(asdict(config))
        self.model = model
        self.metrics = metrics
        self.layerwise_decay = self.config.layerwise_decay

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
        if self.layerwise_decay > 0 and self.layerwise_decay < 1:
            transformer = self.model.encoder[0].transformer
            others = [(n, p) for n, p in self.model.named_parameters()
                      if (".layer." not in n) and ("embeddings." not in n)]
            params = [
                {
                    'params': [p for n, p in others
                               if not any(nd in n for nd in NO_DECAY)],
                    'weight_decay': self.config.weight_decay,
                    "lr": self.config.learning_rate
                },
                {
                    'params': [p for n, p in others
                               if any(nd in n for nd in NO_DECAY)],
                    'weight_decay': 0,
                    "lr":  self.config.learning_rate
                }
            ]
            for i, layer in enumerate(reversed(transformer.encoder.layer)):
                params.extend([
                    {
                        'params': [p for n, p in layer.named_parameters()
                                   if not any(nd in n for nd in NO_DECAY)],
                        'weight_decay': self.config.weight_decay,
                        "lr": self.config.learning_rate * (self.layerwise_decay ** i)
                    },
                    {
                        'params': [p for n, p in layer.named_parameters()
                                   if any(nd in n for nd in NO_DECAY)],
                        'weight_decay': 0,
                        "lr": self.config.learning_rate * (self.layerwise_decay ** i)
                    }
                ])
            params.extend([
                {
                    'params': [p for n, p in transformer.embeddings.named_parameters()
                               if not any(nd in n for nd in NO_DECAY)],
                    'weight_decay': self.config.weight_decay * (self.layerwise_decay ** len(transformer.encoder.layer)),
                    "lr": self.config.learning_rate
                },
                {
                    'params': [p for n, p in transformer.embeddings.named_parameters()
                               if any(nd in n for nd in NO_DECAY)],
                    'weight_decay': 0,
                    "lr": self.config.learning_rate * (self.layerwise_decay ** len(transformer.encoder.layer))
                }
            ])

        else:
            params = [
                {
                    'params': [p for n, p in self.model.named_parameters()
                               if not any(nd in n for nd in NO_DECAY)],
                    'weight_decay': self.config.weight_decay,
                    "lr":  self.config.learning_rate
                },
                {
                    'params': [p for n, p in self.model.named_parameters()
                               if any(nd in n for nd in NO_DECAY)],
                    'weight_decay': 0,
                    "lr":  self.config.learning_rate
                }
            ]
        optimizer = self.config.optimizer_cls(params)
        # print(optimizer)
        steps_per_epochs = math.floor(
            len(self.train_dataloader().dataset) / self.config.batch_size /
            self.config.grad_accu  # / self.num_gpus # dpp mode
        )
        print("Steps per epochs:", steps_per_epochs)
        n_steps = steps_per_epochs * self.config.epochs
        lr_durations = [
            int(n_steps*0.1),
            int(np.ceil(n_steps*0.9)) + 1
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


class NliModule(SentenceEncodingModule):
    """Returns predicted labels instead of raw values when validating"""

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch[0])
        loss = self.config.loss_fn(
            output,
            batch[1]
        )
        return {
            'loss': loss,
            'pred': torch.argmax(torch.softmax(output, dim=-1), dim=-1),
            'target': batch[1]
        }


class DistillModule(SentenceEncodingModule):
    """Get sentence embeddings and attentions from the teacher model in real-time
    """

    def __init__(
        self, config: DistillConfig,
        teacher_model: torch.nn.Module,
        student_model: torch.nn.Module,
        metrics: Sequence[Tuple[str, pl.metrics.Metric]] = (),
        attn_loss_weight: float = 1.
    ):
        super().__init__(
            config, student_model, metrics
        )
        self.teacher_model = teacher_model.eval()
        self.train_attn_loss_tracker = pls.utils.EMATracker(0.02)
        self.attn_loss_weight = config.attn_loss_weight

    def forward(self, features: Sequence[Dict[str, torch.Tensor]]):
        with torch.no_grad():
            teacher_output = self.teacher_model(features[0])
        student_output = self.model(features[1])
        return teacher_output, student_output

    def _get_losses(self, batch, teacher_output, student_output):
        emb_loss = self.config.loss_fn(
            student_output["sentence_embeddings"],
            teacher_output["sentence_embeddings"].detach()
        )
        assert (
            teacher_output["attentions"].shape[1] %
            student_output["attentions"].shape[1] == 0)
        div = teacher_output["attentions"].shape[1] / student_output["attentions"].shape[1]
        student_attn = student_output["attentions"]
        student_attn = torch.where(
            student_attn <= -1e2,
            torch.zeros_like(student_attn).to(student_attn.device),
            student_attn
        )
        teacher_attn = torch.stack(
            [
                teacher_output["attentions"][:, i].detach()
                for i in range(teacher_output["attentions"].shape[1])
                if i % div == (div-1)
            ],
            dim=1
        )
        teacher_attn = torch.where(
            teacher_attn <= -1e2,
            torch.zeros_like(teacher_attn).to(teacher_attn.device),
            teacher_attn
        )
        attn_loss = self.config.loss_fn(
            student_attn,
            teacher_attn
        ) * torch.numel(batch[0]['input_mask']) / batch[0]['input_mask'].sum()
        return emb_loss, attn_loss

    def validation_step(self, batch, batch_idx):
        teacher_output, student_output = self.forward(batch)
        emb_loss, attn_loss = self._get_losses(batch, teacher_output, student_output)
        loss = emb_loss + attn_loss * self.attn_loss_weight
        return {
            'loss': loss,
            'emb_loss': emb_loss,
            'attn_loss': attn_loss,
            'pred': student_output["sentence_embeddings"],
            'target': teacher_output["sentence_embeddings"]
        }

    def training_step(self, batch, batch_idx):
        teacher_output, student_output = self.forward(batch)
        emb_loss, attn_loss = self._get_losses(batch, teacher_output, student_output)
        loss = emb_loss + attn_loss * self.attn_loss_weight
        return {
            "loss": loss,
            'emb_loss': emb_loss,
            'attn_loss': attn_loss,
            "log": batch_idx % self.trainer.accumulate_grad_batches == 0
        }

    def training_step_end(self, outputs):
        """Requires training_step() to return a dictionary with a "loss" entry."""
        self.train_loss_tracker.update(outputs["loss"].detach())
        self.train_attn_loss_tracker.update(outputs["attn_loss"].detach())
        if self._should_log(outputs["log"]):
            self.logger.log_metrics({
                "train_loss": self.train_loss_tracker.value,
                "train_attn_loss": self.train_attn_loss_tracker.value
            }, step=self.global_step)
        self.log(
            "attn_loss",
            self.train_attn_loss_tracker.value * self.attn_loss_weight,
            prog_bar=True, on_step=True, on_epoch=False)
        return outputs["loss"]

    def validation_step_end(self, outputs):
        """Requires validation_step() to return a dictionary with the following entries:

            1. val_loss
            2. pred
            3. target
        """
        self.log('val_loss', outputs['loss'].mean())
        self.log('val_attn_loss', outputs['attn_loss'].mean())
        self.log('val_emb_loss', outputs['emb_loss'].mean())
        for name, metric in self.metrics:
            metric(
                outputs['pred'].view(-1).cpu(),
                outputs['target'].view(-1).cpu()
            )
            self.log("val_" + name, metric)


class SentencePairDataModule(pl.LightningDataModule):
    def __init__(
        self,
        embedder: TransformerWrapper,
        config: BaseConfig,
        dataset_cls: Type[torch.utils.data.Dataset],
        workers: int = 4,
        cache_dir: Optional[str] = "cache/tokenized/",
        name: str = ""
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
        self.cache_dir = cache_dir
        self.name = name
        self.ds_train, self.ds_valid, self.ds_test = None, None, None

    def setup(self, stage=None):
        df_train, df_valid, df_test = self._get_splitted_data()
        # Tokenize:
        if stage == "fit" or stage is None:
            if self.cache_dir:
                cache_path = Path(self.cache_dir) / f"{self.name}_fit.cache"
                if cache_path.exists():
                    print("Loading cached dataset...")
                    self.ds_train, self.ds_valid = joblib.load(
                        cache_path
                    )
            if (self.ds_train is None) or (self.ds_train is None):
                if self.sample_train > 0 and self.sample_train < 1:
                    df_train = df_train.sample(frac=self.sample_train)
                self.ds_train = self.dataset_cls(
                    self.embedder.tokenizer, df_train)
                self.ds_valid = self.dataset_cls(
                    self.embedder.tokenizer, df_valid)
                if self.cache_dir:
                    cache_path = Path(self.cache_dir) / f"{self.name}_fit.cache"
                    joblib.dump([self.ds_train, self.ds_valid], cache_path)
            print(df_train.shape, df_valid.shape)
            print(f'{len(self.ds_train):,} items in train, '
                  f'{len(self.ds_valid):,} items in valid.')
        if stage == "test" or stage is None:
            if self.cache_dir:
                cache_path = Path(self.cache_dir) / f"{self.name}_test.cache"
                if cache_path.exists():
                    self.ds_test = joblib.load(
                        cache_path
                    )
            if self.ds_test is None:
                self.ds_test = self.dataset_cls(
                    self.embedder.tokenizer, df_test)
                if self.cache_dir:
                    cache_path = Path(self.cache_dir) / f"{self.name}_test.cache"
                    joblib.dump(self.ds_test, cache_path)
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
            for col in ("text_1", "text_2", "premise", "hypo"):
                if col in df:
                    df[col] = df[col].apply(convert_t2s)
        if "similarity" in df:
            df["labels"] = df.similarity.astype("category")
        if "hypo" in df:
            df["labels"] = np.zeros(df.label.shape[0], dtype=np.int64)
            df.loc[df.label == "neutral", "labels"] = 1
            df.loc[df.label == "entailment", "labels"] = 2
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
