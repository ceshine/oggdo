from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class LcqmcDataset(Dataset):
    def __init__(
            self,
            tokenizer,
            data_dir: str = "data/LCQMC/",
            filename: str = "train.txt",
            cache_dir: Optional[str] = None):
        if cache_dir is not None:
            cache_path = Path(cache_dir) / (filename + ".cache")
            if cache_path.exists():
                self.text_1, self.text_2, self.labels = joblib.load(
                    cache_path
                )
                return
        df = pd.read_csv(
            Path(data_dir) / filename, delimiter="\t",
            header=None, names=["text_1", "text_2", "label"]
        )
        self.labels = df.label.values
        self.text_1 = np.asarray([
            tokenizer.encode(text) for text in df.text_1.values
        ])
        self.text_2 = np.asarray([
            tokenizer.encode(text) for text in df.text_2.values
        ])
        if cache_dir is not None:
            joblib.dump(
                [self.text_1, self.text_2, self.labels],
                cache_path
            )

    def __getitem__(self, item):
        return (
            self.text_1[item],
            self.text_2[item],
            self.labels[item]
        )

    def __len__(self):
        return len(self.labels)


class XnliDataset(Dataset):
    def __init__(
            self,
            tokenizer,
            data_dir: str = "data/XNLI-1.0/",
            filename: str = "train.csv",
            cache_dir: Optional[str] = None):
        if cache_dir is not None:
            cache_path = Path(cache_dir) / (filename + ".xnli.cache")
            if cache_path.exists():
                self.text_1, self.text_2, self.labels = joblib.load(
                    cache_path
                )
                return
        df = pd.read_csv(Path(data_dir) / filename)
        self.labels = np.zeros(df.label.shape[0], dtype=np.int64)
        self.labels[df.label == "neutral"] = 1
        self.labels[df.label == "entailment"] = 2
        self.text_1 = np.asarray([
            tokenizer.encode(text) for text in df.premise.values
        ])
        self.text_2 = np.asarray([
            tokenizer.encode(text) for text in df.hypo.values
        ])
        if cache_dir is not None:
            joblib.dump(
                [self.text_1, self.text_2, self.labels],
                cache_path
            )

    def __getitem__(self, item):
        return (
            self.text_1[item],
            self.text_2[item],
            self.labels[item]
        )

    def __len__(self):
        return len(self.labels)


class NewsClassificationDataset(Dataset):
    def __init__(
            self, tokenizer, df):
        # drop headlines for now
        df = df[df.label != "headlines"]
        # politics
        self.labels = np.zeros(df.label.shape[0], dtype=np.int64)
        self.labels[df.label == "society"] = 1
        self.labels[df.label == "international"] = 2
        self.labels[df.label == "taiwan"] = 3
        self.text = np.asarray([
            tokenizer.encode(text) for text in df.text.values
        ])

    def __getitem__(self, item):
        return (
            self.text[item],
            self.labels[item]
        )

    def __len__(self):
        return len(self.labels)


class NewsSimilarityDataset(Dataset):
    def __init__(
            self, tokenizer, df):
        # politics
        self.labels = df.similarity.values
        self.text_1 = np.asarray([
            tokenizer.encode(text) for text in df.text_1.values
        ])
        self.text_2 = np.asarray([
            tokenizer.encode(text) for text in df.text_2.values
        ])

    def __getitem__(self, item):
        return (
            self.text_1[item],
            self.text_2[item],
            self.labels[item]
        )

    def __len__(self):
        return len(self.labels)
