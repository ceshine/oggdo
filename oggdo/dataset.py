import os
import enum
from pathlib import Path
from typing import Optional

import joblib
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
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
            tokenizer.encode(text, add_special_tokens=False)
            for text in df.text_1.values
        ])
        self.text_2 = np.asarray([
            tokenizer.encode(text, add_special_tokens=False)
            for text in df.text_2.values
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


class XnliDfDataset(Dataset):
    def __init__(
            self,
            tokenizer,
            df):
        if "text_1" in df:
            df.rename(columns={
                "text_1": "premise",
                "text_2": "hypo"
            }, inplace=True)
        self.text_1 = tokenizer.batch_encode_plus(
            df.premise.values.tolist(), add_special_tokens=False, padding=False
        )["input_ids"]
        self.text_2 = tokenizer.batch_encode_plus(
            df.hypo.values.tolist(), add_special_tokens=False, padding=False
        )["input_ids"]
        self.labels = df["labels"].values

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
            tokenizer.encode(text, add_special_tokens=False)
            for text in df.premise.values
        ])
        self.text_2 = np.asarray([
            tokenizer.encode(text, add_special_tokens=False)
            for text in df.hypo.values
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
            tokenizer.encode(text, add_special_tokens=False)
            for text in df.text.values
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
        if "similarity" in df:
            self.labels = df.similarity.values
        else:
            self.labels = df.labels.values
        self.text_1 = tokenizer.batch_encode_plus(
            df.text_1.values.tolist(), add_special_tokens=False, padding=False
        )["input_ids"]
        self.text_2 = tokenizer.batch_encode_plus(
            df.text_2.values.tolist(), add_special_tokens=False, padding=False
        )["input_ids"]

    def __getitem__(self, item):
        return (
            self.text_1[item],
            self.text_2[item],
            self.labels[item].astype(np.float32)
        )

    def __len__(self):
        return len(self.labels)


class SimilarityDataset(Dataset):
    def __init__(
        self, tokenizer, df,
        sentence_1: str = "text_1",
        sentence_2: str = "text_2",
        label: str = "similarity"
    ):
        self.labels = df[label].values
        self.text_1 = tokenizer.batch_encode_plus(
            df[sentence_1].values.tolist(), add_special_tokens=False, padding=False
        )["input_ids"]
        self.text_2 = tokenizer.batch_encode_plus(
            df[sentence_2].values.tolist(), add_special_tokens=False, padding=False
        )["input_ids"]

    def __getitem__(self, item):
        return (
            self.text_1[item],
            self.text_2[item],
            self.labels[item].astype(np.float32)
        )

    def __len__(self):
        return len(self.labels)


class SentenceDataset(Dataset):
    def __init__(self, tokenizer, sentences):
        self.text = tokenizer.batch_encode_plus(
            sentences, add_special_tokens=False, padding=False
        )["input_ids"]

    def __getitem__(self, item):
        return self.text[item]

    def __len__(self):
        return len(self.text)


class DistillSentenceDataset(Dataset):
    def __init__(self, tokenizer_1, tokenizer_2, sentences):
        self.text_1 = tokenizer_1.batch_encode_plus(
            sentences, add_special_tokens=False, padding=False
        )["input_ids"]
        self.text_2 = tokenizer_2.batch_encode_plus(
            sentences, add_special_tokens=False, padding=False
        )["input_ids"]
        print("Original dataset size:", len(self.text_1))
        matched = [
            len(x[0]) == len(x[1])
            for x in zip(self.text_1, self.text_2)
        ]
        print("Filtered dataset size:", np.sum(matched))
        if np.sum(matched) != len(self.text_1):
            self.text_1 = [x for i, x in enumerate(self.text_1) if matched[i]]
            self.text_2 = [x for i, x in enumerate(self.text_2) if matched[i]]

    def __getitem__(self, item):
        return self.text_1[item], self.text_2[item]

    def __len__(self):
        return len(self.text_1)


class DistillDataset(Dataset):
    def __init__(self, tokenizer, sentences, embeddings):
        self.text = tokenizer.batch_encode_plus(
            sentences, add_special_tokens=False, padding=False
        )["input_ids"]
        self.embeddings = embeddings

    def __getitem__(self, item):
        return self.text[item], self.embeddings[item]

    def __len__(self):
        return len(self.text)


class SBertDataset(enum.Enum):
    AllNLI: str = "allnli"
    Wikipedia: str = "wiki"
    STS: str = "sts"


def http_get(url, path):
    """
    Downloads a URL to a given path on disc
    """
    if os.path.dirname(path) != '':
        os.makedirs(os.path.dirname(path), exist_ok=True)

    req = requests.get(url, stream=True)
    if req.status_code != 200:
        print("Exception when trying to download {}. Response {}".format(url, req.status_code), file=sys.stderr)
        req.raise_for_status()
        return

    download_filepath = path+"_part"
    with open(download_filepath, "wb") as file_binary:
        content_length = req.headers.get('Content-Length')
        total = int(content_length) if content_length is not None else None
        progress = tqdm(unit="B", total=total, unit_scale=True)
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                progress.update(len(chunk))
                file_binary.write(chunk)

    os.rename(download_filepath, path)
    progress.close()


def download_dataset(output_dir: str = "data/", dataset: SBertDataset = SBertDataset.AllNLI) -> Path:
    """Download dataset archives from sbert.net

    Reference: https://github.com/UKPLab/sentence-transformers/blob/7a2c6905d083471ea3ce3850c13bf7ebb604a053/sentence_transformers/util.py
    """
    output_path = Path(output_dir)

    # Download datasets if needed
    if dataset == SBertDataset.AllNLI:
        allnli_path = output_path / "AllNLI.tsv.gz"
        if not allnli_path.exists():
            http_get(
                'https://sbert.net/datasets/AllNLI.tsv.gz',
                str(allnli_path))
        return allnli_path

    if dataset == SBertDataset.Wikipedia:
        wikipedia_path = output_path / "wikipedia-en-sentences.txt.gz"
        if not wikipedia_path.exists():
            http_get(
                'https://sbert.net/datasets/wikipedia-en-sentences.txt.gz',
                str(wikipedia_path))
        return wikipedia_path

    if dataset == SBertDataset.STS:
        sts_path = output_path / "stsbenchmark.tsv.gz"
        if not sts_path.exists():
            http_get(
                'https://sbert.net/datasets/stsbenchmark.tsv.gz',
                str(sts_path))
        return sts_path

    raise ValueError("Unknown dataset!")
