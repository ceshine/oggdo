import csv
import gzip
import logging
from functools import partial
# from pathlib import Path

import typer
import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from oggdo.utils import features_to_device
from oggdo.dataset import SimilarityDataset, download_dataset, SBertDataset
from oggdo.dataloading import SortSampler, collate_pairs
from oggdo.models import SentencePairCosineSimilarity

from common import load_encoder

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)


def orig(encoder, df, batch_size: int):
    embedder = encoder[0]
    model = SentencePairCosineSimilarity(encoder)

    dataset = SimilarityDataset(
        embedder.tokenizer, df,
        "sentence1", "sentence2", "score"
    )
    loader = DataLoader(
        dataset,
        sampler=SortSampler(
            dataset,
            key=lambda x: max(
                len(dataset.text_1[x]),
                len(dataset.text_2[x])
            )
        ),
        collate_fn=partial(
            collate_pairs,
            pad=0,
            opening_id=embedder.cls_token_id,
            closing_id=embedder.sep_token_id,
            truncate_length=embedder.max_seq_length
        ),
        batch_size=batch_size
    )
    preds, references = [], []
    with torch.no_grad():
        for features, labels in tqdm(loader, ncols=100):
            features = features_to_device(features, encoder.device)
            preds.append(
                model(**features).cpu().numpy()
            )
            references.append(labels.cpu().numpy())

    preds = np.concatenate(preds)
    references = np.concatenate(references)

    return preds, references


def main(model_path: str, split: str = "test"):
    file_path = download_dataset(dataset=SBertDataset.STS)
    with gzip.open(file_path, 'rt', encoding='utf8') as fIn:
        df = pd.read_csv(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    df = df[df.split == split].reset_index(drop=True)

    encoder = load_encoder(
        model_path, None, max_length=256, do_lower_case=True, mean_pooling=True
    )
    encoder.eval()
    preds, references = orig(encoder, df, batch_size=32)

    spearman_score = spearmanr(
        preds, references
    )
    print(f"Spearman: {spearman_score.correlation:.4f}")


if __name__ == "__main__":
    typer.run(main)
