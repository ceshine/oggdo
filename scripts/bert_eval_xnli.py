import logging
import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from encoder.dataset import XnliDataset
from encoder.dataloading import SortSampler, collate_pairs
from encoder.models import SentencePairNliClassification


logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)


def orig(args, model):
    encoder = model.encoder
    embedder = model.encoder[0]

    dataset = XnliDataset(embedder.tokenizer, filename=args.filename)
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
        batch_size=16
    )
    preds, references = [], []
    with torch.no_grad():
        for features, labels in tqdm(loader):
            for name in features:
                features[name] = features[name].to(encoder.device)
            preds.append(
                model(features).cpu().numpy()
            )
            references.append(labels.cpu().numpy())

    preds = np.concatenate(preds)
    references = np.concatenate(references)

    return preds, references


def main(args):
    model = SentencePairNliClassification.load(args.model_path)
    model.eval()
    preds, references = orig(args, model)

    class_preds = np.argmax(preds, axis=1)
    acc = (class_preds == references).sum() / len(references)
    print(f"Accuracy: {acc * 100:.4f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model-path', type=str, default="pretrained_models/bert_wwm_ext/")
    arg('--filename', type=str, default="test.csv")
    arg('--threshold', type=float, default=-1)
    args = parser.parse_args()
    main(args)
