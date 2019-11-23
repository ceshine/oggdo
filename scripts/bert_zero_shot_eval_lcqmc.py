import logging
import argparse
from functools import partial

import numpy as np
from scipy.stats import spearmanr
import torch
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from encoder.dataset import LcqmcDataset
from encoder.dataloading import SortSampler, collate_pairs
from encoder.components import BertWrapper, PoolingLayer
from encoder.encoder import SentenceEncoder
from encoder.models import SentencePairCosineSimilarity

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)


def main(args):
    embedder = BertWrapper(
        args.model_path,
        max_seq_length=256
    )
    pooler = PoolingLayer(
        embedder.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
        layer_to_use=args.layer
    )
    encoder = SentenceEncoder(modules=[
        embedder, pooler
    ])
    model = SentencePairCosineSimilarity(
        encoder, linear_transform=False
    )
    model.eval()

    dataset = LcqmcDataset(embedder.tokenizer, filename=args.filename)
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
    spearman_score = spearmanr(
        preds, references
    )
    print(f"Spearman: {spearman_score.correlation:.4f}")

    print(f"Pred Min: {np.min(preds)}, {np.max(preds)}")
    best_thres, best_acc = -1, -1
    for threshold in np.arange(0.05, 1, 0.05):
        binarized = (preds > threshold).astype("int")
        acc = (binarized == references).sum() / len(references)
        if acc > best_acc:
            best_acc = acc
            best_thres = threshold
    print(f"Best acc: {best_acc:.4f} @ {best_thres:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model-path', type=str, default="pretrained_models/bert_wwm_ext/")
    arg('--filename', type=str, default="test.txt")
    arg('--layer', type=int, default=-2)
    args = parser.parse_args()
    main(args)
