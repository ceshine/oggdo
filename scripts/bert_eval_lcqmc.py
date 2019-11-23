import logging
import argparse
from functools import partial

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from sklearn.metrics.pairwise import paired_cosine_distances

from encoder.dataset import LcqmcDataset
from encoder.dataloading import SortSampler, collate_pairs
from encoder.components import BertWrapper, PoolingLayer
from encoder.encoder import SentenceEncoder
from encoder.models import SentencePairCosineSimilarity
from encoder.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction


logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)


def orig(args, model):
    encoder = model.encoder
    embedder = model.encoder[0]

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

    return preds, references


def raw(args, model):
    df = pd.read_csv(
        f"data/LCQMC/{args.filename}", delimiter="\t",
        header=None, names=["text_1", "text_2", "label"]
    )

    tmp = model.encoder.encode(
        df["text_1"].tolist() + df["text_2"].tolist(), batch_size=32,
        show_progress_bar=True
    )
    embeddings1, embeddings2 = tmp[:df.shape[0]], tmp[df.shape[0]:]

    evaluator = EmbeddingSimilarityEvaluator(
        main_similarity=SimilarityFunction.COSINE
    )

    spearman_score = evaluator(
        embeddings1, embeddings2, labels=df["label"].values
    )
    print(f"Spearman: {spearman_score:.4f}")

    preds = 1 - paired_cosine_distances(embeddings1, embeddings2)
    return preds, df["label"].values


def main(args):
    model = SentencePairCosineSimilarity.load(args.model_path)
    model.eval()

    if args.mode == "orig":
        preds, references = orig(args, model)
    else:
        preds, references = raw(args, model)

    print(f"Pred {pd.Series(preds).describe()}")

    if args.threshold == -1:
        best_thres, best_acc = -1, -1
        for threshold in np.arange(0.05, 1, 0.05):
            binarized = (preds > threshold).astype("int")
            acc = (binarized == references).sum() / len(references)
            if acc > best_acc:
                best_acc = acc
                best_thres = threshold
        print(f"Best acc: {best_acc:.4f} @ {best_thres:.2f}")
    else:
        binarized = (preds > args.threshold).astype("int")
        acc = (binarized == references).sum() / len(references)
        print(f"Acc: {acc:.4f} @ {args.threshold:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--mode', type=str, default="orig")
    arg('--model-path', type=str, default="pretrained_models/bert_wwm_ext/")
    arg('--filename', type=str, default="valid.txt")
    arg('--threshold', type=float, default=-1)
    args = parser.parse_args()
    main(args)
