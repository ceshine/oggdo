import logging
import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm.autonotebook import tqdm
from sklearn.metrics.pairwise import paired_cosine_distances
import tensorflow as tf
import tensorflow_hub as hub
import tf_sentencepiece  # Not used directly but needed to import TF ops.

from oggdo.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction


logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)


def raw(args):
    df = pd.read_csv(
        f"data/LCQMC/{args.filename}", delimiter="\t",
        header=None, names=["text_1", "text_2", "label"]
    )

    g = tf.Graph()
    with g.as_default():
        text_input = tf.placeholder(dtype=tf.string, shape=[None])
        embed = hub.Module(
            "https://tfhub.dev/google/universal-sentence-encoder-multilingual/1")
        embedded_text = embed(text_input)
        init_op = tf.group(
            [tf.global_variables_initializer(), tf.tables_initializer()])
        g.finalize()

    # Initialize session.
    session = tf.Session(graph=g)
    session.run(init_op)

    # Compute embeddings.
    embs_1, embs_2 = [], []
    for i in range(0, len(df), args.batch_size):
        embs_1.append(session.run(
            embedded_text,
            feed_dict={text_input: df.text_1.values[i:i+args.batch_size]}
        ))
        embs_2.append(session.run(
            embedded_text,
            feed_dict={text_input: df.text_2.values[i:i+args.batch_size]}
        ))
    embeddings1 = np.concatenate(embs_1)
    embeddings2 = np.concatenate(embs_2)

    evaluator = EmbeddingSimilarityEvaluator(
        main_similarity=SimilarityFunction.COSINE
    )

    spearman_score = evaluator(
        embeddings1, embeddings2, labels=df["label"].values
    )
    print(f"Spearman: {spearman_score:.4f}")

    preds = 1 - paired_cosine_distances(embeddings1, embeddings2)

    df["pred"] = preds
    df.to_csv(f"cache/{Path(args.filename).stem}_pred.csv", index=False)

    return preds, df["label"].values


def main(args):
    preds, references = raw(args)

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
    arg('--batch-size', type=int, default=16)
    arg('--filename', type=str, default="dev.txt")
    arg('--threshold', type=float, default=-1)
    args = parser.parse_args()
    main(args)
