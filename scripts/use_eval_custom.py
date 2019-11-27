import argparse

import numpy as np
import pandas as pd
from opencc import OpenCC
from sklearn.metrics.pairwise import paired_cosine_distances
import tensorflow as tf
import tensorflow_hub as hub
import tf_sentencepiece  # Not used directly but needed to import TF ops.

from oggdo.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction


T2S = OpenCC('t2s')


def convert_t2s(text: str) -> str:
    return T2S.convert(text)


def raw(args):
    df = pd.read_csv(args.file)

    if args.t2s:
        df["text_1"] = df["text_1"].apply(convert_t2s)
        df["text_2"] = df["text_2"].apply(convert_t2s)

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
        embeddings1, embeddings2, labels=df["similarity"].values
    )
    print(f"Spearman: {spearman_score:.4f}")

    preds = 1 - paired_cosine_distances(embeddings1, embeddings2)
    df["pred"] = preds
    df.to_csv("cache/annotated_pred.csv", index=False)
    return preds, df["similarity"].values


def main(args):
    preds, _ = raw(args)

    print(f"Pred {pd.Series(preds).describe()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--file', type=str, default="data/annotated.csv")
    arg('--batch-size', type=int, default=16)
    arg('--t2s', action="store_true")
    args = parser.parse_args()
    main(args)
