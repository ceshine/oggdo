"""
This examples trains BERT for the STSbenchmark from scratch. It generates sentence embeddings
that can be compared using cosine-similarity to measure the similarity.
"""
import logging
import argparse
# from typing import Sequence

import tqdm
import pandas as pd
from opencc import OpenCC
from sklearn.metrics.pairwise import paired_cosine_distances

from oggdo.encoder import SentenceEncoder
from oggdo.components import BertWrapper, PoolingLayer
from oggdo.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction


class LoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


# Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
# /print debug information to stdout


T2S = OpenCC('t2s')


def convert_t2s(text: str) -> str:
    return T2S.convert(text)


def main(args):
    # Read the dataset
    df = pd.read_csv(args.file)
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
    model = SentenceEncoder(modules=[
        embedder, pooler
    ])
    model.eval()

    evaluator = EmbeddingSimilarityEvaluator(
        main_similarity=SimilarityFunction.COSINE
    )

    if args.t2s:
        df["text_1"] = df["text_1"].apply(convert_t2s)
        df["text_2"] = df["text_2"].apply(convert_t2s)

    tmp = model.encode(
        df["text_1"].tolist() + df["text_2"].tolist(), batch_size=16,
        show_progress_bar=True
    )
    embeddings1, embeddings2 = tmp[:df.shape[0]], tmp[df.shape[0]:]

    spearman_score = evaluator(
        embeddings1, embeddings2, labels=df["similarity"].values
    )
    print(spearman_score)

    preds = 1 - paired_cosine_distances(embeddings1, embeddings2)
    df["pred"] = preds
    df.to_csv("cache/annotated_zero_shot_pred.csv", index=False)
    print(f"Pred {pd.Series(preds).describe()}")
    return preds, df["similarity"].values


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model-path', type=str, default="pretrained_models/bert_wwm_ext/")
    arg('--file', type=str, default="data/annotated.csv")
    arg('--t2s', action="store_true")
    arg('--layer', type=int, default=-2)
    args = parser.parse_args()
    main(args)
