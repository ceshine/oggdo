import logging
import argparse

import torch
import pandas as pd
from opencc import OpenCC
from sklearn.metrics.pairwise import paired_cosine_distances

from oggdo.encoder import SentenceEncoder
from oggdo.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction

try:
    from apex import amp
    APEX = True
except:
    APEX = False

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)

T2S = OpenCC('t2s')


def convert_t2s(text: str) -> str:
    return T2S.convert(text)


def raw(args, encoder):
    df = pd.read_csv(args.file)

    if args.t2s:
        df["text_1"] = df["text_1"].apply(convert_t2s)
        df["text_2"] = df["text_2"].apply(convert_t2s)

    tmp = encoder.encode(
        df["text_1"].tolist() + df["text_2"].tolist(), batch_size=32,
        show_progress_bar=True
    )
    embeddings1, embeddings2 = tmp[:df.shape[0]], tmp[df.shape[0]:]

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
    encoder = SentenceEncoder(model_path=args.model_path)
    encoder.eval()
    if APEX and args.amp and (not args.torchscript):
        encoder = amp.initialize(encoder, opt_level=args.amp)
    if args.torchscript:
        if args.amp:
            encoder[0].bert = encoder[0].bert.half()
        traced_model = torch.jit.trace(
            encoder[0].bert,
            (torch.zeros(8, 256).long().cuda(),
             torch.zeros(8, 256).long().cuda(),
             torch.ones(8, 256).long().cuda())
        )
        encoder[0].bert = traced_model
        assert isinstance(encoder[0].bert, torch.jit.TopLevelTracedModule)
    encoder.max_seq_length = 256
    print(encoder[1].get_config_dict())
    encoder[1].pooling_mode_cls_token = False
    encoder[1].pooling_mode_mean_tokens = True
    print(encoder[1].get_config_dict())

    preds, _ = raw(args, encoder)

    print(f"Pred {pd.Series(preds).describe()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--file', type=str, default="data/annotated.csv")
    arg('--t2s', action="store_true")
    arg('--amp', type=str, default="")
    arg('--torchscript', action="store_true")
    arg('--model-path', type=str, default="pretrained_models/bert_wwm_ext/")
    args = parser.parse_args()
    main(args)
