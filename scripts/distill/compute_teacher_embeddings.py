import os
import random

from pathlib import Path
from functools import partial

import typer
import torch
import joblib
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from oggdo.dataset import SBertDataset, SentenceDataset
from oggdo.dataloading import SortSampler, collate_singles
from oggdo.utils import features_to_device

from common import get_splits, load_encoder

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(
    dataset: SBertDataset, model_path: str, output_folder: str = "cache/teacher_embs/",
    batch_size: int = 32, workers: int = 2, attentions: bool = False, sample: float = -1
):
    # this is designed for stsb-roberta-base; some changes might be needed for other models
    encoder = load_encoder(model_path, None, 256, do_lower_case=True, mean_pooling=True).cuda().eval()
    encoder[0].attentions = attentions
    train, valid, test = get_splits("data/", dataset)
    Path(output_folder).mkdir(exist_ok=True, parents=True)
    for name, sentences in (("train", train), ("valid", valid), ("test", test)):
        if name == "train" and sample > 0 and sample < 1:
            np.random.seed(42)
            sentences = np.random.choice(
                sentences, size=round(len(sentences) * sample),
                replace=False
            )
        ds = SentenceDataset(encoder[0].tokenizer, sentences)
        sampler = SortSampler(
            ds,
            key=lambda x: len(ds.text[x])
        )
        loader = DataLoader(
            ds,
            sampler=sampler,
            collate_fn=partial(
                collate_singles,
                pad=0,
                opening_id=encoder[0].cls_token_id,
                closing_id=encoder[0].sep_token_id,
                truncate_length=encoder[0].max_seq_length
            ),
            batch_size=batch_size,
            num_workers=workers
        )
        print(name)
        with torch.no_grad():
            buffer = []
            for batch, _ in tqdm(loader, ncols=100):
                outputs = encoder(
                    features_to_device(batch, torch.device("cuda"))
                )
                buffer.append((
                    x.detach().cpu().numpy() if x is not None else None
                    for x in (outputs["sentence_embeddings"], outputs["attentions"])
                ))
        embs = np.concatenate(buffer)
        embs = embs[np.argsort(list(iter(sampler)))]
        joblib.dump([sentences, embs], Path(output_folder) / (dataset.value + "_" + name + ".jbl"))


if __name__ == "__main__":
    typer.run(main)
