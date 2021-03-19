import csv
import gzip
from typing import Set

from oggdo.encoder import SentenceEncoder
from oggdo.dataset import download_dataset, SBertDataset
from oggdo.components import TransformerWrapper, PoolingLayer


def get_splits(data_folder: str = "data/", dataset: SBertDataset = SBertDataset.AllNLI):
    data_path = download_dataset(data_folder, dataset)
    train: Set[str] = set()
    valid: Set[str] = set()
    test: Set[str] = set()
    with gzip.open(data_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if row['split'] == 'train':
                target = train
            elif row['split'] == 'dev':
                target = valid
            else:
                target = test
            target.add(row['sentence1'])
            target.add(row['sentence2'])
    return (list(x) for x in (train, valid, test))


def load_encoder(
        model_path, model_type, max_length, do_lower_case,
        mean_pooling=True, cls=False, max_pooling=False,
        expand_to_dimension: int = -1):
    embedder = TransformerWrapper(
        model_path,
        max_seq_length=max_length,
        do_lower_case=do_lower_case,
        model_type=model_type
    )
    pooler = PoolingLayer(
        embedder.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=mean_pooling,
        pooling_mode_cls_token=cls,
        pooling_mode_max_tokens=max_pooling,
        layer_to_use=-1,
        expand_to_dimension=expand_to_dimension
    )
    encoder = SentenceEncoder(modules=[
        embedder, pooler
    ])
    return encoder
