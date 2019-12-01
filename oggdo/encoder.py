import json
import logging
import os
from collections import OrderedDict, defaultdict
from typing import List, Iterable
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from .utils import import_from_string, features_to_device
from .dataset import SentenceDataset
from .dataloading import collate_singles, SortSampler
from . import __version__


class SentenceEncoder(nn.Sequential):
    def __init__(self, model_path: str = None, modules: Iterable[nn.Module] = None, device: str = "cuda"):
        if modules is not None and not isinstance(modules, OrderedDict):
            modules = OrderedDict([(str(idx), module)
                                   for idx, module in enumerate(modules)])

        if model_path is not None and model_path != "":
            logging.info(
                "Load pretrained SentenceTransformer: %s",
                model_path
            )
            # Load from disk
            with open(os.path.join(model_path, 'modules.json')) as fIn:
                contained_modules = json.load(fIn)

            modules = OrderedDict()
            for module_config in contained_modules:
                module_class = import_from_string(module_config['type'])
                module = module_class.load(os.path.join(
                    model_path, module_config['path']))
                modules[module_config['name']] = module

        super().__init__(modules)
        self.device = torch.device(device)
        self.to(device)

    def encode(self, sentences: List[str], batch_size: int = 8, show_progress_bar: bool = None) -> List[np.ndarray]:
        """
        Computes sentence embeddings

        :param sentences:
            the sentences to embed
        :param batch_size:
            the batch size used for the computation
        :param show_progress_bar:
                Output a progress bar when encode sentences
        :return:
            a list with ndarrays of the embeddings for each sentence
        """
        if show_progress_bar is None:
            show_progress_bar = (
                logging.getLogger().getEffectiveLevel() == logging.INFO or
                logging.getLogger().getEffectiveLevel() == logging.DEBUG
            )

        ds = SentenceDataset(self[0].tokenizer, sentences)
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
                opening_id=self[0].cls_token_id,
                closing_id=self[0].sep_token_id,
                truncate_length=self[0].max_seq_length
            ),
            batch_size=batch_size,
            num_workers=1
        )

        all_embeddings = []

        iterator = loader
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Batches")

        for features, _ in iterator:
            with torch.no_grad():
                embeddings = self.forward(
                    features_to_device(features, self.device))
                embeddings = embeddings['sentence_embeddings'].to(
                    'cpu').numpy()
                all_embeddings.extend(embeddings)

        reverting_order = np.argsort(list(iter(sampler)))
        all_embeddings = np.asarray(all_embeddings)[reverting_order]

        return all_embeddings

    def tokenize(self, text):
        return self._first_module().tokenize(text)

    def get_sentence_features(self, *features):
        return self._first_module().get_sentence_features(*features)

    def get_sentence_embedding_dimension(self):
        return self._last_module().get_sentence_embedding_dimension()

    def _first_module(self):
        """Returns the first module of this sequential embedder"""
        return self._modules[next(iter(self._modules))]

    def _last_module(self):
        """Returns the last module of this sequential embedder"""
        return self._modules[next(reversed(self._modules))]

    def save(self, path):
        """
        Saves all elements for this seq. sentence embedder into different sub-folders
        """
        logging.info("Save model to {}".format(path))
        contained_modules = []

        for idx, name in enumerate(self._modules):
            module = self._modules[name]
            model_path = os.path.join(path, str(idx)+"_"+type(module).__name__)
            os.makedirs(model_path, exist_ok=True)
            module.save(model_path)
            contained_modules.append({
                'idx': idx,
                'name': name,
                'path': os.path.basename(model_path),
                'type': type(module).__module__ + "." + type(module).__name__
            })

        with open(os.path.join(path, 'modules.json'), 'w') as fOut:
            json.dump(contained_modules, fOut, indent=2)

        with open(os.path.join(path, 'config.json'), 'w') as fOut:
            json.dump({'__version__': __version__}, fOut, indent=2)
