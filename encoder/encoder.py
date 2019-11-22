import json
import logging
import os
from collections import OrderedDict, defaultdict
from typing import List, Iterable

import numpy as np
import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm, trange

from .utils import import_from_string
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

        all_embeddings = []
        length_sorted_idx = np.argsort([len(sen) for sen in sentences])

        iterator = range(0, len(sentences), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Batches")

        for batch_idx in iterator:
            batch_tokens = []
            batch_start = batch_idx
            longest_seq = 0

            for idx in length_sorted_idx[batch_start:batch_start+batch_size]:
                sentence = sentences[idx]
                tokens = self.tokenize(sentence)
                longest_seq = max(longest_seq, len(tokens))
                batch_tokens.append(tokens)

            features = defaultdict(list)
            for text in batch_tokens:
                sentence_features = self.get_sentence_features(
                    text, longest_seq)
                for feature_name in sentence_features:
                    features[feature_name].append(
                        sentence_features[feature_name])

            for feature_name in features:
                features[feature_name] = torch.tensor(
                    np.asarray(features[feature_name])
                ).to(self.device)

            with torch.no_grad():
                embeddings = self.forward(features)
                embeddings = embeddings['sentence_embedding'].to('cpu').numpy()
                all_embeddings.extend(embeddings)

        reverting_order = np.argsort(length_sorted_idx)
        all_embeddings = np.asarray(
            [all_embeddings[idx] for idx in reverting_order]
        )

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
            contained_modules.append({'idx': idx, 'name': name, 'path': os.path.basename(
                model_path), 'type': type(module).__module__})

        with open(os.path.join(path, 'modules.json'), 'w') as fOut:
            json.dump(contained_modules, fOut, indent=2)

        with open(os.path.join(path, 'config.json'), 'w') as fOut:
            json.dump({'__version__': __version__}, fOut, indent=2)
