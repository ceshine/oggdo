import os
import json
import logging
from typing import List, Dict

import torch
from torch import nn
import numpy as np
from transformers import BertModel, BertTokenizer, BertConfig


class BertWrapper(nn.Module):
    """BERT model to generate token embeddings.

    Each token is mapped to an output vector from BERT.
    """

    def __init__(self, model_name_or_path: str, max_seq_length: int = 128, do_lower_case: bool = True):
        super().__init__()
        self.config_keys = ['max_seq_length', 'do_lower_case']
        self.do_lower_case = do_lower_case

        if max_seq_length > 510:
            logging.warning(
                "BERT only allows a max_seq_length of 510 (512 with special tokens). Value will be set to 510")
            max_seq_length = 510
        self.max_seq_length = max_seq_length

        config = BertConfig.from_pretrained(model_name_or_path)
        config.output_hidden_states = True
        self.bert = BertModel.from_pretrained(
            model_name_or_path, config=config)
        self.tokenizer = BertTokenizer.from_pretrained(
            model_name_or_path, do_lower_case=do_lower_case)
        self.cls_token_id = self.tokenizer.convert_tokens_to_ids(
            [self.tokenizer.cls_token])[0]
        self.sep_token_id = self.tokenizer.convert_tokens_to_ids(
            [self.tokenizer.sep_token])[0]

    def forward(self, features) -> Dict:
        """Returns token_embeddings, cls_token"""
        _, _, hidden_states = self.bert(
            input_ids=features['input_ids'],
            token_type_ids=features['token_type_ids'],
            attention_mask=features['input_mask']
        )
        features.update({
            'hidden_states': hidden_states,
            'input_mask': features['input_mask']
        })
        return features

    def get_word_embedding_dimension(self) -> int:
        return self.bert.config.hidden_size

    def tokenize(self, text: str) -> List[int]:
        """
        Tokenizes a text and maps tokens to token-ids
        """
        # return self.tokenizer.convert_tokens_to_ids(
        #     self.tokenizer.tokenize(text)
        # )
        return self.tokenizer.encode(text)

    def get_sentence_features(self, tokens: List[str], pad_seq_length: int) -> Dict:
        """
        Convert tokenized sentence in its embedding ids, segment ids and mask

        :param tokens:
            a tokenized sentence
        :param pad_seq_length:
            the maximal length of the sequence. Cannot be greater than self.sentence_transformer_config.max_seq_length
        :return: embedding ids, segment ids and mask for the sentence
        """
        pad_seq_length = min(pad_seq_length, self.max_seq_length) + 2

        # Truncate to the left
        tokens = tokens[:pad_seq_length - 2]
        sentence_length = len(tokens) + 2

        # Zero-pad up to the sequence length. BERT: Pad to the right
        input_ids = np.zeros(pad_seq_length, dtype=np.int64)
        input_ids[:sentence_length] = np.array(
            [self.cls_token_id] + tokens +
            [self.sep_token_id], dtype=np.int64)
        token_type_ids = np.zeros(pad_seq_length, dtype=np.int64)
        input_mask = np.zeros(pad_seq_length, dtype=np.int64)
        input_mask[:sentence_length] = 1

        assert len(input_ids) == pad_seq_length
        assert len(input_mask) == pad_seq_length
        assert len(token_type_ids) == pad_seq_length

        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'input_mask': input_mask
            # 'sentence_lengths': sentence_length
        }

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.bert.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, 'sentence_bert_config.json')) as fIn:
            config = json.load(fIn)
        return BertWrapper(model_name_or_path=input_path, **config)


class PoolingLayer(nn.Module):
    """Performs pooling (max or mean) on the token embeddings.

    Using pooling, it generates from a variable sized sentence a fixed sized sentence embedding. This layer also allows to use the CLS token if it is returned by the underlying word embedding model.
    You can concatenate multiple poolings together.
    """

    def __init__(self,
                 word_embedding_dimension: int,
                 pooling_mode_cls_token: bool = False,
                 pooling_mode_max_tokens: bool = False,
                 pooling_mode_mean_tokens: bool = True,
                 pooling_mode_mean_sqrt_len_tokens: bool = False,
                 layer_to_use: int = -1
                 ):
        super().__init__()

        self.config_keys = ['word_embedding_dimension',  'pooling_mode_cls_token',
                            'pooling_mode_mean_tokens', 'pooling_mode_max_tokens', 'pooling_mode_mean_sqrt_len_tokens']

        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode_cls_token = pooling_mode_cls_token
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
        self.pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_tokens
        self.layer_to_use = layer_to_use

        pooling_mode_multiplier = sum([
            pooling_mode_cls_token, pooling_mode_max_tokens,
            pooling_mode_mean_tokens, pooling_mode_mean_sqrt_len_tokens
        ])
        self.pooling_output_dimension = (
            pooling_mode_multiplier * word_embedding_dimension)

    def forward(self, features: Dict[str, torch.Tensor]):
        token_embeddings = features['hidden_states'][self.layer_to_use]
        cls_tokens = token_embeddings[:, 0, :]  # CLS token is first token
        input_mask = features['input_mask']

        # Pooling strategy
        output_vectors = []
        if self.pooling_mode_cls_token:
            output_vectors.append(cls_tokens)
        if self.pooling_mode_max_tokens:
            input_mask_expanded = input_mask.unsqueeze(
                -1).expand(token_embeddings.size()).float()
            # Set padding tokens to large negative value
            token_embeddings[input_mask_expanded == 0] = -1e9
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            input_mask_expanded = input_mask.unsqueeze(
                -1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(
                token_embeddings * input_mask_expanded, 1)
            # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if 'token_weights_sum' in features:
                sum_mask = features['token_weights_sum'].unsqueeze(
                    -1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)
            # TODO: rather than clip this, ensure that sum_mask is never zero
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

        output_vector = torch.cat(output_vectors, 1)
        features.update({'sentence_embedding': output_vector})
        return features

    def get_sentence_embedding_dimension(self):
        return self.pooling_output_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)
        return PoolingLayer(**config)
