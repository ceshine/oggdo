import json
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import SentenceEncoder


class SentencePairCosineSimilarity(nn.Module):
    def __init__(self, sentence_encoder: SentenceEncoder, linear_transform: bool = False):
        super().__init__()
        self.linear_transform = linear_transform
        if linear_transform:
            self.scaler = nn.Parameter(
                torch.tensor([1.], dtype=torch.float))
            self.shift = nn.Parameter(
                torch.tensor([0.], dtype=torch.float))
        self.encoder = sentence_encoder
        self.to(self.encoder.device)

    def forward(self, features):
        embeddings = self.encoder(features)["sentence_embeddings"]
        assert len(embeddings) % 2 == 0
        embeddings_1 = embeddings[:len(embeddings) // 2]
        embeddings_2 = embeddings[len(embeddings) // 2:]
        similarities = F.cosine_similarity(embeddings_1, embeddings_2)
        if self.linear_transform:
            similarities = self.scaler * similarities + self.shift
        return similarities

    def save(self, output_path: Union[Path, str]):
        Path(output_path).mkdir(parents=True, exist_ok=True)
        if self.linear_transform:
            torch.save(
                {
                    "scaler": self.scaler.data,
                    "shift": self.shift.data
                },
                Path(output_path) / "linear_transform.pth"
            )
        self.encoder.save(str(output_path))

    @classmethod
    def load(cls, output_path: Union[Path, str]):
        linear_transform = False
        if (Path(output_path) / "linear_transform.pth").exists():
            linear_transform_params = torch.load(
                Path(output_path) / "linear_transform.pth"
            )
            linear_transform = True
        encoder = SentenceEncoder(str(output_path))
        model = cls(encoder, linear_transform=linear_transform)
        if linear_transform:
            model.scaler.data = linear_transform_params["scaler"]
            model.shift.data = linear_transform_params["shift"]
        return model


class SentencePairNliClassification(nn.Module):
    def __init__(
            self, sentence_encoder: SentenceEncoder,
            n_classes: int = 3,
            concatenation_sent_rep: bool = True,
            concatenation_sent_difference: bool = True,
            concatenation_sent_multiplication: bool = False):
        super().__init__()
        self.encoder = sentence_encoder
        self.n_classes = n_classes
        self.concatenation_sent_rep = concatenation_sent_rep
        self.concatenation_sent_difference = concatenation_sent_difference
        self.concatenation_sent_multiplication = concatenation_sent_multiplication

        self.config_keys = [
            'sentence_embeddings_dim',  'concatenation_sent_rep',
            'concatenation_sent_difference', 'concatenation_sent_multiplication',
            'n_classes']

        self.sentence_embeddings_dim = (
            self.encoder[1].get_sentence_embedding_dimension()
        )

        num_vectors_concatenated = 0
        if concatenation_sent_rep:
            num_vectors_concatenated += 2
        if concatenation_sent_difference:
            num_vectors_concatenated += 1
        if concatenation_sent_multiplication:
            num_vectors_concatenated += 1

        self.classifier = nn .Linear(
            num_vectors_concatenated * self.sentence_embeddings_dim,
            n_classes
        )
        self.to(self.encoder.device)

    def forward(self, features):
        embeddings = self.encoder(features)["sentence_embeddings"]
        assert len(embeddings) % 2 == 0
        embeddings_1 = embeddings[:len(embeddings) // 2]
        embeddings_2 = embeddings[len(embeddings) // 2:]

        vectors_concat = []
        if self.concatenation_sent_rep:
            vectors_concat.append(embeddings_1)
            vectors_concat.append(embeddings_2)
        if self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(embeddings_1 - embeddings_2))
        if self.concatenation_sent_multiplication:
            vectors_concat.append(embeddings_1 * embeddings_2)
        features = torch.cat(vectors_concat, 1)
        output = self.classifier(features)
        return output

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: Union[str, Path]):
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / 'classifier_config.json', 'w') as fout:
            json.dump(self.get_config_dict(), fout, indent=2)
        torch.save(
            self.classifier.state_dict(),
            output_path / "classifier.pth"
        )
        self.encoder.save(str(output_path))

    @classmethod
    def load(cls, model_path: Union[Path, str]):
        model_path = Path(model_path)
        with open(model_path / 'classifier_config.json') as fin:
            config = json.load(fin)
        encoder = SentenceEncoder(str(model_path))
        model = cls(encoder, **config)
        model.classifier.load_state_dict(
            torch.load(
                Path(model_path) / "classifier.pth"
            )
        )
        return model
