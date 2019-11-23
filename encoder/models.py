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
