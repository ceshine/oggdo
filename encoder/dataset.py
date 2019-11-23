from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class LcqmcDataset(Dataset):
    def __init__(self, tokenizer, data_dir: str = "data/LCQMC/", filename: str = "train.txt"):
        df = pd.read_csv(
            Path(data_dir) / filename, delimiter="\t",
            header=None, names=["text_1", "text_2", "label"]
        )
        self.labels = df.label.values
        self.text_1 = np.asarray([
            tokenizer.encode(text) for text in df.text_1.values
        ])
        self.text_2 = np.asarray([
            tokenizer.encode(text) for text in df.text_2.values
        ])

    def __getitem__(self, item):
        return (
            self.text_1[item],
            self.text_2[item],
            self.labels[item]
        )

    def __len__(self):
        return len(self.labels)
