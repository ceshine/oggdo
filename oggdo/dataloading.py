from typing import Tuple, Dict, Optional, List

import numpy as np
import torch
from torch.utils.data.sampler import Sampler


class SortishSampler(Sampler):
    """Returns an iterator that traverses the the data in randomly ordered batches that are approximately the same size.
    The max key size batch is always returned in the first call because of pytorch cuda memory allocation sequencing.
    Without that max key returned first multiple buffers may be allocated when the first created isn't large enough
    to hold the next in the sequence.

    Taken from fastai library.
    """

    def __init__(self, data_source, key, bs, chunk_size=100):
        self.data_source, self.key, self.bs = data_source, key, bs
        self.chunk_size = chunk_size

    def __len__(self): return len(self.data_source)

    def __iter__(self):
        while True:
            idxs = np.random.permutation(len(self.data_source))
            sz = self.bs * self.chunk_size
            ck_idx = [idxs[i:i+sz] for i in range(0, len(idxs), sz)]
            sort_idx = np.concatenate(
                [sorted(s, key=self.key, reverse=True) for s in ck_idx])
            sz = self.bs
            ck_idx = [sort_idx[i:i+sz]for i in range(0, len(sort_idx), sz)]
            # find the chunk with the largest key,
            max_ck = np.argmax([self.key(ck[0]) for ck in ck_idx])
            # then make sure it goes first.
            if len(ck_idx[max_ck]) != self.bs:
                continue
            ck_idx[0], ck_idx[max_ck] = ck_idx[max_ck], ck_idx[0]
            sort_idx = np.concatenate(np.random.permutation([
                np.random.permutation(chunk.reshape(self.bs, -1)).reshape(-1)
                for chunk in ck_idx[1:-1]
            ]))
            sort_idx = np.concatenate((ck_idx[0], sort_idx, ck_idx[-1]))
            break
        return iter(sort_idx)


class SortSampler(Sampler):
    """
    Taken from fastai library.
    """

    def __init__(self, data_source, key):
        self.data_source, self.key = data_source, key

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        return iter(sorted(
            range(len(self.data_source)),
            key=self.key, reverse=True
        ))


def collate_singles(
    batch, pad, opening_id, closing_id, truncate_length
) -> Tuple[Dict, Optional[torch.Tensor]]:
    """Batch preparation.

    1. Pad the sequences.
    2. Or truncate the longer sequences.
    """
    if isinstance(batch[0], list):
        transposed = [
            batch,
            [None] * len(batch)
        ]
    else:
        transposed = list(zip(*batch))
    max_len = min(
        max((len(x) for x in transposed[0])) + 2,
        truncate_length
    )
    tokens = np.zeros((len(batch), max_len), dtype=np.int64) + pad
    tokens[:, 0] = opening_id
    for j, row in enumerate(transposed[0]):
        row = row[:max_len-2]
        tokens[j, 1:len(row)+1] = row
        tokens[j, len(row)+1] = closing_id

    assert (
        np.sum(tokens == closing_id) == len(batch)), \
        f"{np.sum(tokens == closing_id)}, {len(batch)}"
    token_tensor = torch.from_numpy(tokens)
    mask_tensor = (token_tensor != pad).float()
    # Labels
    if transposed[1][0] is None:
        labels = None
    else:
        labels = torch.tensor(transposed[1])
    return (
        {
            "input_ids": token_tensor,
            "input_mask": mask_tensor,
        },
        labels
    )


def collate_distill(
    batch, pad, opening_id, closing_id, truncate_length
) -> List[Dict[str, torch.Tensor]]:
    if isinstance(batch[0], list):
        transposed = [
            batch,
            [None] * len(batch)
        ]
    else:
        transposed = list(zip(*batch))
    max_len = min(
        max((len(x) for x in transposed[0])) + 2,
        truncate_length
    )
    results = []
    for i in range(2):
        tokens = np.zeros((len(batch), max_len), dtype=np.int64) + pad
        tokens[:, 0] = opening_id
        for j, row in enumerate(transposed[i]):
            row = row[:max_len-2]
            tokens[j, 1:len(row)+1] = row
            tokens[j, len(row)+1] = closing_id
        assert (
            np.sum(tokens == closing_id) == len(batch)), \
            f"{np.sum(tokens == closing_id)}, {len(batch)}"
        token_tensor = torch.from_numpy(tokens)
        mask_tensor = (token_tensor != pad).float()
        results.append(
            {
                "input_ids": token_tensor,
                "input_mask": mask_tensor,
            }
        )
    return results


def collate_pairs(
    batch, pad, opening_id, closing_id, truncate_length
) -> Tuple[Dict, Optional[torch.Tensor]]:
    """Batch preparation.

    1. Pad the sequences.
    2. Or truncate the longer sequences.
    """
    transposed = list(zip(*batch))
    token_tensors = [None, None]
    mask_tensors = [None, None]
    max_len = min(
        max(
            max((len(x) for x in transposed[0])),
            max((len(x) for x in transposed[1]))
        ) + 2,
        truncate_length
    )
    for i in range(2):
        tokens = np.zeros((len(batch), max_len), dtype=np.int64) + pad
        tokens[:, 0] = opening_id
        for j, row in enumerate(transposed[i]):
            row = row[:max_len-2]
            tokens[j, 1:len(row)+1] = row
            tokens[j, len(row)+1] = closing_id
        assert np.sum(tokens == closing_id) == len(batch)
        token_tensors[i] = torch.from_numpy(tokens)
        mask_tensors[i] = (token_tensors[i] != pad).float()
    # Labels
    if transposed[2][0] is None:
        labels = None
    else:
        labels = torch.tensor(transposed[2])
    return (
        {
            "input_ids": torch.cat(token_tensors, dim=0),
            "input_mask": torch.cat(mask_tensors, dim=0)
        },
        labels
    )
