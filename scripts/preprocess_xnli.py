import re
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/XNLI-1.0/")


def process_train():
    df = pd.read_csv(
        DATA_DIR / "multinli.train.zh.tsv",
        sep="\t", error_bad_lines=False
    )
    pattern = re.compile(r"(?<![A-Za-z\s])\s+(?![A-Za-z])")
    df["premise"] = df["premise"].str.replace(pattern, "")
    df["hypo"] = df["hypo"].str.replace(pattern, "")
    df.to_csv(DATA_DIR / "train.csv", index=False)


def process_test(filename="xnli.dev.jsonl", target="valid.csv", language="zh"):
    results = []
    with open(DATA_DIR / filename) as fin:
        for line in fin.readlines():
            row = json.loads(line)
            if row["language"] != language:
                continue
            results.append((
                row["sentence1"], row["sentence2"], row["gold_label"]
            ))
    df = pd.DataFrame(results, columns=["premise", "hypo", "label"])
    df.to_csv(DATA_DIR / target, index=False)


if __name__ == "__main__":
    process_train()
    process_test("xnli.dev.jsonl", "valid.csv")
    process_test("xnli.test.jsonl", "test.csv")
