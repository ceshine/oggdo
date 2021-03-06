"""Extends the news similarity dataset

Create title -> title + summary pairs for search applications.
"""
from pathlib import Path

import typer
import pandas as pd


def extract_title(text):
    return text.split("。 ")[0]


def main(data_path: str = typer.Argument("data/annotated.csv")):
    df = pd.read_csv(data_path)
    print("Input size:", df.shape)
    title_1 = df.text_1.apply(extract_title)
    title_2 = df.text_2.apply(extract_title)
    df_1 = df.copy()
    df_1["text_1"] = title_1
    df_2 = df.copy()
    df_2["text_2"] = title_2
    df_final = pd.concat([df, df_1, df_2], axis=0, ignore_index=True)
    print("Output size:", df_final.shape)
    # Write results
    data_path_ = Path(data_path)
    output_path = data_path_.parent / f"{data_path_.stem}_ext.csv"
    df_final.to_csv(output_path, index=False)


if __name__ == "__main__":
    typer.run(main)
