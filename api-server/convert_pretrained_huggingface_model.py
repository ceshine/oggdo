from pathlib import Path
from typing import Optional

import typer
from oggdo.encoder import SentenceEncoder
from oggdo.components import TransformerWrapper, PoolingLayer


def main(
    model_name, mean_pooling: bool = True, max_pooling: bool = False, cls_token: bool = False,
    max_seq_length: int = 256, model_type: Optional[str] = None,
    output_folder: Optional[str] = None
):
    if output_folder is None:
        output_folder = f"models/{model_name.split('/')[-1]}"
        print(f"Using the default output folder: {output_folder}")
    output_path = Path(output_folder)
    output_path.mkdir(parents=True)
    embedder = TransformerWrapper(
        model_name,
        max_seq_length=max_seq_length,
        model_type=model_type
    )
    pooler = PoolingLayer(
        embedder.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=mean_pooling,
        pooling_mode_cls_token=cls_token,
        pooling_mode_max_tokens=max_pooling,
        layer_to_use=-1
    )
    encoder = SentenceEncoder(modules=[
        embedder, pooler
    ])
    encoder.save(output_path.resolve())


if __name__ == '__main__':
    typer.run(main)
