import os
from typing import List, Optional

import typer
import uvicorn
from opencc import OpenCC
from fastapi import FastAPI
from pydantic import BaseModel, Field, constr

from oggdo.encoder import SentenceEncoder


os.environ["TOKENIZERS_PARALLELISM"] = "false"

PORT = int(os.environ.get("PORT", "8666"))
APP = FastAPI()
T2S = OpenCC('t2s')
MODEL: Optional[SentenceEncoder] = None
if os.environ.get("MODEL", None):
    MODEL = SentenceEncoder(os.environ["MODEL"], device="cpu").eval()

app = APP


class TextInput(BaseModel):
    text: str = Field(
        None, title="The piece of text you want to create embeddings for.", max_length=384
    )
    t2s: bool = False


class BatchTextInput(BaseModel):
    text_batch: List[str] = Field([], title="Pieces of text you want to create embeddings for.")
    t2s: bool = False


class EmbeddingsResult(BaseModel):
    vector: List[float]


class BatchEmbeddingsResult(BaseModel):
    vectors: List[List[float]]


@APP.post("/", response_model=EmbeddingsResult)
def get_embeddings(text_input: TextInput):
    assert MODEL is not None, "MODEL is not loaded."
    text = text_input.text.replace("\n", " ")
    if text_input.t2s:
        text = T2S.convert(text)
    vector = MODEL.encode(
        [text],
        batch_size=1,
        show_progress_bar=False
    )[0]
    return EmbeddingsResult(vector=vector.tolist())


@APP.post("/batch", response_model=BatchEmbeddingsResult)
def get_batch_embeddings(text_input: BatchTextInput):
    assert MODEL is not None, "MODEL is not loaded."
    batch = [x.replace("\n", " ") for x in text_input.text_batch]
    if text_input.t2s:
        batch = [T2S.convert(x) for x in batch]
    vectors = MODEL.encode(
        batch,
        batch_size=int(os.environ.get("BATCH_SIZE", 8)),
        show_progress_bar=False
    )
    return BatchEmbeddingsResult(vectors=vectors.tolist())


def main(model_path: str = typer.Argument("streamlit-model/")):
    global MODEL
    MODEL = SentenceEncoder(model_path, device="cpu").eval()
    print(f"Listening to port {PORT}")
    uvicorn.run(APP, host='0.0.0.0', port=PORT)


if __name__ == '__main__':
    typer.run(main)
