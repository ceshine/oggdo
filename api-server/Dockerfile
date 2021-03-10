FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

RUN pip install --no-cache-dir https://github.com/ceshine/oggdo/archive/0.0.4.zip
RUN pip install --no-cache-dir typer opencc-python-reimplemented pandas
ENV MAX_WORKERS 1
ENV WEB_CONCURRENCY 1
COPY main.py /app