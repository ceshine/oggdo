FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

LABEL maintainer="ceshine@ceshine.net"

ARG CONDA_PYTHON_VERSION=3
ARG PYTHON_VERSION=3.6
ARG USERNAME=docker
ARG USERID=1000
ARG CONDA_DIR=/opt/conda

ENV LANG C.UTF-8
ENV PYTHONIOENCODING UTF-8

# Instal basic utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget unzip bzip2 sudo build-essential ca-certificates libjpeg-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV PATH $CONDA_DIR/bin:$PATH
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda$CONDA_PYTHON_VERSION-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    echo 'export PATH=$CONDA_DIR/bin:$PATH' > /etc/profile.d/conda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm -rf /tmp/*

# Create the user
RUN useradd --create-home -s /bin/bash --no-user-group -u $USERID $USERNAME && \
    chown $USERNAME $CONDA_DIR -R && \
    adduser $USERNAME sudo && \
    echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

USER $USERNAME
WORKDIR /home/$USERNAME

RUN conda install python=$PYTHON_VERSION && \
    conda install faiss-cpu -c pytorch && \
    conda clean -tipsy

ARG PIP_MIRROR=https://pypi.python.org/simple

COPY requirements.txt /home/$USERNAME/requirements.txt
RUN pip install -i $PIP_MIRROR -U pip && rm -rf ~/.cache/pip
RUN pip install -i $PIP_MIRROR -r requirements.txt && rm -rf ~/.cache/pip
RUN pip install tensorflow-gpu==1.13.1 tensorflow-hub sentencepiece tf-sentencepiece ipython dataclasses
