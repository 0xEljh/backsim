FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye

RUN pip install --upgrade pip

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libxml2-dev \
    libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    setuptools \
    wheel \
    twine