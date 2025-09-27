#!/bin/bash

cd ../data/openqa

# Downloading data from source and unzip them.

# wiki passages
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
gunzip psgs_w100.tsv.gz

# train and dev set (json, for train)
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz
gunzip biencoder-nq-train.json.gz

wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz
gunzip biencoder-nq-dev.json.gz

wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-train.json.gz
gunzip biencoder-trivia-train.json.gz

wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-dev.json.gz
gunzip biencoder-trivia-dev.json.gz

wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-train.json.gz
gunzip biencoder-squad1-train.json.gz

wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-dev.json.gz
gunzip biencoder-squad1-dev.json.gz

# test, train, and dev set (csv, for eval)
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/squad1-test.qa.csv
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-test.qa.csv
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-test.qa.csv.gz
gunzip trivia-test.qa.csv.gz

wget https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-dev.qa.csv
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-train.qa.csv
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-dev.qa.csv.gz
gunzip trivia-dev.qa.csv.gz
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-train.qa.csv.gz
gunzip trivia-train.qa.csv.gz
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/squad1-dev.qa.csv

# for squad, extract training queries (in csv form) from json training set
cd ../../QCPTgc
python -m peach.datasets.openqa.json2csv


# download msmarco from https://microsoft.github.io/msmarco/ and https://rocketqa.bj.bcebos.com/corpus/marco.tar.gz

# download beir from https://github.com/beir-cellar/beir

