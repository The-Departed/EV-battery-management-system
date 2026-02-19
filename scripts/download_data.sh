#!/bin/bash
# Download NASA Battery Dataset from Kaggle

echo "Downloading NASA Battery Dataset..."
curl -L -o data/raw/nasa-battery-dataset.zip \
  https://www.kaggle.com/api/v1/datasets/download/patrickfleith/nasa-battery-dataset

echo "Extracting dataset..."
cd data/raw
unzip -q nasa-battery-dataset.zip
rm nasa-battery-dataset.zip
cd ../..

echo "Dataset downloaded and extracted to data/raw/"
