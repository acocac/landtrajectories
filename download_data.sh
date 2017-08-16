#!/bin/sh

mkdir -p data
cd data
wget -O  dataset.zip https://www.dropbox.com/s/107zw1xjwcte0u3/model.zip?dl=0
unzip dataset.zip
rm dataset.zip
