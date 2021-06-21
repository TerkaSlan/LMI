#!/bin/bash

DATA_DIR=Mtree-Cophir-100k
mkdir -p $DATA_DIR
cd $DATA_DIR

wget https://www.fi.muni.cz/~xslanin/lmi/knn_gt.json
wget https://www.fi.muni.cz/~xslanin/lmi/level-1.txt
wget https://www.fi.muni.cz/~xslanin/lmi/level-2.txt
wget https://www.fi.muni.cz/~xslanin/lmi/objects.txt
wget https://www.fi.muni.cz/~xslanin/lmi/info.txt