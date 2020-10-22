#!/bin/bash

cp data_explore* data/
python3 mergeData.py
python3 examineData.py
cp data_filtered regress_pyrep/
python3 regress_pyrep/learn_ik.py
cp weights.npy ../pyrepgym/envs/weights.npy
