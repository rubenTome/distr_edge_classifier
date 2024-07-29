#!/bin/bash
# node_topic model ip wheighting
gnome-terminal -- bash -c "python3 ./classifierNode.py 3.0 knn 10.20.36.78 pnw; exec bash"
gnome-terminal -- bash -c "python3 ./classifierNode.py 3.1 knn 10.20.36.78 pnw; exec bash"
gnome-terminal -- bash -c "python3 ./classifierNode.py 3.2 knn 10.20.36.78 pnw; exec bash"
sleep 5
# partition_sizes data_size train_size load_data_func dataset_path decision_rule rep_conf
#if rep_conf == -1 do not create csv
#if rep_conf == 0 create csv with train subsets and test data
#if rep_conf == 1 use csv created before
gnome-terminal -- bash -c "python3 ./centralNode.py 3 3500 0.75 0.25 balanced ../datasets/covtype.csv sum 0; exec bash"