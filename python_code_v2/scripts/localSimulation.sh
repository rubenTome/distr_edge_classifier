#!/bin/bash
# node_topic model ip wheighting
gnome-terminal -- bash -c "python3 ./classifierNode.py 3.0 knn 10.20.35.226 piw; exec bash"
gnome-terminal -- bash -c "python3 ./classifierNode.py 3.1 knn 10.20.35.226 piw; exec bash"
gnome-terminal -- bash -c "python3 ./classifierNode.py 3.2 knn 10.20.35.226 piw; exec bash"
sleep 5
# partition_sizes data_size train_size load_data_func dataset_path
gnome-terminal -- bash -c "python3 ./centralNode.py 3 3500 0.75 0.25 random ../datasets/covtype.csv; exec bash"