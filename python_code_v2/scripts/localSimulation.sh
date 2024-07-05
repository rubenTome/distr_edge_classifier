#!/bin/bash
# node_topic model ip wheighting
gnome-terminal -- bash -c "python3 ./classifierNode.py 3.0 svm 10.20.35.226 piw; exec bash"
gnome-terminal -- bash -c "python3 ./classifierNode.py 3.1 svm 10.20.35.226 piw; exec bash"
gnome-terminal -- bash -c "python3 ./classifierNode.py 3.2 svm 10.20.35.226 piw; exec bash"
sleep 10
# partition_sizes data_size train_size load_data_func dataset_path
gnome-terminal -- bash -c "python3 ./centralNode.py 3 1000 0.7 0.3 random ../datasets/HIGGS.csv; exec bash"