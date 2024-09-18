#!/bin/bash
# node_topic model ip wheighting
gnome-terminal -- bash -c "python3 ./classifierNode.py 3.0 knn 10.20.35.226 pnw en; exec bash"
gnome-terminal -- bash -c "python3 ./classifierNode.py 3.1 knn 10.20.35.226 pnw en; exec bash"
gnome-terminal -- bash -c "python3 ./classifierNode.py 3.2 knn 10.20.35.226 pnw en; exec bash"
sleep 5
# partition_sizes data_size train_size load_data_func dataset_path decision_rule rep_conf sel_part_conf_file
# if rep_conf == -1 do not create csv
# if rep_conf == 0 create csv with train subsets and test data
# if rep_conf == 1 use csv created before
# in sel_part_conf_file put the classes labels, whitespace separated, one line per node. If needed, put a blank line and then specify discarded classes
gnome-terminal -- bash -c "python3 ./centralNode.py 3 10000 0.75 0.25 selected ../datasets/reordered_mnist_train.csv sum -1 conf_files/sel_part_conf_mnist.txt; exec bash"