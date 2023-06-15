#!/bin/bash
gnome-terminal -- bash -c "python3 /home/ruben/FIC/Q8/TFG/clean_partition/python_code/mqttClientClas.py 1.0 knn 192.168.1.141; exec bash"
#gnome-terminal -- bash -c "python3 /home/ruben/FIC/Q8/TFG/clean_partition/python_code/mqttClientClas.py 3.1 rf 192.168.1.141; exec bash"
#gnome-terminal -- bash -c "python3 /home/ruben/FIC/Q8/TFG/clean_partition/python_code/mqttClientClas.py 3.2 rf 192.168.1.141; exec bash"
sleep 5
gnome-terminal -- bash -c "python3 /home/ruben/FIC/Q8/TFG/clean_partition/python_code/mqttClientPart.py [1] 151 101 piw /home/ruben/FIC/Q8/TFG/clean_partition/scenariosimul/iris.csv; exec bash"