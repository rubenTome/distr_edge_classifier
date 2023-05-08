#!/bin/bash
gnome-terminal -- bash -c "python3 /home/ruben/FIC/Q8/TFG/clean_partition/python_code/mqttClientClas.py 4.0 rf; exec bash"
gnome-terminal -- bash -c "python3 /home/ruben/FIC/Q8/TFG/clean_partition/python_code/mqttClientClas.py 4.1 rf; exec bash"
gnome-terminal -- bash -c "python3 /home/ruben/FIC/Q8/TFG/clean_partition/python_code/mqttClientClas.py 4.2 rf; exec bash"
gnome-terminal -- bash -c "python3 /home/ruben/FIC/Q8/TFG/clean_partition/python_code/mqttClientClas.py 4.3 rf; exec bash"
gnome-terminal -- bash -c "python3 /home/ruben/FIC/Q8/TFG/clean_partition/python_code/mqttClientClas.py 1.0 rf; exec bash"
sleep 2
gnome-terminal -- bash -c "python3 /home/ruben/FIC/Q8/TFG/clean_partition/python_code/mqttClientPart.py [4,1] /home/ruben/FIC/Q8/TFG/clean_partition/scenariosimul/scenariosimulC8D3G3STDEV0.05.csv; exec bash"