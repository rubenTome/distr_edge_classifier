#!/bin/bash
gnome-terminal -- bash -c "python3 /home/ruben/FIC/Q8/TFG/clean_partition/python_code/mqttClientClas.py 3.0 rf 192.168.1.141; exec bash"
gnome-terminal -- bash -c "python3 /home/ruben/FIC/Q8/TFG/clean_partition/python_code/mqttClientClas.py 3.1 rf 192.168.1.141; exec bash"
gnome-terminal -- bash -c "python3 /home/ruben/FIC/Q8/TFG/clean_partition/python_code/mqttClientClas.py 3.2 rf 192.168.1.141; exec bash"
sleep 5
gnome-terminal -- bash -c "python3 /home/ruben/FIC/Q8/TFG/clean_partition/python_code/mqttClientPart.py [3] 3000 500 balanced random /home/ruben/FIC/Q8/TFG/clean_partition/scenariosimul/HIGGS.csv; exec bash"