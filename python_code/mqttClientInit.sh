#!/bin/bash
gnome-terminal -- bash -c "python3 /home/ruben/FIC/Q8/TFG/clean_partition/python_code/mqttClientClas.py 4.0 rf 192.168.1.141; exec bash"
gnome-terminal -- bash -c "python3 /home/ruben/FIC/Q8/TFG/clean_partition/python_code/mqttClientClas.py 4.1 rf 192.168.1.141; exec bash"
gnome-terminal -- bash -c "python3 /home/ruben/FIC/Q8/TFG/clean_partition/python_code/mqttClientClas.py 4.2 rf 192.168.1.141; exec bash"
gnome-terminal -- bash -c "python3 /home/ruben/FIC/Q8/TFG/clean_partition/python_code/mqttClientClas.py 4.3 rf 192.168.1.141; exec bash"
gnome-terminal -- bash -c "python3 /home/ruben/FIC/Q8/TFG/clean_partition/python_code/mqttClientClas.py 1.0 rf 192.168.1.141; exec bash"
sleep 2
gnome-terminal -- bash -c "python3 /home/ruben/FIC/Q8/TFG/clean_partition/python_code/mqttClientPart.py [4,1] 2000 1000 piw /home/ruben/FIC/Q8/TFG/clean_partition/scenariosimul/scenariosimulC8D3G3STDEV0.05.csv; exec bash"