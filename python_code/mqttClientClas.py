import paho.mqtt.client as mqtt

#los clientes se subscriben a su particion y publican los resultados

#se llama al conectarse al broker
def on_connect(client, userdata, flags, rc):
    print("Connected classification client with result code " + str(rc))
    client.subscribe("partition/1") #nos subscribimos a este tema

#se llama al obtener un mensaje del broker
def on_message(client, userdata, msg):
    print(msg.topic + " " + str(msg.payload)) #imprimimos la respuesta
    client.publish("partition/results/1", "results_clas_client_1")#publicamos los resultados

client = mqtt.Client("clas_client_1")
client.on_connect = on_connect
client.on_message = on_message

#conectamos con el broker
client.connect("192.168.1.138", 1883)#debe ser el nombre o ip

client.loop_forever()