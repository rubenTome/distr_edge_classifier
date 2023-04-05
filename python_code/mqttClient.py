import paho.mqtt.client as mqtt

#tendremos un cliente para crear y publicar las particiones
#los clientes restantes se subscriben a su particion y publican los resultados

#se llama al conectarse al broker
def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe("partition/1") #nos subscribimos a este tema

#se llama al obtener un mensaje del broker
def on_message(client, userdata, msg):
    print(msg.topic + " " + str(msg.payload)) #imprimimos la respuesta
    client.publish("partition/1/results", "resultados1")#publicamos los resultados

client = mqtt.Client("client1")
client.on_connect = on_connect
client.on_message = on_message

#conectamos con el broker
client.connect("192.168.1.138", 1883)#debe ser el nombre o ip

client.loop_forever()