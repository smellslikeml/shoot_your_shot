import subprocess
import os, signal
import paho.mqtt.client as mqtt

def check_kill_process(pstring):
    for line in os.popen("ps ax | grep " + pstring + " | grep -v grep"):
        fields = line.split()
        pid = fields[0]
        os.kill(int(pid), signal.SIGKILL)

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("camera/on")
    client.subscribe("camera/off")

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    if msg.topic == "camera/on":
        print('camera on')
        subprocess.Popen(['nohup', 'python', '../pose/run_pose.py', '&'])
        subprocess.Popen(['nohup', 'python', '../darts/vogelpik.py', '&'])
    elif msg.topic == "camera/off":
        print('camera off')
        check_kill_process('run_pose')
        check_kill_process('vogelpik')
    else:
        print('other')
    #print(msg.topic+" "+str(msg.payload))

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect("YOUR.SERVER.IP.HERE", 1883, 60)

# Blocking call that processes network traffic, dispatches callbacks and
# handles reconnecting.
# Other loop*() functions are available that give a threaded interface and a
# manual interface.
client.loop_forever()
