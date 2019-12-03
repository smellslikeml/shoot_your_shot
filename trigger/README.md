# Pushbutton Trigger for Camera Synchronization

Follow these instructions to set up a simple pushbutton trigger to synchronize cameras for dart throwing sessions.

## Install
Make sure you have mosquitto installed on your server device. We used a raspberry pi 3 with the pushbutton attached as our server.
```
sudo apt-get install mosquitto
```

Mosquitto will run as a service in the background.

On both your server and client machines, install paho-mqtt
```
pip install paho-mqtt
```

## Run
On your client machine (jetson nano), run:
```
python client.py
```

And on your server (raspberry pi pushbutton device), run:
```
python pushbutton.py
```

The client.py script will start/stop both the dart and pose scripts in sync when the button is pushed.
