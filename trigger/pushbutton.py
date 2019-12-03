import time
import RPi.GPIO as GPIO
import paho.mqtt.publish as publish

hostname = "YOUR.IP.ADDRESS.HERE"

GPIO.setmode(GPIO.BCM)

GPIO.setup(23, GPIO.IN, pull_up_down=GPIO.PUD_UP) #Button to GPIO23

# Camera is off
camera = False
print('camera state: ', camera)

try:
    while True:
         button_state = GPIO.input(23)
         # Check if button has been pressed
         if button_state == False:
             if camera:
                 publish.single("camera/off", "OFF", hostname=hostname)
                 print('camera off')
                 camera = False
                 time.sleep(0.2)
             else:
                 publish.single("camera/on", "ON", hostname=hostname)
                 print('camera on')
                 camera = True
                 time.sleep(0.2)
         else:
             pass
except:
    GPIO.cleanup()
