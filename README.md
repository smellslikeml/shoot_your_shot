# Shoot your shot!

![Shoot Your Shot](shoot_it.gif)

A weekend hackathon project to build a smart dart application. This demo uses two cameras to view the thrower and view the dartboard and track poses and dart placement.

## Getting started
You can run the pose demo in the pose directory by running
```
python run_pose.py
```

You can also run the dart demo in the dart directory by running
```
python vogelpik.py
```

To run the entire demo, run:
```
python client.py 
```
on the jetson nano and 
```
python pushbutton.py
```
on the raspberry pi to trigger each session.

## Requirements
* [trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose)
* [pytorch](https://pytorch.org/)
* [opencv](https://opencv.org/)
* [pillow](https://pillow.readthedocs.io/en/stable/)
* [mosquitto](https://mosquitto.org/)
* [paho-mqtt](https://pypi.org/project/paho-mqtt/)

### Hardware
* [Jetson Nano](https://developer.nvidia.com/embedded/jetson-nano-developer-kit)
* [Intel RealSense D415 (optional)](https://store.intelrealsense.com/buy-intel-realsense-depth-camera-d415.html)
* [USB Webcam](https://www.amazon.com/gp/product/B07R4YGFT6/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1)
* [Raspberry Pi](https://www.raspberrypi.org/)
* [large pushbutton](https://www.amazon.com/Raspberry-Squid-Button-Twin-Pack/dp/B0170B75EU)

## Install
Clone the trt_pose repo:
```
git clone https://github.com/NVIDIA-AI-IOT/trt_pose.git
```

Follow the instructions to install on your jetson nano.

Install the rest of the dependencies with pip:
```
pip install -r requirements.txt
```

If you are using the RealSense camera, make sure to follow [these instructions](https://www.jetsonhacks.com/2019/05/16/jetson-nano-realsense-depth-camera/) to install the librealsense library.

If you are using AWS to push your data to the cloud, make sure to configure your device with with your AWS credentials and edit each main script in the pose/ and darts/ directory with you S3 bucket name and DynamoDB table name.


