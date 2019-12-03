# Pose estimation using trt_pose

## Install
Clone the trt_pose repo in this directory:
```
git clone https://github.com/NVIDIA-AI-IOT/trt_pose.git
```

Follow the instructions to install on your jetson nano.

Install the rest of the dependencies with pip:
```
pip install -r requirements.txt
```

If you are using the RealSense camera, make sure to follow [these instructions](https://www.jetsonhacks.com/2019/05/16/jetson-nano-realsense-depth-camera/) to install the librealsense library.

If you are using AWS to push your data up, make sure to configure your device with with your AWS credentials and edit run_pose.py with you S3 bucket name and DynamoDB table name.

## Run
Running with the Intel Realsense camera:
```
python run_pose.py
```

Running with a regular webcam:
```
python run_pose_webcam.py
```

