#!/usr/bin/env python3
import time
import json
import boto3
import torch
import PIL.Image
import trt_pose.coco
import trt_pose.models
from torch2trt import TRTModule
import torchvision.transforms as transforms

from trt_pose.parse_objects import ParseObjects
from draw_objects import DrawObjects

import cv2
import numpy as np
import pyrealsense2 as rs

# Initialize AWS resources

bucket_name = 'YOU-BUCKET-HERE'
table_name = 'YOU-TABLE-HERE'
s3 = boto3.resource('s3')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(table_name)

WIDTH = 224
HEIGHT = 224

ASSET_DIR = './trt_pose/tasks/human_pose/'
OPTIMIZED_MODEL = ASSET_DIR + 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

with open(ASSET_DIR + 'human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)
num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

print('Loading Model')
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
print('Model loaded!')

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.resize(image, (HEIGHT, WIDTH))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

def inference(image):
    data = preprocess(image)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf) #, cmap_threshold=0.15, link_threshold=0.15)
    body_dict = draw_objects(image, counts, objects, peaks)
    return image, body_dict


print('Initializing RealSense cam..')
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
print('Pipeline started')


while True:
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue

    print('captured frame and depth')
    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    # Show images
    image, pose_dict = inference(color_image)
    
    timestamp = int(time.time() * 10000)
    flname = '{}.jpg'.format(str(timestamp))
    img_filename = 'poses/' + flname
    depth_filename= 'depth/' + flname

    upload_img = cv2.imencode('.jpg', image)[1].tostring()
    upload_dd = cv2.imencode('.jpg', depth_colormap.copy())[1].tostring()

    upload_obj = s3.Object(bucket_name, img_filename)
    upload_depth = s3.Object(bucket_name, depth_filename)

    upload_obj.put(Body=upload_img)
    upload_depth.put(Body=upload_dd)

    table.put_item(Item={
        'filename': img_filename,
        'poses': str(pose_dict)
        }
    )
    print(pose_dict)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop streaming
pipeline.stop()
