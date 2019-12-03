import json
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

# Define the codec and create VideoWriter object
name = 'test_pose_vid.mp4'
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(name, fourcc, 30.0, (1024,768))

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


print('Initializing Webcam..')
cap = cv2.VideoCapture(0)
fourcc_cap = cv2.VideoWriter_fourcc(*'MJPG')
cap.set(cv2.CAP_PROP_FOURCC, fourcc_cap)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

while True:
    ret, frame = cap.read()

    # Show images
    image, pose_dict = inference(frame)
    out.write(image)
    print(pose_dict)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop streaming
cap.release()
out.release()
