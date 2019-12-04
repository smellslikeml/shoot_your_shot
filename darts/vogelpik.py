#!/usr/bin/env python3
from __future__ import print_function
import cv2
import argparse
import numpy as np
import options as opt
from time import time

parser = argparse.ArgumentParser(description='This script uses motion detection to track darts.')
parser.add_argument('--source', type=str, help='source uri', default='0')
parser.add_argument('--quality', type=str, help='raw/high/med/low', default='med')
args = parser.parse_args()

k_tup = (opt.k_size, opt.k_size)
kernel = np.ones(k_tup, np.uint8)

if args.source.isdigit():
    source = int(args.source) 

cap = cv2.VideoCapture(source)

if args.quality != 'raw':
    # MJPEG compression
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)

# Set up parameters based on camera specs
# To check, run: v4l2-ctl --list-formats-ext --device /dev/video0
if args.quality == 'raw' or args.quality == 'high': # For YUV (6 fps) / MJPEG (30 fps)
    width = 1920
    height = 1080
elif args.quality == 'med':  # 60 fps
    width = 1280
    height = 720
elif args.quality == 'low':  # 120 fps
    width = 640
    height = 480

s_config = opt.SourceConfig(width, height)
w = h = s_config.offset

cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

try:
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('Camera supports {} FPS at {} x {}'.format(fps, width, height))
except:
    pass

if not cap.isOpened:
    print('Unable to open: ' + args.input)
    exit(0)

# Initialize detectors
detector = cv2.SimpleBlobDetector_create(opt.params)
backSub = cv2.createBackgroundSubtractorMOG2(history=opt.history)

while True:
    ret, frame = cap.read()
    ts = int(time() * 1000)

    if frame is None:
        break
    
    fgMask = backSub.apply(frame)
    
    ###################
    # Detect Motion
    ###################

    # filter shadows by thresholding mask
    ret, fgMask = cv2.threshold(fgMask, 150, 255, cv2.THRESH_BINARY)
    # smooth mask, close small blobs
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel, iterations=1)
    # smooth mask, spread white area
    #fgMask = cv2.blur(fgMask, k_tup)
    # sharpen mask by thresholding
    #ret, fgMask = cv2.threshold(fgMask, 100, 255, cv2.THRESH_BINARY)

    if np.mean(fgMask / 255) < 0.1:   # ignore large/rapid changes, filter camera obstruction

        #########################
        # Detect Blobs & Circles
        #########################

        keypoints = detector.detect(fgMask)
        circles = cv2.HoughCircles(fgMask, cv2.HOUGH_GRADIENT, 1, 70, 
                                   param1=50, param2=25, 
                                   minRadius=s_config.l_cir, maxRadius=s_config.h_cir)
         
        #################
        # Filter/Annotate
        #################

        # Convert to color images
        im_with_keypoints = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2BGR)

        # Iterate through Keypoints
        for kp in keypoints:
            tup = tuple(map(int, kp.pt))

            try:
                xx, yy = center
                x, y = tup

                # Constrain to Keypoints Near Dartboard
                dist = np.linalg.norm(np.array(center) - np.array(tup))
                if s_config.c_radius is not None and dist  < 1.1 * s_config.c_radius:
                    print('dart position: ', tup)
                    dart_crop = frame[y-h:y+h,x-w:x+w,:]
            except NameError:
                pass

            im_with_keypoints = cv2.circle(im_with_keypoints, tup, 40, opt.dart_color, 2)

        if keypoints:
            try:
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    center = [(i[0], i[1]) for i in circles[0,:]]
                    center = tuple(np.mean(center, axis=0).astype(int))
                    s_config.c_radius = circles[0][0][2]
                    im_with_keypoints = cv2.circle(im_with_keypoints, center, s_config.c_radius, opt.board_color, 5)
                    im_with_keypoints = cv2.drawMarker(im_with_keypoints, center, opt.bullseye_color, 0, 30, 4)


                # Saving Images
                if opt.save:
                    cv2.imwrite("bg/{}_blk.png".format(ts), im_with_keypoints)
                    cv2.imwrite("bg/{}_raw.png".format(ts), frame)
                    cv2.imwrite("crops/{}_{}_{}_{}_{}.png".format(ts, x, y, xx, yy), dart_crop)

            except NameError:
                pass

        try:
            # Display Detection/Annotation
            cv2.imshow("Keypoints", im_with_keypoints)
        except:
            pass

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
