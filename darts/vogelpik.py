#!/usr/bin/env python3
from __future__ import print_function
import cv2
import math
import argparse
import numpy as np
import options as opt
from time import time
from collections import deque

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

if opt.annotate:
    opt.model = True

if opt.model:
    from joblib import load
    from skimage.feature import hog
    clf = load('dart.dat')

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

def cropFeature(im, dist):
    im = cv2.resize(im, (200, 200))
    fd, _ = hog(im, orientations=4, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)
    return np.concatenate((fd, np.expand_dims(dist, axis=0)))


class DartBoard(object):
    def __init__(self):
        self.circle_q = deque(maxlen=opt.Q_LEN)

    def update(self, center, radius):
        self.center = center
        self.radius = radius
        if not math.isnan(radius):
            self.circle_q.append(radius)

    def get_smooth_radius(self):
        mean_r = np.mean(self.circle_q)
        if not math.isnan(mean_r):
            return int(mean_r)
        else:
            return int(np.min((width, height)) / 2)

        

db = DartBoard()
center = (width / 2, height / 2)
while True:
    ret, frame = cap.read()
    ts = int(time() * 1000)

    if frame is None:
        break
    
    ###################
    # Detect Motion
    ###################

    fgMask = backSub.apply(frame)
    # filter shadows by thresholding mask
    ret, fgMask = cv2.threshold(fgMask, 150, 255, cv2.THRESH_BINARY)
    # smooth mask, close small blobs
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel, iterations=1)


    if np.mean(fgMask / 255) < 0.1:   # ignore large/rapid changes, filter camera obstruction

        #########################
        # Detect Blobs & Circles
        #########################

        keypoints = detector.detect(fgMask)
        
        blur = cv2.cvtColor(cv2.blur(frame, k_tup), cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, height / 8, 
                                   param1=50, param2=25, 
                                   minRadius=s_config.l_cir, maxRadius=s_config.h_cir)
         
        #################
        # Filter/Annotate
        #################

        # Convert to color images
        im_with_keypoints = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2BGR)

        # Iterate through Keypoints (but not too many)
        if len(keypoints) < opt.MAX_BLOBS:    
            for kp in keypoints:
                tup = tuple(map(int, kp.pt))
                disp = np.array(tup) - np.array(center)
                theta = np.round(np.arctan(disp[1] / (disp[0] + 1e-3)), 2)

                try:
                    xx, yy = center
                    x, y = tup
                    
                    dist = np.linalg.norm(disp)
                    rad = np.round(dist / db.get_smooth_radius(), 2)

                    # Constrain to Keypoints Near Dartboard
                    if rad  < 1.1 * db.get_smooth_radius():
                        print('dart position: ', tup)
                        dart_crop = frame[y-h:y+h,x-w:x+w,:]
                        if opt.teach:
                            score = input('Please Enter the Score using keys: 0-6: ')
                            print('Dart Score: {}'.format(int(score) * 5 if int(score) < 6 else 50))
                            crop_path = "crops/{}/{}_{}_{}_{}.png".format(score, ts, theta, rad)
                        else:
                            crop_path = "crops/{}_{}_{}.png".format(ts, theta, rad)
                        if opt.model:
                            try:
                                ft_vec = cropFeature(dart_crop, dist)
                                pred = int(clf.predict(np.expand_dims(ft_vec, axis=0))[0])
                                points = 5 * pred if pred < 6 else 50
                                print('Estimated Score: {}'.format(points))
                                if opt.annotate:
                                    f = frame.copy()
                                    if points:
                                        cv2.putText(f, '+{} pts'.format(points), tup, cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 200), 2)
                                        cv2.imwrite("bg/{}_ann.png".format(ts), f)
                            except:
                                pass
                            
                except NameError:
                    pass

                im_with_keypoints = cv2.circle(im_with_keypoints, tup, 40, opt.dart_color, 2)

        if keypoints:
            try:
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    center = [(i[0], i[1]) for i in circles[0,:]]
                    center = tuple(np.mean(center, axis=0).astype(int))
                    db.update(center, circles[0][0][2])
                    im_with_keypoints = cv2.circle(im_with_keypoints, center, db.get_smooth_radius(), opt.board_color, 5)
                    im_with_keypoints = cv2.drawMarker(im_with_keypoints, center, opt.bullseye_color, 0, 30, 4)


                # Saving Images
                if opt.save:
                    cv2.imwrite("bg/{}_blk.png".format(ts), im_with_keypoints)
                    cv2.imwrite("bg/{}_raw.png".format(ts), frame)
                    cv2.imwrite(crop_path, dart_crop)

            except NameError:
                pass
        try:
            im_with_keypoints = cv2.circle(im_with_keypoints, center, db.get_smooth_radius(), opt.board_color, 5)
            im_with_keypoints = cv2.drawMarker(im_with_keypoints, center, opt.bullseye_color, 0, 30, 4)
        except:
            pass

        try:
            # Display Detection/Annotation
            cv2.imshow("Keypoints", im_with_keypoints)
        except:
            pass

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
