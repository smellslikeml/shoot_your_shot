import cv2
import numpy as np

save = True
model = True
k_size = 7
history = 50
Q_LEN = 100
MAX_BLOBS = 7

board_color = (132, 90, 205)
bullseye_color = (0, 0, 255)
dart_color = (255, 255, 0)
tracker_color = (200, 162, 200)
white_blob = (100, 200, 100)

params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 30
params.filterByCircularity = True
params.minCircularity = 0.1

class SourceConfig(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.min_len = np.min((width, height))
        self.h_cir = int(0.5 *  self.min_len // 2)
        self.l_cir = int(0.3 * self.min_len // 2)
        self.offset = self.min_len // 10
        self.c_radius = None

