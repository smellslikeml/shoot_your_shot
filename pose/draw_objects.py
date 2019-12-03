import cv2

class DrawObjects(object):

    def __init__(self, topology):
        self.topology = topology
        self.body_labels = {0:'nose', 1: 'rEye', 2: 'lEye', 3:'rEar', 4:'lEar', 5:'rShoulder', 6:'lShoulder', 
               7:'rElbow', 8:'lElbow', 9:'rWrist', 10:'lWrist', 11:'rHip', 12:'lHip', 13:'rKnee', 14:'lKnee',
              15:'rAnkle', 16:'lAnkle', 17:'chest'}

    def __call__(self, image, object_counts, objects, normalized_peaks):
        topology = self.topology
        height = image.shape[0]
        width = image.shape[1]

        K = topology.shape[0]
        count = int(object_counts[0])
        K = topology.shape[0]
        body_dict = {}
        for i in range(count):
            #color = (255, 0, 255)
            color = (112,107,222)
            obj = objects[0][i]
            C = obj.shape[0]
            for j in range(C):
                k = int(obj[j])
                if k >= 0:
                    peak = normalized_peaks[0][j][k]
                    x = round(float(peak[1]) * width)
                    y = round(float(peak[0]) * height)
                    cv2.circle(image, (x, y), 3, color, 2)
                    body_dict[self.body_labels[j]] = (x,y)
            for k in range(K):
                c_a = topology[k][2]
                c_b = topology[k][3]
                if obj[c_a] >= 0 and obj[c_b] >= 0:
                    peak0 = normalized_peaks[0][c_a][obj[c_a]]
                    peak1 = normalized_peaks[0][c_b][obj[c_b]]
                    x0 = round(float(peak0[1]) * width)
                    y0 = round(float(peak0[0]) * height)
                    x1 = round(float(peak1[1]) * width)
                    y1 = round(float(peak1[0]) * height)
                    cv2.line(image, (x0, y0), (x1, y1), color, 2)
        return body_dict
