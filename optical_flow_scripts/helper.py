import os
from natsort import natsorted, ns
import numpy as np
import cv2
import yaml
import itertools
import csv
import argparse

colormap = {'blue': [255, 0, 0], 'green': [0, 255, 0], 'red': [0, 0, 255],
            'yellow': [0, 255, 255], 'white': [255, 255, 255]}
config = yaml.load(open('config.yaml'))
vector_scale = 60.0
size = 2
line_color = 'red'
line = 2
circle_color = 'yellow'

def lucas_kanade(file1, file2, path):
    conf = config['LucasKanade']
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners = 100,
                          qualityLevel = conf['quality_level'],
                          minDistance = 7,
                          blockSize = 7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize = (conf['window_size'],
                                conf['window_size']),
                     maxLevel = 2,
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    img1 = cv2.imread(file1)
    img2 = cv2.imread(file2)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(img1_gray, mask = None, **feature_params)
    mask = np.zeros_like(img1)
    p1, st, err = cv2.calcOpticalFlowPyrLK(img1_gray, img2_gray, p0, None, **lk_params)

    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    data = [['p1_x', 'p1_y', 'p2_x', 'p2_y', 'dx', 'dy']]
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        dx = vector_scale * (a - c)
        dy = vector_scale * (b - d)
        cv2.line(mask, (c, d), (int(c + dx), int(d + dy)), colormap[line_color], line)
        cv2.line(img2, (c, d), (int(c + dx), int(d + dy)), colormap[line_color], line)
        cv2.circle(mask, (c, d), size, colormap[circle_color], -1)
        cv2.circle(img2, (c, d), size, colormap[circle_color], -1)
        data.append([c, d, c+dx, d+dy, dx, dy])

    cv2.imwrite(os.path.join(path, 'vectors.png'), mask)
    cv2.imwrite(os.path.join(path, 'result.png'), img2)
    with open(os.path.join(path, 'data.csv'), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(data)

def farneback(file1, file2):
    conf = config['Farneback']
    frame1 = cv2.imread(file1)
    prv = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(frame1)
    frame2 = cv2.imread(file2)
    nxt = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prv, nxt, 0.5, 3,
                                        conf['window_size'],
                                        3, 5, 1.1, 0)
    height, width = prv.shape

    data = []
    for x, y in itertools.product(range(0, width, conf['stride']),
                                  range(0, height, conf['stride'])):
        if np.linalg.norm(flow[y, x]) >= conf['min_vec']:
            dy, dx = flow[y, x].astype(int)
            dx = vector_scale * dx
            dy = vector_scale * dy
            cv2.line(mask, (x, y), (x + int(dx), y + int(dy)), colormap[line_color], line)
            cv2.line(frame2, (x, y), (x + int(dx), y + int(dy)), colormap[line_color], line)
            cv2.circle(mask, (x, y), size, colormap[circle_color], -1)
            cv2.circle(frame2, (x, y), size, colormap[circle_color], -1)
            data.append([x, y, dx, dy])
    cv2.imwrite('vectors.png', mask)
    cv2.imwrite('result.png', frame2)
    with open('data.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(data)


directory_list = list()
for root, dirs, files in os.walk(r'C:\Users\shalea2\PycharmProjects\PredNet\PredNet\PredNet\optical_flow', topdown=False):
    for name in dirs:
        directory_list.append(os.path.join(root, name))

directory_list = natsorted(directory_list)

for i, d in enumerate(directory_list):
    # print(d)
    if i % 5 == 0:
        continue
    l = natsorted(os.listdir(d))
    file1, file2 = os.path.join(d, l[-4]), os.path.join(d, l[-2])
    # print(l)
    # print(file1, file2)
    lucas_kanade(file1, file2, d)

# lucas_kanade(r'C:\Users\shalea2\PycharmProjects\PredNet\PredNet\PredNet\optical_flow\penguin1.png',
#              r'C:\Users\shalea2\PycharmProjects\PredNet\PredNet\PredNet\optical_flow\penguin2.png',
#              r'C:\Users\shalea2\PycharmProjects\PredNet\PredNet\PredNet\optical_flow')


