import numpy as np
import cv2
import yaml
import itertools
import csv
import argparse

colormap = {'blue': [255, 0, 0], 'green': [0, 255, 0], 'red': [0, 0, 255],
            'yellow': [0, 255, 255], 'white': [255, 255, 255]}
parser = argparse.ArgumentParser(description='Calculate optical flow.')
parser.add_argument('image1', type=str, help='Input image No1.')
parser.add_argument('image2', type=str, help='Input image No2.')
parser.add_argument('--method', '-m', type=str, default='lk', choices=['lk', 'fb'], help='Select a method.')
parser.add_argument('--circle_color', '-cc', type=str, default='red', choices=colormap.keys(), help='Select a color for circle.')
parser.add_argument('--line_color', '-lc', type=str, default='red', choices=colormap.keys(), help='Select a color for line.')
parser.add_argument('--vector_scale', '-vs', type=float, default=1.0, help='Scale saving vector data.')
parser.add_argument('--size', '-s', type=int, default=5, help='Size of original point marker.')
parser.add_argument('--line', '-l', type=int, default=2, help='Width of vector line.')
args = parser.parse_args()
config = yaml.load(open('config.yaml'))

def lucas_kanade(file1, file2):
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
    data = []
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        dx = args.vector_scale * (a - c)
        dy = args.vector_scale * (b - d)
        cv2.line(mask, (c, d), (int(c + dx), int(d + dy)), colormap[args.line_color], args.line)
        cv2.line(img2, (c, d), (int(c + dx), int(d + dy)), colormap[args.line_color], args.line)
        cv2.circle(mask, (c, d), args.size, colormap[args.circle_color], -1)
        cv2.circle(img2, (c, d), args.size, colormap[args.circle_color], -1)
        data.append([c, d, dx, dy])

    cv2.imwrite('vectors.png', mask)
    cv2.imwrite('result.png', img2)
    with open('data.csv', 'w') as f:
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
            dx = args.vector_scale * dx
            dy = args.vector_scale * dy
            cv2.line(mask, (x, y), (x + int(dx), y + int(dy)), colormap[args.line_color], args.line)
            cv2.line(frame2, (x, y), (x + int(dx), y + int(dy)), colormap[args.line_color], args.line)
            cv2.circle(mask, (x, y), args.size, colormap[args.circle_color], -1)
            cv2.circle(frame2, (x, y), args.size, colormap[args.circle_color], -1)
            data.append([x, y, dx, dy])
    cv2.imwrite('vectors.png', mask)
    cv2.imwrite('result.png', frame2)
    with open('data.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(data)

if __name__ == "__main__":
    if args.method == 'lk':
        lucas_kanade(args.image1, args.image2)
    elif args.method == 'fb':
        farneback(args.image1, args.image2)
