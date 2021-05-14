import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from PIL import Image
import pandas as pd
import pickle

def cvt_keypoint(kp):
    '''
    convert keypoint result to numpy array
    '''
    N = len(kp)
    locations = np.zeros((N,2))
    for i in range(N):
        locations[i, :] = np.array(kp[i].pt)
    return locations

sift = cv2.xfeatures2d.SIFT_create()

dataset_dir = 'Mushrooms'
save_path = 'figures/sift/'
train_csv_path = os.path.join(dataset_dir, 'train_test_split', 'train.csv')
test_csv_path = os.path.join(dataset_dir, 'train_test_split', 'test.csv')

train_data_frame = pd.read_csv(train_csv_path, header=None)

ft_loc, ft_id = [], []

for i in range(len(train_data_frame)):
	if random.random() < 0.01:
		print(i)
		im_path = train_data_frame.iloc[i,0]
		img = cv2.imread(im_path)
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

		kp = sift.detect(gray, None)

		img = cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		p = im_path.split('/')[-1]
		cv2.imwrite('%s%s'%(save_path, p),img)
