import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import pandas as pd
import pickle
from sklearn import svm

kmeans = pickle.load(open('bovw.pkl', 'rb'))

sift = cv2.xfeatures2d.SIFT_create()

dataset_dir = 'Mushrooms'
train_csv_path = os.path.join(dataset_dir, 'train_test_split', 'train.csv')
test_csv_path = os.path.join(dataset_dir, 'train_test_split', 'test.csv')

train_data_frame = pd.read_csv(train_csv_path, header=None)
test_data_frame = pd.read_csv(test_csv_path, header=None)

# process training images
train_bovw_desc = []
train_gt_labels = []
num_vocab = 500

for i in range(len(train_data_frame)):
	print(i)
	im_path = train_data_frame.iloc[i,0]
	img = np.asarray(Image.open(im_path))
	_, desc = sift.detectAndCompute(img, None)
	labels = kmeans.predict(desc)
	hist, _ = np.histogram(labels, np.arange(num_vocab+1))

	train_bovw_desc.append(hist)
	train_gt_labels.append(train_data_frame.iloc[i,1])

train_bovw_desc = np.vstack(train_bovw_desc)
train_gt_labels = np.stack(train_gt_labels)

# process test images
test_bovw_desc = []
test_gt_labels = []

for i in range(len(test_data_frame)):
	print(i)
	im_path = test_data_frame.iloc[i,0]
	img = np.asarray(Image.open(im_path))
	_, desc = sift.detectAndCompute(img, None)
	labels = kmeans.predict(desc)
	hist, _ = np.histogram(labels, np.arange(num_vocab+1))

	test_bovw_desc.append(hist)
	test_gt_labels.append(test_data_frame.iloc[i,1])

test_bovw_desc = np.vstack(test_bovw_desc)
test_gt_labels = np.stack(test_gt_labels)

# save bovw descriptors
np.save('bovw_result/train_bovw_desc.npy', train_bovw_desc)
np.save('bovw_result/test_bovw_desc.npy', test_bovw_desc)
np.save('bovw_result/train_gt_labels.npy', train_gt_labels)
np.save('bovw_result/test_gt_labels.npy', test_gt_labels)

# svm classification
clf = svm.SVC(decision_function_shape='ovr')
clf.fit(train_bovw_desc, train_gt_labels)

pickle.dump(clf, open('bovw_result/bovw_clf.pkl', 'wb'))