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

# save bovw descriptors
train_bovw_desc = np.load('bovw_result/train_bovw_desc.npy')
test_bovw_desc = np.load('bovw_result/test_bovw_desc.npy')
train_gt_labels = np.load('bovw_result/train_gt_labels.npy')
test_gt_labels = np.load('bovw_result/test_gt_labels.npy')

clf = pickle.load(open('bovw_result/bovw_clf.pkl', 'rb'))

num_class = 9


# construct confusion matrix on training result
confusion = np.zeros((num_class, num_class)).astype('float')
num_img = train_gt_labels.shape[0]
for i in range(num_img):
	class_label = train_gt_labels[i]
	pred = clf.predict(train_bovw_desc[i].reshape(1,-1))
	confusion[class_label, pred] += 1

accuracy = np.sum(np.diag(confusion)) / np.sum(confusion)
print('training accuracy is %f'%(accuracy))
confusion = confusion / np.sum(confusion, axis=1, keepdims=True)
# plot confusion matrix
train_fig, ax1 = plt.subplots(1,1)
ax1.imshow(confusion)
train_fig.savefig('bovw_result/confusion_train.png')


# construct confusion matrix on training result
confusion = np.zeros((num_class, num_class)).astype('float')
num_img = test_gt_labels.shape[0]
for i in range(num_img):
	class_label = test_gt_labels[i]
	pred = clf.predict(test_bovw_desc[i].reshape(1,-1))
	confusion[class_label, pred] += 1

accuracy = np.sum(np.diag(confusion)) / np.sum(confusion)
print('testing accuracy is %f'%(accuracy))
confusion = confusion / np.sum(confusion, axis=1, keepdims=True)
# plot confusion matrix
test_fig, ax2 = plt.subplots(1,1)
ax2.imshow(confusion)
test_fig.savefig('bovw_result/confusion_test.png')
