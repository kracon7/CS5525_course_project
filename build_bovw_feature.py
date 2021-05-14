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


sift = cv2.xfeatures2d.SIFT_create()

dataset_dir = 'Mushrooms'
train_csv_path = os.path.join(dataset_dir, 'train_test_split', 'train.csv')
test_csv_path = os.path.join(dataset_dir, 'train_test_split', 'test.csv')

train_data_frame = pd.read_csv(train_csv_path, header=None)

train_feature = []

for i in range(len(train_data_frame)):
	print(i)
	im_path = train_data_frame.iloc[i,0]
	img = np.asarray(Image.open(im_path))#.transpose(2,0,1)
	_, desc = sift.detectAndCompute(img, None)

	# select samples from features
	ratio = 0.05
	choice = np.random.choice(desc.shape[0], int(ratio*desc.shape[0]), replace=False)
	train_feature.append(desc[choice])

# build BoW
train_feature = np.vstack(train_feature)
print('Extracted %d features in total'%(train_feature.shape[0]))

num_vocab = 500
kmeans = KMeans(n_clusters=num_vocab, max_iter=1000, n_init=1, 
			random_state=0, init='random', verbose=1).fit(train_feature)

pickle.dump(kmeans, open('bovw_result/bovw.pkl', 'wb'))