import os
import csv
import random

ROOT = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(ROOT, 'Mushrooms/')
sub_dirs = os.listdir(data_dir)

train_csv_path = os.path.join(ROOT, 'train_test_split/train.csv')
test_csv_path = os.path.join(ROOT, 'train_test_split/test.csv')
train_csv = open(train_csv_path, 'w')
test_csv = open(test_csv_path, 'w')

mushroom_names = sub_dirs

train_csv_writer = csv.writer(train_csv, delimiter=',')
test_csv_writer = csv.writer(test_csv, delimiter=',')

for i, d in enumerate(sub_dirs):
    im_names = os.listdir(os.path.join(data_dir, d))
    random.shuffle(im_names)
    
    N = len(im_names)
    ratio = 0.8

    train_flist = im_names[:int(0.8*N)]
    test_flist = im_names[int(0.8*N):]

    for name in train_flist:
        im_path = os.path.join(data_dir, d, name)
        label, label_name = i, d
        csv_row = [im_path, label, label_name]
        train_csv_writer.writerow(csv_row)

    for name in test_flist:
        im_path = os.path.join(data_dir, d, name)
        label, label_name = i, d
        csv_row = [im_path, label, label_name]
        test_csv_writer.writerow(csv_row)