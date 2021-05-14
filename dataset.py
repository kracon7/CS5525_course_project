import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
from PIL import Image

class MushroomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the voxels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file, header=None)
        self.num_classes = int(self.data_frame.iloc[-1,1] + 1)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        im_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        # im = np.asarray(Image.open(im_path)).transpose(2,0,1)
        # torch_data = torch.from_numpy(im).float()/255
        torch_data = self.transform(Image.open(im_path))

        # label
        label = self.data_frame.iloc[idx, 1]
        torch_label = torch.tensor(label).long()
        # torch_label = F.one_hot(torch_label, num_classes=self.num_classes).float()

        # mushroom name
        name = self.data_frame.iloc[idx, 2]

        sample = {'data': torch_data, 'label': torch_label, 'name': name}
        return sample