import sys, os
default_path = os.path.join(os.path.dirname(__file__), '../../')
sys.path.append(default_path + 'utils/')
import csv
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import h5py
from dataset_utils import h5_to_np, np_to_h5
import pandas as pd
import matplotlib.pyplot as plt
class PushDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.h5f_data = h5py.File(os.path.join(os.path.dirname(__file__), self.image_dir), "r")
        self.images = h5_to_np(self.h5f_data, 400, 400)
        self.images[self.images < 0.5] = 0
        self.images[self.images == 0.5] = 0.5
        self.images[self.images > 0.5] = 1
        self.length = self.images.shape[0]
        print('images shape is ' + str(self.images.shape))
        self.label_data_pd = pd.read_csv(self.mask_dir)
        self.masks = self.label_data_pd.label.to_numpy().reshape(-1,1)

    def preprocess_data(self):
        self.images[self.images == 0.500] = 0
        self.images[self.images == 0.2188] = 0
        self.images[self.images == 0.203] = 0

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        image = self.images[index, :]
        mask = self.masks[index]
        if self.transform is not None:
            augmentation = self.transform(image=image)
            image = augmentation["image"]

        return image, mask

def random_test_data():
    x = np.random.uniform(0, 1, (10, 400, 400))
    print(x.shape)
    np.save('./aaa.npy', x)
    y = np.random.randint(0, 2, (10, 1))
    print(y.shape)
    with open('./bbb.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for mask in y:
            spamwriter.writerow(mask)


if __name__ == "__main__":
    random_test_data()
    pushdata = PushDataset('./aaa.npy', './bbb.csv')
    pushdata.__getitem__(0)