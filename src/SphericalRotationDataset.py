from os.path import join
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import torch


### ==================== SPHERICAL ROTATION DATASET ==================================
class SphericalRotationDataset(Dataset):
    """SphericalRotation dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rotated_img = join(self.root_dir,
                        self.csv_data.iloc[idx, 1])
        rotated_img = cv2.imread(rotated_img)
        original_img = join(self.root_dir,
                        self.csv_data.iloc[idx, 1])
        original_img = cv2.imread(original_img)
        y = self.csv_data.iloc[idx, 2:]
        y = np.array([y])
        y = y.astype('float').reshape(-1, 3)
        sample = {'rotated_img': rotated_img, 'original_img': original_img, 'y': y}

        if self.transform:
            sample = self.transform(sample)

        return sample



### ==================== TO TENSOR CLASS ==================================
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        rotated_img, original_img, y = sample['rotated_img'], sample['original_img'], sample['y']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        rotated_img = rotated_img.transpose((2, 0, 1))
        original_img = original_img.transpose((2, 0, 1))
        return {'rotated_img': torch.from_numpy(rotated_img),
                'original_img': torch.from_numpy(original_img),
                'y': torch.from_numpy(y)}








### ==================== TEST FUCTION ==================================


mydataset = SphericalRotationDataset(csv_file='../data/data.csv',
                                    root_dir='../data/',
                                    transform=transforms.Compose([
                                               ToTensor()
                                           ]))

dataloader = DataLoader(mydataset, batch_size=2,
                        shuffle=True, num_workers=0)

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['rotated_img'].size(),
            sample_batched['original_img'].size(), sample_batched['y'].size())
