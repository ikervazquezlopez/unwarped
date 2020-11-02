from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

from SphericalRotationDataset import SphericalRotationDataset,ToTensor

plt.ion()   # interactive mode


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(6, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 36, 5)
        self.fc1 = nn.Linear(36 * 46 * 96, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 36 * 46 * 96)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



net = Net()
net = net.float()



criterion = nn.L1Loss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


dataset = SphericalRotationDataset(csv_file='../data/data.csv',
                                    root_dir='../data/',
                                    transform=transforms.Compose([
                                               ToTensor()
                                           ]))

dataloader = DataLoader(dataset, batch_size=2,
                        shuffle=True, num_workers=0)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, sample in enumerate(dataloader):

        rotated, original, y = sample['rotated_img'], sample['original_img'], sample['y']
        inputs = torch.cat([rotated, original],dim=1)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.float())
        loss = criterion(outputs.float(), y)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
