import os
import pandas as pd
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
from torchvision.transforms import Compose, ToTensor, Lambda



class faceImg(Dataset):
    def __init__(self, dataPath, imageDir, transform=None):
        self.data = pd.read_csv(dataPath).dropna()
        self.imageDir = imageDir
        self.transform = transform
        self.oneHotEncoder = OneHotEncoder(sparse_output=False)
        self.labels = self.oneHotEncoder.fit_transform(self.data['class'].values.reshape(-1, 1))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.imageDir, self.data.iloc[idx]['filename'])
        image = cv2.imread(img_name)
        image = cv2.resize(image, (128, 128))
        image = image / 255.0 # normalize
        # horizontal flip
        # image = cv2.flip(image, 1) 
        image = gaussian_noise(image) 
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


def adversarial_noise(image, img_gradient, epsilon = 0.1): # defualt noise
    sign_data_gradient = img_gradient.sign()
    noisy_image = image + epsilon * sign_data_gradient
    noisy_image = torch.clamp(noisy_image, 0, 1)
    return noisy_image

def gaussian_noise(image): # usable
    std = 0.1
    mean = 0
    noise = np.random.normal(mean, std, image.shape)
    imgWithNoise = image + noise
    return np.clip(imgWithNoise, 0, 1)


def preprocess(dataPath, imageDir, batch_size = 32,drop_last = False): # <<<<---
    transform = Compose([
        ToTensor(),  
        Lambda(lambda x: x.float()) 
        ])
    dataset = faceImg(dataPath, imageDir,transform)
    dataLoader = DataLoader(dataset, batch_size, shuffle = True,drop_last=drop_last)
    # print(dataLoader)
    return dataLoader


# dataPath = "Data/train/_annotations.csv"
# imageDir = "Data/train/"
# data = preprocess(dataPath, imageDir)

# for images, labels in data:
#     print(images.shape, labels.shape)