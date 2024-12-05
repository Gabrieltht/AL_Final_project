import os
import pandas as pd
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
from torchvision.transforms import Compose, ToTensor, Lambda, Compose, ToTensor, RandomHorizontalFlip, RandomRotation, ColorJitter, Normalize, Lambda
# import torchvision.transforms as transforms

class FER_2013(Dataset):
    def __init__(self, dataPath,type,transform=None):
        self.data = pd.read_csv(dataPath).dropna()
        self.transform = transform
        self.type = type
        if type not in ['Training', 'PublicTest', 'PrivateTest']:
            raise ValueError(f"Invalid type '{type}'. Must be one of ['Training', 'PublicTest', 'PrivateTest']")

        # Filter rows by the 'Usage' column
        self.data = self.data[self.data['Usage'] == type].reset_index(drop=True)
        self.labels = torch.tensor(self.data['emotion'].values,dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load the image and reshape to (48, 48)
        pixels = self.data.iloc[idx]['pixels']
        image = np.fromstring(pixels, dtype=np.uint8, sep=' ').reshape(48, 48)
        
        # Resize the cropped image to 128x128
        image = image / 255.0
        resized_image = cv2.resize(image, (128, 128))
        if self.type == 'Training':
            resized_image = gaussian_noise(resized_image)
        
        label = self.labels[idx]


        # Apply optional transformation (if any)
        if self.transform:
            resized_image = self.transform(resized_image)
        

        return resized_image, label


class faceImg(Dataset):
    def __init__(self, dataPath, imageDir, transform=None):
        self.data = pd.read_csv(dataPath).dropna()
        self.imageDir = imageDir
        self.transform = transform
        # self.oneHotEncoder = OneHotEncoder(sparse_output=False)
        labels = self.data['class'].values.reshape(-1, 1)
        classes = np.array(['angry', 'disgust', 'fear', 'surprise', 'neutral', 'happy', 'sad'])
        class_to_index = {cls: idx for idx, cls in enumerate(classes)}
        num_classes = len(classes)
        self.one_hot_encoded_label = np.zeros((len(labels), num_classes), dtype=float)

        # Populate the one-hot encoded array
        for i, label in enumerate(labels):
            index = class_to_index[label[0]]
            self.one_hot_encoded_label[i, index] = 1.0
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image path and read the image
        img_name = os.path.join(self.imageDir, self.data.iloc[idx]['filename'])
        image = cv2.imread(img_name)
        
        # Crop the image using bounding box
        xmin = int(self.data.iloc[idx]['xmin'])
        ymin = int(self.data.iloc[idx]['ymin'])
        xmax = int(self.data.iloc[idx]['xmax'])
        ymax = int(self.data.iloc[idx]['ymax'])
        cropped_image = image[ymin:ymax, xmin:xmax]
        
        # Resize the cropped image to 128x128
        resized_image = cv2.resize(cropped_image, (128, 128))
        
        # Get the corresponding label
        label = self.one_hot_encoded_label[idx]
        
        # test,save img
        # cv2.imwrite(f"./test/{xmin}_{xmax}_{ymin}_{ymax}_{self.data.iloc[idx]['class']}_"+self.data.iloc[idx]['filename'],resized_image)
        
        # Normalize the image
        resized_image = resized_image / 255.0
        
        
        resized_image = gaussian_noise(resized_image)
        # Apply optional transformation (if any)
        if self.transform:
            resized_image = self.transform(resized_image)
        

        label = torch.tensor(np.argmax(self.one_hot_encoded_label[idx])).long()
        return resized_image, label


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


def FER_preprocess(dataPath,type, batch_size = 32,drop_last = False): # <<<<---
    # transform = Compose([
    #     ToTensor(),  
    #     Lambda(lambda x: x.float()) 
    #     ])
    transform = Compose([
        ToTensor(),  
        RandomHorizontalFlip(p=0.3),  
        RandomRotation(degrees=5),    
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        Lambda(lambda x: x.float()),
        Normalize(mean=[0.5074], std=[0.2553])  # Normalization for grayscale
        ])
    dataset = FER_2013(dataPath,type,transform)
    dataLoader = DataLoader(dataset, batch_size, shuffle = True,drop_last=drop_last)
    return dataLoader


# dataPath = "Data/train/_annotations.csv"
# imageDir = "Data/train/"
# data = preprocess(dataPath, imageDir)

# for images, labels in data:
#     print(images.shape, labels.shape)