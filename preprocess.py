import os
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def add_noise(image):
    std = 0.1
    mean = 0
    noise = np.random.normal(mean, std, image.shape)
    imgWithNoise = image + noise
    return np.clip(imgWithNoise, 0, 1)


def preprocess(dataPath, imageDir):
    data = pd.read_csv(dataPath)
    data = data.dropna()

    # take image name from data and load it into list
    image_paths = data['filename'].apply(lambda x: os.path.join(imageDir, x)).values
    labels = data['class'].values

    # OneHot Encoding on labels
    encoder = OneHotEncoder(sparse_output=False)
    labels = encoder.fit_transform(labels.reshape(-1, 1))

    # load img
    images = []
    for path in image_paths:
        try:
            img = cv2.imread(path)
            img = cv2.resize(img, (128, 128))
            #normalize
            img = img / 255.0
            # add noise
            img = add_noise(img)
            # horizontal flip
            img = cv2.flip(img, 1) 
            images.append(img)
        except Exception as e:
            print("There is problem loading image ----> ", str(e))
    
    images = np.array(images)

    print(images)
    print(images.shape)
    print(labels)
    print(labels.shape)
    return images, labels


dataPath = "Data/train/_annotations.csv"
imageDir = "Data/train/"
preprocess(dataPath, imageDir)