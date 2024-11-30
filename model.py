import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.utils import shuffle,resample
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import RandomSampler, DataLoader, Subset
from torchvision.transforms import Compose, RandomHorizontalFlip, Grayscale, Resize, RandomCrop, ToTensor


class CNN_layer(nn.Module):
    def __init__(self,in_cn,mid_cn,out_cm=None,if_pool=False):
        super(CNN_layer, self).__init__()
        
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_cn, out_channels=mid_cn, kernel_size=3,stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_cn, out_channels=out_cm, kernel_size=3,stride = 1, padding = 1),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(kernel_size=2)
        
    def forward(self,x):
        
        x = self.conv_layer1(x)
        x = self.pool(x)
        
        return x



class Self_Attention_layer(nn.Module):
    def __init__(self, in_channels):
        """
        :param in_channels: Number of input channels
        """
        super(Self_Attention_layer, self).__init__()
        # Convolutional layers for Q, K, V
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        # Output transformation
        self.fc_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape

        # Compute Q, K, V
        Q = self.query_conv(x)  # (batch_size, in_channels, h, w)
        K = self.key_conv(x)    # (batch_size, in_channels, h, w)
        V = self.value_conv(x)  # (batch_size, in_channels, h, w)

        # Flatten spatial dimensions (h, w) for attention
        Q = Q.view(b, c, -1)  # (batch_size, in_channels, h*w)
        K = K.view(b, c, -1)  # (batch_size, in_channels, h*w)
        V = V.view(b, c, -1)  # (batch_size, in_channels, h*w)

        # Compute attention scores
        attention = torch.softmax(torch.bmm(Q.permute(0, 2, 1), K) / (c ** 0.5), dim=-1)  # (batch_size, h*w, h*w)

        # Apply attention to V
        out = torch.bmm(V, attention.permute(0, 2, 1))  # (batch_size, in_channels, h*w)
        out = out.view(b, c, h, w)  # Reshape back to original dimensions

        # Final transformation
        out = self.fc_out(out)  # (batch_size, in_channels, h, w)
        return out

