import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for the optimizer ")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--use_gpu',type = bool, default=True , help="Flag to use GPU if available")
    parser.add_argument('--depth',type = int, default=3, help="Depth of model")
    parser.add_argument('--inter_channel',type = int, default=16, help="channel of inter layer ")

    args = parser.parse_args()


if __name__ == "__main__":
    main()
