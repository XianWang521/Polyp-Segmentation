import os
import numpy as np
import cv2
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader

def load_names(path, file_path):
    f = open(file_path, "r")
    data = f.read().split("\n")[:-1]
    images = [str(path)+"/images/"+ name + ".jpg" for name in data]
    masks = [str(path)+"/masks/"+ name + ".jpg" for name in data]
    return images, masks

def load_data(path):
    train_names_path = f"{path}/train.txt"
    valid_names_path = f"{path}/val.txt"
    test_names_path = f"{path}/test.txt"

    train_x, train_y = load_names(path, train_names_path)
    valid_x, valid_y = load_names(path, valid_names_path)
    test_x, test_y = load_names(path, test_names_path)
    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

""" Dataset for the Polyp dataset. """
class PolypDataset(Dataset):
    def __init__(self, images_path, masks_path, size):
        """
        Arguments:
            images_path: A list of path of the images.
            masks_path: A list of path of the masks.
        """
        self.images_path = images_path
        self.masks_path = masks_path
        self.size = size
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, self.size)
        mask = cv2.resize(mask, self.size)

        image = np.transpose(image, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)

        image = image/255.0
        mask = mask/255.0

        image = image.astype(np.float32)
        mask = mask.astype(np.float32)

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        return image, mask

    def __len__(self):
        return self.n_samples