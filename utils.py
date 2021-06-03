import os
import time
import random
from sklearn.utils import shuffle

def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

def create_dir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print("Error")
        
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count/1024/1024