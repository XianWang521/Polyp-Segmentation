import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

def reverse_image(x):
    pred_y = x.cpu().numpy()
    pred_y = np.squeeze(pred_y, axis=0)
    pred_y = pred_y.astype(np.float64)
    pred_y = pred_y * 255
    pred_y = np.transpose(pred_y, (2, 1, 0))
    pred_y = np.array(pred_y, dtype=np.uint8)

    image = cv2.cvtColor(pred_y, cv2.COLOR_BGR2RGB)
    return image

def show_image(x, y, p):
    plt.figure(figsize = (20,20))
    f, axarr = plt.subplots(1, 3)
    plt.subplots_adjust(left = 5, right = 7)

    image = reverse_image(x)
    axarr[0].set_title("image")
    axarr[0].imshow(image)

    mask = torch.squeeze(y.cpu(), dim = 0).numpy().T
    axarr[1].set_title("mask")
    axarr[1].imshow(mask)

    p = (p >= 0.5).float()
    pred = torch.squeeze(p.cpu()).numpy().T
    axarr[2].set_title("predict")
    axarr[2].imshow(pred)