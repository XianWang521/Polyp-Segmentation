import os
import numpy as np
import cv2
from tqdm import tqdm
from albumentations import Crop, RandomRotate90, ElasticTransform, GridDistortion, OpticalDistortion, HorizontalFlip, VerticalFlip, CenterCrop

def augment_data(images, masks, save_path, augment=True):
    size = (512, 512)
    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        image_name = x.split("/")[-1].split(".")[0]
        mask_name = y.split("/")[-1].split(".")[0]

        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_COLOR)

        if x.shape[0] >= size[0] and x.shape[1] >= size[1]:
            if augment == True:
                x_max=size[0]
                y_max=size[1]
                aug1 = Crop(p=1, x_min=0, x_max=x_max, y_min=0, y_max=y_max)
                x1 = aug1(image=x)['image']
                y1 = aug1(mask=y)['mask']

                aug2 = RandomRotate90(p=1)
                x2 = aug2(image=x)['image']
                y2 = aug2(mask=y)['mask']

                aug3 = ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
                x3 = aug3(image=x)['image']
                y3 = aug3(mask=y)['mask']

                aug4 = GridDistortion(p=1)
                x4 = aug4(image=x)['image']
                y4 = aug4(mask=y)['mask']

                aug5 = OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
                x5 = aug5(image=x)['image']
                y5 = aug5(mask=y)['mask']
    
                aug6 = VerticalFlip(p=1)
                x6 = aug6(image=x)['image']
                y6 = aug6(mask=y)['mask']

                aug7 = HorizontalFlip(p=1)
                x7 = aug7(image=x)['image']
                y7 = aug7(mask=y)['mask']
                
                aug8 = CenterCrop(384, 384, p=1)
                x8 = aug8(image=x)['image']
                y8 = aug8(mask=y)['mask']

                aug9 = CenterCrop(448, 448, p=1)
                x9 = aug9(image=x)['image']
                y9 = aug9(mask=y)['mask']

                # x8
                aug10 = VerticalFlip(p=1)
                x10 = aug10(image=x8)['image']
                y10 = aug10(mask=y8)['mask']

                # x8
                aug11 = HorizontalFlip(p=1)
                x11 = aug11(image=x8)['image']
                y11 = aug11(mask=y8)['mask']

                # x9
                aug12 = VerticalFlip(p=1)
                x12 = aug12(image=x9)['image']
                y12 = aug12(mask=y9)['mask']

                # x9
                aug13 = HorizontalFlip(p=1)
                x13 = aug13(image=x9)['image']
                y13 = aug13(mask=y9)['mask']

                images = [
                    x, x1, x2, x3, x4, x5, x6, 
                    x7, x8, x9, x10, x11, x12, x13
                ]
                masks  = [
                    y, y1, y2, y3, y4, y5, y6, 
                    y7, y8, y9, y10, y11, y12, y13
                ]
            else:
                images = [x]
                masks  = [y]
            
            idx = 0
            
        for i, m in zip(images, masks):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)

            if augment == True:
                tmp_image_name = f"{image_name}_{idx}.jpg"
                tmp_mask_name  = f"{mask_name}_{idx}.jpg"
            else:
                tmp_image_name = f"{image_name}.jpg"
                tmp_mask_name  = f"{mask_name}.jpg"
                

            image_path = os.path.join(save_path, "image/", tmp_image_name)
            cv2.imwrite(image_path, i)
                                       
            mask_path  = os.path.join(save_path, "mask/", tmp_mask_name)
            cv2.imwrite(mask_path, m)

            idx += 1