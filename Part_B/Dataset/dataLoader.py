  
import albumentations.pytorch
import albumentations as alb
import cv2
import glob
import os
import copy
import numpy as np
from matplotlib import pyplot as plt
import string
import json
import random

from torch.utils.data import Dataset, Subset
from scipy.ndimage import rotate

def preprocess_mask(mask):
    mask              = mask.astype(np.float32)
    mask[mask == 2.0] = 0.0
    mask[(mask == 1.0) | (mask == 3.0)] = 1.0
    return mask

######################################################################################
#
# CLASS DESCRIBING HOW TO LOAD AND ITERATE OVER THE DATASET 
# This custom class inherits from the abstract class torch.utils.data.Dataset
# An instance of OxfordPetDataset has been created in the model.py file
# 
######################################################################################


class OxfordPetDataset(Dataset):
    # --------------------------------------------------------------------------------
    # INITIALISATION OF THE OXFORDPETDATASET INSTANCE
    # INPUTS: 
    #     - imgDirectory (str): path to the folder containing the images 
    #     - maskDirectory (str): path to the folder containing the masks
    #     - setName (string): choose among "train", "val" or "test"
    #     - param (dic): dictionnary containing the parameters defined in the 
    #                    configuration (yaml) file
    # --------------------------------------------------------------------------------
    def __init__(self, imgDirectory, maskDirectory, setName, param):
        self.imgDirectory       = imgDirectory
        self.maskDirectory      = maskDirectory
        self.setName            = setName        
        self.transform_train    = getTransforms_train(param)
        self.transform_val_test = getTransforms_val_test(param)
        self.all_transform   = alb.Compose(
                                            [
                                                alb.Resize(256, 256),
                                                alb.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                                                alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                                alb.GaussNoise(p=1),
                                                alb.ToGray(p=1),
                                                alb.VerticalFlip(p=1),
                                                alb.HorizontalFlip(p=1),
                                                alb.pytorch.transforms.ToTensorV2(),
                                            ]
)
        
        with open("./Dataset/" + self.setName + ".json", 'r') as f:
            self.files = json.load(f)
        
    # -----------------------------------
    # GET NUMBER OF IMAGES IN THE DATASET
    # -----------------------------------
    def __len__(self):
        return len(self.files)

    # ------------------------------------------------------
    # GET THE IDX'TH IMAGE OF THE DATASET
    # INPUTS: 
    #     - idx (int): index of the image you want to access
    # ------------------------------------------------------
    def __getitem__(self, idx):
        filename_img = os.path.join(self.imgDirectory, self.files[idx] + '.jpg')        
        originalImg  = cv2.imread(filename_img)
        originalImg  = cv2.cvtColor(originalImg, cv2.COLOR_BGR2RGB)
        copyImg      = copy.deepcopy(originalImg)

        filename_mask = os.path.join(self.maskDirectory, self.files[idx] + '.png')
        mask          = preprocess_mask(cv2.imread(filename_mask, cv2.IMREAD_UNCHANGED))
        
        if self.setName == 'train':
            transform = self.transform_train
        else:
            transform = self.transform_val_test
            
        if transform is not None:
            transformed = transform(image=originalImg, mask=mask)
            image       = transformed["image"]
            mask_       = transformed["mask"]
            # Used for ploting results (transform = Resize only!)
            resizedTransform = alb.Compose([t for t in  transform if isinstance(t, (alb.Resize))])
            resized          = resizedTransform(image=copyImg, mask=mask)
            resizedImg       = resized["image"]
            resizedMask      = resized["mask"]

            returnResized       = alb.Compose([t for t in  transform if isinstance(t, (alb.pytorch.transforms.ToTensorV2))])
            resized_            = returnResized(image=resizedImg, mask=resizedMask)

            verticalTransform   = alb.Compose([t for t in self.all_transform if isinstance(t, (alb.Normalize,alb.VerticalFlip, alb.pytorch.transforms.ToTensorV2))])
            verticalFlip        = verticalTransform(image=resizedImg, mask=resizedMask)["image"]
            verticalMask        = verticalTransform(image=resizedImg, mask=resizedMask)["mask"]

            horizontalTransform = alb.Compose([t for t in self.all_transform if isinstance(t, (alb.Normalize,alb.HorizontalFlip, alb.pytorch.transforms.ToTensorV2))])
            horizontalFlip      = horizontalTransform(image=resizedImg, mask=resizedMask)["image"]

            rotated_resizedImg  = rotate(resizedImg, 45, reshape=False)
            rotateTransform     = alb.Compose([t for t in self.all_transform if isinstance(t, (alb.Normalize, alb.pytorch.transforms.ToTensorV2))])
            rotateFlip          = rotateTransform(image=rotated_resizedImg, mask=resizedMask)["image"]

            gaussianTransform   = alb.Compose([t for t in self.all_transform if isinstance(t, (alb.Normalize,alb.GaussNoise, alb.pytorch.transforms.ToTensorV2))])
            gaussianNoise       = gaussianTransform(image=resizedImg, mask=resizedMask)["image"]

            grayTransform       =  alb.Compose([t for t in self.all_transform if isinstance(t, (alb.Normalize,alb.ToGray, alb.pytorch.transforms.ToTensorV2))])
            grayImage           = grayTransform(image=resizedImg, mask=resizedMask)["image"]

            brightTransform     = alb.Compose([t for t in self.all_transform if isinstance(t, (alb.Normalize,alb.RandomBrightnessContrast, alb.pytorch.transforms.ToTensorV2))])
            brightImage         = brightTransform(image=resizedImg, mask=resizedMask)["image"]
        
        
        return image, mask_, resized_, verticalFlip, horizontalFlip, rotateFlip, gaussianNoise, grayImage, brightImage

# -----------------------------------------------------------------------
# GET A LIST OF ALBUMENTATION AUGMENTATION TRANSFORMS
# INPUTS: 
#     - param (dic): dictionnary containing the parameters defined in the 
#                    configuration (yaml) file
#
# Do not hesitate to consider other useful transforms!
# -----------------------------------------------------------------------
def getTransforms_train(param): 
    imgTransformsList = [
                        alb.Resize(height = int(param["DATASET"]["RESIZE_SHAPE"].split("x")[0]), 
                                   width  = int(param["DATASET"]["RESIZE_SHAPE"].split("x")[1])),
                        alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                        alb.OneOf([
                            alb.HorizontalFlip(p=1),
                            alb.VerticalFlip(p=1),
                            alb.GaussNoise(p=1),
                            alb.Rotate(limit=45, p=1),
                            alb.Rotate(limit=(-45, 0), p=1),
                            alb.ToGray(p=1)
                        ]),
                        alb.pytorch.transforms.ToTensorV2()
                        ]

    return alb.Compose(imgTransformsList)

def getTransforms_val_test(param): 
    imgTransformsList = [alb.Resize(height = int(param["DATASET"]["RESIZE_SHAPE"].split("x")[0]), 
                                    width  = int(param["DATASET"]["RESIZE_SHAPE"].split("x")[1])), 
                        alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                        alb.pytorch.transforms.ToTensorV2(), 
                        ]
    return alb.Compose(imgTransformsList)

def align_mask(masks,transform_type):

    if transform_type == "vertical":
        batch_transformed = []
        for i in range(len(masks)):
            rotate_transform = alb.VerticalFlip(p=1)
            rotated_mask     = rotate_transform(image=masks[i].transpose(1,2,0))
            batch_transformed.append(rotated_mask["image"].transpose(2,0,1))
        transformed_masks = np.stack(batch_transformed)
        
        return transformed_masks
    
    elif transform_type == "horizontal":
        batch_transformed = []
        for i in range(len(masks)):
            rotate_transform = alb.HorizontalFlip(p=1)
            rotated_mask     = rotate_transform(image=masks[i].transpose(1,2,0)) 
            batch_transformed.append(rotated_mask["image"].transpose(2,0,1))
        transformed_masks = np.stack(batch_transformed)
        return transformed_masks
    
    elif transform_type == "rotation":
        batch_transformed = []
        for i in range(len(masks)):
            rotated_mask = rotate(masks[i].transpose(1,2,0), -45, reshape=False)
            batch_transformed.append(rotated_mask.transpose(2,0,1))
        transformed_masks = np.stack(batch_transformed)
        return transformed_masks 
        

