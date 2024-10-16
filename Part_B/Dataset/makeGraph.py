from matplotlib import pyplot as plt
import numpy as np
import cv2
import os


# --------------------------------------------------------------------------------
# CREATE A FOLDER IF IT DOES NOT EXIST
# INPUT: 
#     - desiredPath (str): path to the folder to create
# --------------------------------------------------------------------------------
def createFolder(desiredPath): 
    if not os.path.exists(desiredPath):
        os.makedirs(desiredPath)


# --------------------------------------------------------------------------------
# DISPLAY A 5X5 IMAGES FROM A DATALOADER INSTANCE
# INPUTS: 
#     - dataLoader (Dataset): Instance of Dataset
#     - param (dic): dictionnary containing the parameters defined in the 
#                    configuration (yaml) file
# --------------------------------------------------------------------------------
def showDataLoader(dataLoader, param):
    cols, rows = 5, 5
    figure, ax = plt.subplots(nrows=rows, ncols=cols, dpi=280)
    row = 0
    for (imgBatch, maskBatch, _) in dataLoader:
        if row == rows: 
            break
        for col in range(cols): 
            img  = imgBatch[col].numpy().astype('uint8')
            mask = maskBatch[col].numpy().astype('uint8')
            ax[row, col].imshow(img.transpose((1, 2, 0)))
            ax[row, col].imshow(mask*255, interpolation="nearest", alpha=0.3, cmap='Oranges')
            ax[row, col].set_axis_off()
        row += 1 
    plt.tight_layout()
    plt.show()
    

# --------------------------------------------------------------------------------
# DISPLAY A SINGLE IMAGE AND GT MASK WITH THE CORRESPONDING PRECICTION 
# INPUTS: 
#     - img (arr): a 3D numpy array containing the image (ch x depth x width)
#     - pred (arr): a binary 2D numpy array containing the predicted mask
#                  (depth x width)
#     - GT (arr): a binary 2D numpy array containing the ground truth mask
#                  (depth x width)
#     - filePath (str): path to save the image file
# --------------------------------------------------------------------------------
def singlePrediction(img, pred, GT, filePath): 
    figure, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(img.transpose((1, 2, 0))) 
    ax[0].set_title("Input")
    ax[0].set_axis_off()

    ax[1].imshow(GT, interpolation="nearest")
    ax[1].set_title("GT")
    ax[1].set_axis_off()

    ax[2].imshow(np.squeeze(pred)*255, interpolation="nearest")
    ax[2].set_title("Prediction")
    ax[2].set_axis_off()

    plt.tight_layout()
    plt.show()
    plt.savefig(filePath)
    plt.close()

# --------------------------------------------------------------------------------
# DISPLAY ALL IMAGES AND PREDICTIONS
# INPUTS: 
#     - allInputs (list): list of 3D numpy arrays containing the input images 
#     - allPreds (list): list of 2D numpy arrays containing the predicted masks
#     - allGT (list): list of 2D numpy arrays containing the ground truth masks 
#     - resultPath (str): path to folder in which to save the image files
# --------------------------------------------------------------------------------
def showPredictions(allInputs, allPreds, allGT, resultPath):
    idx = 0
    for (img, pred, GT) in zip(allInputs, allPreds, allGT): 
        filePath = os.path.join(resultPath, "Test", str(idx))
        createFolder(os.path.join(resultPath, "Test"))
        singlePrediction(img, pred, GT, filePath)
        if idx > 30: 
            break
        idx += 1


# --------------------------------------------------------------------------------
# DENORMALIZE IMAGES FOR PLOTTING
# INPUTS: 
#     - norm_image(list): one 3D numpy array containing the input image 
# --------------------------------------------------------------------------------    
def denorm(norm_image):
    denorm_image = norm_image
    for ch in np.arange(0,3):
        denorm_image[:,:,ch] = (denorm_image[:,:,ch]-np.min(denorm_image[:,:,ch]))/(np.max(denorm_image[:,:,ch])-np.min(denorm_image[:,:,ch]))
    return(denorm_image)
    


# --------------------------------------------------------------------------------
# DISPLAY ALL IMAGES ALL TRANSFORMS AND THEIR PREDICTIONS
# INPUTS: 
#     - resizedImg (tensor): resized 3D array containing the input images 
#     - GT (tensor): 2D pytorch tensor containing the predicted mask
#     - Transformation (tensor): every transform of the image (not one parameter)
#     - Transformation predictions (tensor): prediction segmentation mask of a transformation(not one parameter)
#     - Entropy of the predicted masks (tensor): entropy matrix of predicted segmentation masks
# --------------------------------------------------------------------------------
def augmentation_plots(resizedImg, GT, vertical_flip, horizontal_flip, rotateFlip, grayImage, brightImage, p_vertical, p_horizontal, p_rotate, p_gray, p_randomBright, align_vertical, align_horizontal, align_rotation, entropy_matrix):
    
    plt.figure(figsize=(15, 7.5),constrained_layout = True)
    # transformation
    plt.subplot(3,6,1)
    plt.imshow(resizedImg.permute(1,2,0))
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(3,6,2)
    plt.imshow(denorm(vertical_flip.permute(1,2,0).numpy()))
    plt.title('Vertical Flip')
    plt.axis('off')
    plt.subplot(3,6,3)
    plt.imshow(denorm(horizontal_flip.permute(1,2,0).numpy()))
    plt.title('Horizontal flip')
    plt.axis('off')
    plt.subplot(3,6,4)
    plt.imshow(denorm(rotateFlip.permute(1,2,0).numpy()))
    plt.title('45° rotation')
    plt.axis('off')
    plt.subplot(3,6,5)
    plt.imshow(denorm(grayImage.permute(1,2,0).numpy()))
    plt.title('Gray Image')
    plt.axis('off')
    plt.subplot(3,6,6)
    plt.imshow(denorm(brightImage.permute(1,2,0).numpy()))
    plt.title('Brightened Image')
    plt.axis('off')
    
    # masks
    plt.subplot(3,6,7)
    plt.imshow(GT)
    plt.title('GT')
    plt.axis('off')
    plt.subplot(3,6,8)
    plt.imshow(p_vertical.permute(1,2,0))
    plt.title('Vertical Flip \n prediction mask', fontsize = 10)
    plt.axis('off')
    plt.subplot(3,6,9)
    plt.imshow(p_horizontal.permute(1,2,0))
    plt.title('Horizontal flip \n prediction mask', fontsize = 10)
    plt.axis('off')
    plt.subplot(3,6,10)
    plt.imshow(p_rotate.permute(1,2,0))
    plt.title('45° rotation \n prediction mask', fontsize = 10)
    plt.axis('off')
    plt.subplot(3,6,11)
    plt.imshow(p_gray.permute(1,2,0))
    plt.title('Gray Image \n prediction mask', fontsize = 10)
    plt.axis('off')
    plt.subplot(3,6,12)
    plt.imshow(p_randomBright.permute(1,2,0))
    plt.title('Brightened Image \n prediction mask', fontsize = 10)
    plt.axis('off')
    
    # entroy mask + re alligned masks
    plt.subplot(3,6,13)
    plt.imshow(entropy_matrix.permute(1,2,0))
    plt.title('Uncertainty entropy \n mask')
    plt.axis('off')
    plt.subplot(3,6,14)
    plt.imshow(align_vertical.permute(1,2,0))
    plt.title('Vertical Flip \n aligned mask', fontsize = 10)
    plt.axis('off')
    plt.subplot(3,6,15)
    plt.imshow(align_horizontal.permute(1,2,0))
    plt.title('Horizontal flip \n aligned mask', fontsize = 10)
    plt.axis('off')
    plt.subplot(3,6,16)
    plt.imshow(align_rotation.permute(1,2,0))
    plt.title('45° rotation \n aligned mask', fontsize = 10)
    plt.axis('off')
    plt.subplot(3,6,17)
    plt.imshow(p_gray.permute(1,2,0))
    plt.title('Gray Image \n aligned mask', fontsize = 10)
    plt.axis('off')
    plt.subplot(3,6,18)
    plt.imshow(p_randomBright.permute(1,2,0))
    plt.title('Brightened Image \n aligned mask', fontsize = 10)
    plt.axis('off')

    plt.savefig("entropyVStransform.pdf", format="pdf")
    plt.show()
