# Binary Segmentation with u-net model  - School Project

This project is divided into two parts and focuses on implementing image segmentation techniques for the Oxford-IIIT Pet Dataset. It involves both a classic segmentation task and the exploration of an open research question.

## Project Overview

### Part A: Classic Segmentation Task

We will implement a deep learning-based approach for binary segmentation of cat and dog images from the Oxford-IIIT Pet Dataset.

- **Dataset**: The dataset consists of 7,390 images of cats and dogs, where each image has a corresponding binary mask that segments the animal from the background.
- **Objective**: Train a neural network that predicts a binary segmentation mask for each image.
- **Libraries**:
  - **PyTorch**: For building and training the deep learning models.
  - **Albumentations**: For performing data augmentation during the training process.


### Part B: Open Research Question

This part explores an advanced research topic based on the work done in Part A. You can choose one of the following research tracks:

- **Uncertainty Estimation**
  - Use test-time data augmentation (TTA) to estimate the model’s uncertainty over segmentation masks.
  - Steps include augmenting the test images, generating multiple predictions, and aligning them to compute pixel-level uncertainty.
