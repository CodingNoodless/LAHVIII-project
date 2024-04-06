import numpy as np
import torch
import cv2
import tensorflow as tf
import tarfile
import os
import glob
from PIL import Image


def cv2disp(name, image, xpos, ypos):
    cv2.imshow(name, image)
    cv2.moveWindow(name, xpos, ypos)

def getBlurry(index):
    folder_dir = "C:/Users/kaide/OneDrive/Desktop/vulcan type shit/vulcanfinal/LAHVIII-project/imageproc/archive/lowres"
    image = os.listdir(folder_dir)[index]
    pil_image = Image\
        .open("C:/Users/kaide/OneDrive/Desktop/vulcan type ""shit/vulcanfinal/LAHVIII-project/imageproc/archive/lowres/" + image)\
        .resize((250,250))\
        .convert('RGB')
    open_cv_image = np.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image
def getClear(index):
    folder_dir = "C:/Users/kaide/OneDrive/Desktop/vulcan type shit/vulcanfinal/LAHVIII-project/imageproc/archive/clear"
    image = os.listdir(folder_dir)[index]
    pil_image = Image.open("C:/Users/kaide/OneDrive/Desktop/vulcan type "
                           "shit/vulcanfinal/LAHVIII-project/imageproc/archive/clear/" + image).convert('RGB')
    open_cv_image = np.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


blur = getBlurry(0)
clear = getClear(0)

cv2disp("blur", blur, 0,0)
cv2disp("clear", clear, 100,100)

cv2.waitKey(0)