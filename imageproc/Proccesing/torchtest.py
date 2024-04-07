import numpy as np
import cv2
import os
from PIL import Image



def cv2disp(name, image, xpos, ypos):
    cv2.imshow(name, image)
    cv2.moveWindow(name, xpos, ypos)

def getBlurry(index):
    folder_dir = "C:/Users/kaide/OneDrive/Desktop/vulcan type shit/vulcanfinal/betterproj/imageproc/archive/lowres/"
    image = os.listdir(folder_dir)[index]
    pil_image = Image\
        .open("C:/Users/kaide/OneDrive/Desktop/vulcan type shit/vulcanfinal/betterproj/imageproc/archive/lowres/" + image)\
        .resize((250,250))\
        .convert('HSV')
    open_cv_image = np.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image
def getClear(index):
    folder_dir = "C:/Users/kaide/OneDrive/Desktop/vulcan type shit/vulcanfinal/betterproj/imageproc/archive/clear/"
    image = os.listdir(folder_dir)[index]
    pil_image = Image.open("C:/Users/kaide/OneDrive/Desktop/vulcan type "
                           "shit/vulcanfinal/betterproj/imageproc/archive/clear/" + image).convert('HSV')
    open_cv_image = np.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


blur = getBlurry(500)
clear = getClear(500)


cv2disp("blur", blur, 0,0)
cv2disp("clear", clear, 100,100)

cv2.waitKey(0)

