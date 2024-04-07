import numpy as np
import cv2
import os
from PIL import Image
import torch
import matplotlib.pyplot as plt

from imageproc.Proccesing.ConvolutionalHandler import deCNN, SuperResolutionLoss

finalsize = 250,250
model = deCNN(10)
criterion = SuperResolutionLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def cv2disp(name, image, xpos, ypos):
    cv2.imshow(name, image)
    cv2.moveWindow(name, xpos, ypos)


def getBlurry(index):
    folder_dir = "C:/Users/kaide/OneDrive/Desktop/vulcan type shit/vulcanfinal/betterproj/imageproc/archive/lowres/"
    image = os.listdir(folder_dir)[index]
    return "C:/Users/kaide/OneDrive/Desktop/vulcan type shit/vulcanfinal/betterproj/imageproc/archive/lowres/" + image


def getClear(index):
    folder_dir = "C:/Users/kaide/OneDrive/Desktop/vulcan type shit/vulcanfinal/betterproj/imageproc/archive/clear/"
    image = os.listdir(folder_dir)[index]
    return "C:/Users/kaide/OneDrive/Desktop/vulcan type shit/vulcanfinal/betterproj/imageproc/archive/clear/" + image


def load_and_preprocess_image(image_path, size):
    image = Image.open(image_path)
    image = image.resize(size)
    image_tensor = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0
    return image_tensor.unsqueeze(0)


def draw(filepath1, filepath2, output):
    pil_image = Image.open(filepath1).resize((finalsize), 4)
    open_cv_image = np.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    cv2disp("blur", np.array(open_cv_image), 0, 0)

    pil_image = Image.open(filepath2).resize((finalsize), 4)
    open_cv_image = np.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    cv2disp("clear", np.array(open_cv_image), 250, 0)

    open_cv_image = np.array(output)
    open_cv_image = np.squeeze(open_cv_image, axis=0)
    open_cv_image = open_cv_image.T
    open_cv_image = np.flip(open_cv_image)
    open_cv_image = np.rot90(open_cv_image)
    open_cv_image = np.flip(open_cv_image, axis=1)
    cv2disp("model", open_cv_image, 500, 0)

def runOne(epochs):
    losses = []  # List to store the loss for each epoch

    learner = SuperResolutionLoss()
    for epoch in range(epochs):
        blur = load_and_preprocess_image(getBlurry(epoch), (25, 25))
        clear = load_and_preprocess_image(getClear(epoch), finalsize)

        output = model.forward(blur)
        loss = learner.forward(output, clear)
        print("Loss:", loss.item())
        draw(getBlurry(epoch), getClear(epoch), output)
        optimizer.zero_grad()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        cv2.waitKey(1)

runOne(100)