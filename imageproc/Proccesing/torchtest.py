import numpy as np
import cv2
import os
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torch.autograd._functions import tensor

from imageproc.Proccesing.ConvolutionalHandler import deCNN, SuperResolutionLoss

finalsize = 500, 500


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
    resize = filepath1.unsqueeze(0)
    resize = np.flip(resize.squeeze().permute(1, 2, 0).cpu().numpy(), axis=2)
    cv2disp("blur", resize.repeat(20, axis=0).repeat(20, axis=1), 0, 0)
    cv2disp("output", np.flip(output.squeeze().permute(1, 2, 0).cpu().detach().numpy(), axis=2), 1000, 0)
    # cv2disp("clear", np.flip(filepath2.squeeze().permute(1, 2, 0).cpu().numpy(), axis=2), 500, 0)


class SuperResLoss:
    def __init__(self):
        super(SuperResLoss, self).__init__()

    def forward(self, output, target):
        return F.mse_loss(output, target)


def runOne(model, criterions, optimizers, epochs, blur, clear):
    losses = []  # List to store the loss for each epoch
    for epoch in range(epochs):
        output = model.forward(blur)
        loss = criterions.forward(output, clear)

        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizers.step()
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")

        draw(blur, clear, output)
        cv2.waitKey(1)

    return losses


def runMany(model, criterions, optimizers, epochs, indexstart, indexlength):
    totallosses = []  # List to store the loss for each epoch
    for epoch in range(epochs):
        losses = []  # List to store the loss for each epoch
        for _ in range(indexstart, indexstart + indexlength):
            blur = load_and_preprocess_image(getBlurry(_), (25, 25))
            clear = load_and_preprocess_image(getClear(_), finalsize)
            output = model.forward(blur)
            loss = criterions.forward(output, clear)

            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizers.step()
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")

            draw(blur, clear, output)
            cv2.waitKey(1)
        totallosses.append(losses)
    return totallosses


def optimizeTrain(model, criterions, optimizers, epochs):
    totallosses = []
    epoch = 0
    indexstart = 0
    indexlength = 5
    blurries = []
    clears = []
    while epoch < epochs:
        losses = []  # List to store the loss for each epoch
        if epoch == 0:
            blurries = [load_and_preprocess_image(getBlurry(_), (25, 25)) for _ in
                        range(indexstart, indexstart + indexlength)]
            clears = [load_and_preprocess_image(getClear(_), (500, 500)) for _ in
                      range(indexstart, indexstart + indexlength)]

        for _ in range(indexstart, indexstart + indexlength):
            blur = blurries[_]
            clear = clears[_]
            output = model.forward(blur)
            loss = criterions.forward(output, clear)

            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizers.step()
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}, " + "Index: " + str(_))

            draw(blur, clear, output)
            cv2.waitKey(1)
        totallosses.append(losses)
        if len(totallosses) > 0 and epoch >= 5 and np.max(totallosses) < 0.06 and np.mean(
                np.abs(totallosses[len(totallosses) - 1] - totallosses[len(totallosses)- 2])) < 0.001:
            print("Ephoc skipi!!")
            epoch = 0
            indexstart += indexlength

        epoch += 1

    return totallosses


criterion = SuperResolutionLoss()
model = deCNN(scale=5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

index = 0

# blur = load_and_preprocess_image(getBlurry(index), (25, 25))
# clear = load_and_preprocess_image(getClear(index), finalsize)
epochs = 50
# lose = runOne(model, criterion, optimizer, epochs, blur, clear)
loseMany = optimizeTrain(model, criterion, optimizer, epochs)
print(loseMany)

plt.plot(range(1, epochs + 1), loseMany, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
# plt.legend()
plt.show()
