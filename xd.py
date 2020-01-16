import os
import cv2 as cv
import numpy as np

def calculateDiceC(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    intersection = np.logical_and(im1, im2)
    return 2 * intersection.sum() / (im1.sum() + im2.sum())

prediction = cv.imread("bad/prediction_01_00_003_05.png",0)
groundtruth = cv.imread('bad/label_01_00_003_05.png',0)
#print(prediction)
#print(groundtruth)
# for file in os.listdir("/"):
#     print(file)
dice = calculateDiceC(prediction, groundtruth)
jaccard = dice/(2-dice)
print("bad")
print("Dice: ", dice, " Jaccard: ", jaccard)