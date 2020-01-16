import os
import cv2 as cv
import numpy as np

pathBinaryMasks = "results/part1/"
pathTrueBinaryMasks = "binary_masks/"
pathResults = "results/part2/"
pathResultsTBM = "results/part2_tbm/"

#raport
pathBadCircles = "results/raport/bad/circles/"
pathBadErosion = "results/raport/bad/erosion/"
pathGoodCircles = "results/raport/good/circles/"
pathGoodErosion = "results/raport/good/erosion/"
pathBadCirclesTBM = "results/raport/bad/circles/tbm/"
pathBadErosionTBM = "results/raport/bad/erosion/tbm/"
pathGoodCirclesTBM = "results/raport/good/circles/tbm/"
pathGoodErosionTBM = "results/raport/good/erosion/tbm/"
pathBadTBM = "results/raport/bad/binary_masks/tbm/"
pathBadPrediction = "results/raport/bad/binary_masks/prediction/"
pathGoodTBM = "results/raport/good/binary_masks/tbm/"
pathGoodPrediction = "results/raport/good/binary_masks/prediction/"

def getDistance(x, y, center):
    return np.sqrt((center[0]-y)**2 + (center[1]-x)**2)

def findClosestCenter(x, y, centers):
    if len(centers) == 0:
        return 0
    res = centres[0]
    j = 0
    for i, center in enumerate(centers): 
        if getDistance(x, y, centers[i]) < getDistance(x, y, res):
            res = centres[i]
            j = i
    return j

def calculateDiceC(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    intersection = np.logical_and(im1, im2)
    return 2 * intersection.sum() / (im1.sum() + im2.sum())

leafColors = [[255,0,0],[0,255,0],[255,255,0],[0,0,255],[255,0,255], 
[0,255,255],[128,128,128],[128,0,0]]

i = 0
results = []
iters = 0
MIN_AREA = 40
for file in os.listdir(pathBadPrediction):
    img = cv.imread(pathBadPrediction+file, 0)
    kernel = np.ones((4, 4), np.uint8)
    day = int(file.split("_")[3])
    plantId = int(file.split("_")[2])
    if day <= 2:
        iters = 2
    elif day <= 4:
        iters = 5
    elif day <= 7:
        iters = 7
    else:
        iters = 10
    erosion = cv.erode(img, kernel, iterations=iters)
    #cv.imwrite(pathBadErosion+"erosion_"+file.split("_", 1)[1], erosion)
    contours, _ = cv.findContours(erosion, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    centres = []
    for j, contour in enumerate(contours):
        if cv.contourArea(contours[j]) < MIN_AREA:
            continue
        moments = cv.moments(contours[j])
        # if moments['m00'] == 0:
        #     moments['m00'] = 1
        centres.append((int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00'])))
        #cv.circle(erosion, centres[-1], 3, (0, 0, 0), -1)
    #cv.imwrite(pathBadCircles+"centers_"+file.split("_", 1)[1], erosion)
    #print(file, day, len(centres), centres)
    rows = img.shape[0]
    columns = img.shape[1]
    result = np.zeros((rows, columns, 3))
    for m in range(rows):
        for n in range(columns):
            if img[m][n] > 0:
                result[m][n] = leafColors[findClosestCenter(m, n, centres)]
    cv.imwrite(pathResults+"prediction_"+file.split("_", 1)[1], result)


cv.waitKey(0)
