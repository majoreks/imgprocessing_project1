import os
import cv2 as cv
import numpy as np

pathBinaryMasks = "results/part1/" # path for obtained binary masks
pathTrueBinaryMasks = "binary_masks/" # path for ground truth binary masks
pathResults = "results/part2/" # path for results
pathResultsTBM = "results/part2_tbm/" # using ground truth binary masks instead of obtained ones

# helper function to find distane between a point and a center
def getDistance(x, y, center):
    return np.sqrt((center[0]-y)**2 + (center[1]-x)**2)

# helper function to find closest center to given coordinates
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

# helper function to calculate dice coeff
def calculateDiceC(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    intersection = np.logical_and(im1, im2)
    return 2 * intersection.sum() / (im1.sum() + im2.sum())

leafColors = [[255,0,0],[0,255,0],[255,255,0],[0,0,255],[255,0,255], 
[0,255,255],[128,128,128],[128,0,0],[64,0,255],[128,128,128]]

i = 0
results = []
iters = 0
MIN_AREA = 40
for file in os.listdir(pathBinaryMasks):
    img = cv.imread(pathBinaryMasks+file, 0)
    kernel = np.ones((4, 4), np.uint8)
    # reading plant id and day of image
    plantId = int(file.split("_")[2])
    day = int(file.split("_")[3])
    # changing iters for erosion based on day the image was taken
    if day <= 2:
        iters = 2
    elif day <= 4:
        iters = 5
    elif day <= 7:
        iters = 7
    else:
        iters = 10
    iters = 7
    erosion = cv.erode(img, kernel, iterations=iters)
    # cv.imwrite(pathResTMP+"erosion_"+file.split("_", 1)[1], erosion)
    contours, _ = cv.findContours(erosion, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    centres = []
    for j, contour in enumerate(contours):
        # eliminating contours that were too small to be considered as leaves (trying to not mark noise as leaf)
        if cv.contourArea(contours[j]) < MIN_AREA:
            continue
        moments = cv.moments(contours[j])
        centres.append((int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00'])))
        # cv.circle(erosion, centres[-1], 3, (0, 0, 0), -1) # for report
    # cv.imwrite(pathBadCircles+"centers_"+file.split("_", 1)[1], erosion)
    # getting size of the image 
    rows = img.shape[0]
    columns = img.shape[1]
    # coloring every pixel which was not black in input mask to given color based on the closest center
    result = np.zeros((rows, columns, 3))
    for m in range(rows):
        for n in range(columns):
            if img[m][n] > 0:
                result[m][n] = leafColors[findClosestCenter(m, n, centres)]
    cv.imwrite(pathResults+"prediction_"+file.split("_", 1)[1], result)


cv.waitKey(0)
