import os
import cv2 as cv
import numpy as np

pathTmp= "tmp/"
pathLabels = "multi_label/multi_label/"
pathSave = "binary_masks/"
#putting ground truth binary masks into list
for file in os.listdir(pathLabels):
    plantId = int(file.split("_")[2])
    img = cv.imread(pathLabels+file)
    #cv.imshow(file, img)
    #truePlants[plantId].append(img)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # creating mask to catch plant's "green"
    mask = cv.inRange(hsv, (0, 0, 0), (255, 255, 255))
    imask = mask > 0
    green = np.zeros_like(img, np.uint8)
    # image with only green part
    green[imask] = img[imask]
    # cv.imwrite("test/xd/"+file, green) # to save green only image
    # converting the green only image to binary
    grey = cv.cvtColor(green, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(grey, 0, 255, cv.THRESH_BINARY)
    #cv.imshow(file, thresh)
    #print(grey[grey==255])
    cv.imwrite(pathSave+"binaryMask_"+file.split("_", 1)[1], thresh)

cv.waitKey(0)