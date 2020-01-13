import os
import cv2 as cv
import numpy as np

# helper function to calculate dice coefficient for two images
# based on JDWarner's solution found on github
def calculateDiceC(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    intersection = np.logical_and(im1, im2)
    return 2 * intersection.sum() / (im1.sum() + im2.sum())

truePlantsPath = "binary_masks/"
truePlants = [[] for y in range(5)]
#putting ground truth binary masks into list
for file in os.listdir(truePlantsPath):
    plantId = int(file.split("_")[2])
    img = cv.imread(truePlantsPath+file, 0)
    truePlants[plantId].append(img)

plantsPath = "multi_plant/multi_plant/"
predictedPlants = [[] for y in range(5)]
for file in os.listdir(plantsPath):
    plantId = int(file.split("_")[2])
    day = int(file.split("_")[3])
    img = cv.imread(plantsPath+file)

    # converting the image to HSV colour space
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # creating mask to catch plant's "green"
    mask = cv.inRange(hsv, (40, 40, 37), (70, 200, 150))
    imask = mask > 0
    green = np.zeros_like(img, np.uint8)
    # image with only green part
    green[imask] = img[imask]
    # cv.imwrite("test/xd/"+file, green) # to save green only image
    # converting the green only image to binary
    grey = cv.cvtColor(green, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(grey, 0, 255, cv.THRESH_BINARY)
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
    # writing results to a directory and a list
    cv.imwrite("results/part1/"+"prediction_"+file.split("_", 1)[1], closing)
    predictedPlants[plantId].append(closing)

f = open("part1_results.text", "w")
# calculating dice coeff and jaccard index
diceCeoffWS = 0
diceCoeff = [0 for x in range(5)]
jaccardIndex = [0 for x in range(5)]
for i in range(180):
    for j in range(5):
        diceCoeff[j] = diceCoeff[j] + calculateDiceC(predictedPlants[j][i], truePlants[j][i])
for i in range(5):
    diceCoeff[i] = diceCoeff[i] / 180
    jaccardIndex[i] = diceCoeff[i]/(2-diceCoeff[i])
    # printing info for individual plants
    print("Plant " + i.__str__() + " " + diceCoeff[i].__str__() + " "
        + jaccardIndex[i].__str__())
    f.write("Plant " + str(i) + " " + str(diceCoeff[i]) + " "
        + str(jaccardIndex[i]))
diceCeoffWS = sum(diceCoeff) / 5
jaccardIndexWS = diceCeoffWS/(2-diceCeoffWS)
# printing info for the whole sample
print("Dice coeff for the whole set = " + diceCeoffWS.__str__())
print("Jaccard index  for the whole set = " + jaccardIndexWS.__str__())
f.write("Dice coeff for the whole set = " + str.(diceCeoffWS))
f.write("Jaccard index  for the whole set = " + str(jaccardIndexWS))
cv.waitKey(0)
f.close()
