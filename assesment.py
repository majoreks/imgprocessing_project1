import cv2 as cv
import numpy as np
import os

# helper class 
class myDict(dict):
    def __init__(self):
        self = dict()
    def myAdd(self,key,value):
        tmp=self.get(key,0)
        self[key]=tmp+value
    def myGet(self, key):
        if self.get(key) == None:
            return 0
        else:
            return self.get(key)
    def mySet(self, key, value):
        self[key]=value
    def myGetKeys(self):
        return self.keys()
    def myLen(self):
        return len(self)

def calculateDiceC(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    intersection = np.logical_and(im1, im2)
    return 2 * intersection.sum() / (im1.sum() + im2.sum())

# helper function to remover array from list of arrays
def removearray(L,arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind],arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')

truePlantsPath = 'multi_label/multi_label/'
pathPredictionPlants = 'results/part2/'
pathPredictionPlantsTBM = 'results/part2_tbm/'

truePlants = [[] for y in range(5)]
predictedPlants = [[] for y in range(5)]

# reading images to lists
for file in os.listdir(truePlantsPath):
    plantId = int(file.split("_")[2])
    img = cv.imread(truePlantsPath+file)
    truePlants[plantId].append(img)

for file in os.listdir(pathPredictionPlants):
    plantId = int(file.split("_")[2])
    img = cv.imread(pathPredictionPlants+file)
    predictedPlants[plantId].append(img)

leafColors = [[255,0,0],[0,255,0],[255,255,0],[0,0,255],[255,0,255], 
[0,255,255],[128,128,128],[128,0,0],[64,0,64],[64,64,64],[128,64,255], 
[255,128,64],[128,64,64],[64,255,0],[128,32,32]]

header = '_myPredictions'
f = open("results/part2_results" + header + ".txt", "w")
f.write(header+"\n")
diceFinal = 0
iofFinal = 0
# we go through each plant separately
for m in range(5):
    dicePerLeaf = myDict()
    dicePerPlant = 0
    iofPerLeaf = myDict()
    iofPerPlant = 0
    colorTimes = myDict()
    numOfColors = 0
    # list used to determine how many times we got each colour
    for leafColor in leafColors:
        colorTimes[str(leafColor)]=0
    for n in range(180):
        trouePlant = truePlants[m][n]
        prediction = predictedPlants[m][n]
        colorsTrue = []
        colorsPrediction = []
        # getting colors present in ground truth and input image
        for i in trouePlant:
            for j in i:
                if j.tolist() not in colorsTrue:
                    colorsTrue.append(j.tolist())
        for i in prediction:
            for j in i:
                if j.tolist() not in colorsPrediction :
                    colorsPrediction.append(j.tolist())
        # removing background color
        colorsTrue.remove([0,0,0])
        colorsPrediction.remove([0,0,0])

        # creating masks of each color
        masks=colorsTrue.copy()
        for i, elem in enumerate(colorsTrue):
            color = np.array(colorsTrue[i])
            masks[i] = cv.inRange(trouePlant, color, color)

        predictions=colorsPrediction.copy()
        for i, elem in enumerate(colorsPrediction):
            color = np.array(colorsPrediction[i])
            predictions[i] = cv.inRange(prediction, color, color)

        iofPredictions=[]
        dicePredictions=[]
        for i,mask in enumerate(masks):
            ioufori=[]
            dicefori=[]
            if(len(predictions)<1): # case when we have less colors in input img than in ground truth
                dicePredictions.append(0)
                iofPredictions.append(0)
                dicePerLeaf.myAdd(str(colorsTrue[i]), 0)
                iofPerLeaf.myAdd(str(colorsTrue[i]), 0)
                #xd = colorTimes.get(str(colorsTrue[i]))
                colorTimes.myAdd(str(colorsTrue[i]), 1)
                continue
            for prediction in predictions: # we find max dice between plants and add it to result set
                dice = calculateDiceC(prediction,mask)
                dicefori.append(dice)
                ioufori.append(dice/(2-dice))
            vs = ioufori.index(max(ioufori))
            removearray(predictions, predictions[vs])
            dicePerLeaf.myAdd(str(colorsTrue[i]), max(dicefori)) # dictionary keys are colors so that we know
            iofPerLeaf.myAdd(str(colorsTrue[i]), max(ioufori)) # the result for each leaf color
            xd = colorTimes.get(str(colorsTrue[i]))
            colorTimes[str(colorsTrue[i])] = xd + 1
        for trueColor in colorsTrue: # setting results as 0 for leaves which were not detected by the algorithm
            if trueColor not in colorsPrediction:
                dicePerLeaf.mySet(str(trueColor), 0)
                iofPerLeaf.mySet(str(trueColor), 0)
        numOfColors = len(colorsTrue)
    tmp = 0
    tmpd = 0
    tmpf = 0
    # writing results to file as well as printing them out
    for xd, k in enumerate(dicePerLeaf.myGetKeys()):
        if(xd>numOfColors):
            continue
        d = dicePerLeaf[k]/colorTimes[k]
        ff = iofPerLeaf[k]/colorTimes[k]
        tmpd = tmpd + d
        tmpf = tmpf + ff
        msg = "leaf: " + str(k) + " dice: " + "{:.4f}".format(round(d, 4)) + " iof: " + "{:.4f}".format(round(ff, 4))
        msg.format(round(d, 4), round(ff, 4))
        print(msg)
        f.write(msg + "\n")
    tmpd = tmpd/5
    tmpf = tmpf/5
    diceFinal = diceFinal + tmpd
    iofFinal = iofFinal + tmpf
    msg = "plant id: " + str(m) + " dice: " + "{:.4f}".format(round(tmpd, 4)) + " iof: " + "{:.4f}".format(round(tmpf, 4))
    msg.format(round(tmpd, 5), round(tmpf, 5))
    print(msg)
    f.write(msg + "\n")
    print()
diceFinal = diceFinal/5
iofFinal = iofFinal/5
msg = "for the whole sample: dice: " + "{:.4f}".format(round(diceFinal, 4)) + " iof: " + "{:.4f}".format(round(iofFinal, 4))
msg.format(round(diceFinal, 4), round(iofFinal, 4))
print(msg)
f.write(msg + "\n")
f.close()
