import cv2 as cv
import numpy as np
import os

class myDict(dict):
    #mydict = dict()
    def __init__(self):
        self = dict()
    def myAdd(self,key,value):
        self[key]=value
    def myGet(self, key):
        if self.get(key) == None:
            return 0
        else:
            return self.get(key)
    def myGetKeys(self):
        return self.keys()
    def myLen(self):
        return len(self)

def calculateDiceC(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    intersection = np.logical_and(im1, im2)
    return 2 * intersection.sum() / (im1.sum() + im2.sum())
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

for file in os.listdir(truePlantsPath):
    plantId = int(file.split("_")[2])
    img = cv.imread(truePlantsPath+file)
    truePlants[plantId].append(img)

for file in os.listdir(pathPredictionPlantsTBM):
    plantId = int(file.split("_")[2])
    img = cv.imread(pathPredictionPlantsTBM+file)
    predictedPlants[plantId].append(img)

header = '_tbm'
f = open("results/part2_results" + header + ".txt", "w")
f.write(header)
diceFinal = 0
iofFinal = 0
for m in range(5):
    dicePerLeaf = myDict()
    dicePerPlant = 0
    iofPerLeaf = myDict()
    iofPerPlant = 0
    colorIter = myDict()
    colorIter.myAdd([0, 0, 255], 0)
    colorIter.myAdd([0, 255, 0], 0)
    colorIter.myAdd([0, 255, 255], 0)
    colorIter.myAdd([255, 0, 0], 0)
    colorIter.myAdd([255, 0, 255], 0)
    colorIter.myAdd([255, 255, 0], 0)
    colorIter.myAdd([128, 128, 128], 0)
    colorIter.myAdd([0, 0, 128], 0)
    for n in range(180):
        trouePlant = truePlants[m][n]
        prediction = predictedPlants[m][n]
        #print(trouePlant)
        colorsTrue = []
        colorsPrediction = []
        for i in trouePlant:
            #print("i: ", i)
            for j in i:
                #print("j: ", j)
                if j.tolist() not in colorsTrue:
                    colorsTrue.append(j.tolist())
        #print("colors true: ", colorsTrue)
        for i in prediction:
            for j in i:
                if j.tolist() not in colorsPrediction :
                    colorsPrediction.append(j.tolist())
        # print(n, "pred colors: ", colorsPrediction)
        # print(n, "true colors: ", colorsTrue)
        colorsTrue.remove([0,0,0])
        colorsPrediction.remove([0,0,0])
        #print(colorsTrue)
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
        for j,mask in enumerate(masks):
            ioufori=[]
            dicefori=[]
            if(len(predictions)<1):
                dicefori.append(0)
                ioufori.append(0)
                continue
            for prediction in predictions:
                dice = calculateDiceC(prediction,mask)
                dicefori.append(dice)
                ioufori.append(dice/(2-dice))
            #print(str(colorsTrue[z]))
            #print(m, n, "predictions: ", predictions)
            #f.write(str(m) + " " +  str(n) + " prediction:\n" + str(predictions) + "\n")
            vs = ioufori.index(max(ioufori))
            #indexToDelete = 
            iofPredictions.append(max(ioufori))
            dicePredictions.append(max(dicefori))
            #print(m, n, "test ", predictions[vs])
            #ions.pop(predictions[vs])
            removearray(predictions, predictions[vs])
            # oldDice = dicePerLeaf.myGet(str(colorsTrue[j]))
            # oldDice = oldDice + max(dicefori)
            # dicePerLeaf.myAdd(str(colorsTrue[j]), oldDice)
            # oldIof = iofPerLeaf.myGet(str(colorsTrue[j]))
            # oldIof = oldIof + max(ioufori)
            # iofPerLeaf.myAdd(str(colorsTrue[j]), oldIof)
        #print("len dicePred", len(dicePredictions))
        dicePerLeafL = []
        iofPerLeafL = []
        for i, z in enumerate(colorsTrue):
            if(i<len(dicePredictions)):
                dicePerLeafL.append(dicePredictions[i])
                iofPerLeafL.append(iofPredictions[i]) 
            else:
                dicePerLeafL.append(0)
                iofPerLeafL.append(0)
        for i, color in enumerate(colorsTrue):
            #print(dicePerLeaf.myGet(dl))
            dicePerLeaf.myAdd(str(colorsTrue[i]), dicePerLeafL[i]/len(colorsTrue))
            iofPerLeaf.myAdd(str(colorsTrue[i]), iofPerLeafL[i]/len(colorsTrue))
        #print("dice and iof per leaf:")
        #for i, elem in enumerate(iofPredictions):
        #    print(dicePerLeaf[str(colorsTrue[i])], iofPerLeaf[str(colorsTrue[i])])
        #print(len())
        diceF = sum(dicePredictions)/len(dicePredictions)
        iofF = sum(iofPredictions)/len(iofPredictions)
        dicePerPlant = dicePerPlant + diceF
        iofPerPlant = iofPerPlant + iofF
        #print("dice for plant: ", diceFinal, " iof for plant: ", iofFinal)
    dicePerPlant = dicePerPlant/180
    iofPerPlant = iofPerPlant/180
    diceFinal = diceFinal + dicePerPlant
    iofFinal = iofFinal + iofPerPlant
    msg = "plant id: " + str(m) + " dice: " + str(dicePerPlant) + " iof: " + str(iofPerPlant)
    print(msg)
    f.write(msg + "\n")
    for k in dicePerLeaf.myGetKeys():
        msg = "leaf: " + str(k) + " dice: " + str(dicePerLeaf.myGet(k)) + " iof: " + str(iofPerLeaf.myGet(k))
        print(msg)
        f.write(msg + "\n")
diceFinal = diceFinal/5
iofFinal = iofFinal/5
msg = "for the whole sample: dice: " + str(diceFinal) + " iof: " + str(iofFinal)
print(msg)
f.write(msg + "\n")
f.close()
