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

def calculateDiceC(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    intersection = np.logical_and(im1, im2)
    return 2 * intersection.sum() / (im1.sum() + im2.sum())

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
    plantId = int(file.split("_")[3])
    img = cv.imread(pathPredictionPlantsTBM+file)
    predictedPlants[plantId].append(img)

header = '_tbm'
f = open("part2_results" + header + ".txt", "w")
f.write(header)
#print(len(predictedPlants[4]))
diceFinal = 0
iofFinal = 0
for m in range(5):
    dicePerLeaf = myDict()
    dicePerPlant = 0
    iofPerLeaf = myDict()
    iofPerPlant = 0
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

        for i in prediction:
            for j in i:
                if j.tolist() not in colorsPrediction :
                    colorsPrediction .append(j.tolist())

        colorsTrue.remove([0,0,0])
        colorsPrediction.remove([0,0,0])
        #print(colorsTrue)
        masks=colorsTrue.copy()
        for i, elem in enumerate(colorsTrue):
            color = np.array(colorsTrue[i])
            masks[i] = cv.inRange(trouePlant, color, color)

        predictions=colorsPrediction .copy()
        for i, elem in enumerate(colorsPrediction ):
            color = np.array(colorsPrediction [i])
            predictions[i] = cv.inRange(prediction, color, color)

        iofPredictions=[]
        dicePredictions=[]
        for j,mask in enumerate(masks):
            ioufori=[]
            dicefori=[]
            
            for prediction in predictions:
                dice = calculateDiceC(prediction,mask)
                dicefori.append(dice)
                ioufori.append(dice/(2-dice))
            #print(str(colorsTrue[z]))
            iofPredictions.append(max(ioufori))
            dicePredictions.append(max(dicefori))
            # oldDice = dicePerLeaf.myGet(str(colorsTrue[j]))
            # oldDice = oldDice + max(dicefori)
            # dicePerLeaf.myAdd(str(colorsTrue[j]), oldDice)
            # oldIof = iofPerLeaf.myGet(str(colorsTrue[j]))
            # oldIof = oldIof + max(ioufori)
            # iofPerLeaf.myAdd(str(colorsTrue[j]), oldIof)

        for i,z in enumerate(colorsTrue):
            dicePerLeaf.myAdd(str(colorsTrue[i]), dicePredictions[i])
            iofPerLeaf.myAdd(str(colorsTrue[i]), iofPredictions[i])
        #print("dice and iof per leaf:")
        #for i, elem in enumerate(iofPredictions):
        #    print(dicePerLeaf[str(colorsTrue[i])], iofPerLeaf[str(colorsTrue[i])])

        diceF = sum(dicePredictions)/len(dicePredictions)
        iofF = sum(iofPredictions)/len(iofPredictions)
        dicePerPlant = dicePerPlant + diceF
        iofPerPlant = iofPerPlant + iofF
        #print("dice for plant: ", diceFinal, " iof for plant: ", iofFinal)
    dicePerPlant = dicePerPlant/180
    iofPerPlant = iofPerPlant/180
    diceFinal = diceFinal + dicePerPlant
    iofFinal = iofFinal + iofPerPlant
    print("plant id: ", m, " dice: ", dicePerPlant, " iof: ", iofPerPlant)
    f.write("plant id: " + str(m) + " dice: " + str(dicePerPlant) + " iof: " + str(iofPerPlant))
    for k in dicePerLeaf.myGetKeys():
        print("leaf: ", k, " dice: ", dicePerLeaf.myGet(k), " iof: ", iofPerLeaf.myGet(k))
        f.write("leaf: " + str(k) + " dice: " + str(dicePerLeaf.myGet(k)) + " iof: " + str(iofPerLeaf.myGet(k)))
diceFinal = diceFinal/5
iofFinal = iofFinal/5
print("for the whole sample: dice: ", diceFinal, " iof: ", iofFinal)
f.write("for the whole sample: dice: " + str(diceFinal) + " iof: " + str(iofFinal))
f.close()