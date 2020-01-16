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
        for i, elem in enumerate(colorsPrediction ):
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
        for i, z in enumerate(colorsTrue):
            # if(dicePerLeaf.myLen()==0):
            #     continue
            oldDice = dicePerLeaf.myGet(str(colorsTrue[i]))
            oldIof = iofPerLeaf.myGet(str(colorsTrue[i]))
            addDice = 0
            addIof = 0
            if(i<len(dicePredictions)):
                addDice = dicePredictions[i]
                addIof = iofPredictions[i]
            newDice = oldDice + addDice
            newIof = oldIof + addIof
            dicePerLeaf.myAdd(str(colorsTrue[i]), newDice)
            iofPerLeaf.myAdd(str(colorsTrue[i]), newIof)
        print(len(colorsTrue))
        for dl in dicePerLeaf:
            print(dicePerLeaf.myGet(dl))
            dicePerLeaf.myAdd(dl, dicePerLeaf.myGet(dl)/len(colorsTrue))
            iofPerLeaf.myAdd(dl, iofPerLeaf.myGet(dl)/len(colorsTrue))
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
    print("plant id: ", m, " dice: ", dicePerPlant, " iof: ", iofPerPlant)
    f.write("plant id: " + str(m) + " dice: " + str(dicePerPlant) + " iof: " + str(iofPerPlant) + "\n")
    for k in dicePerLeaf.myGetKeys():
        print("leaf: ", k, " dice: ", dicePerLeaf.myGet(k), " iof: ", iofPerLeaf.myGet(k))
        f.write("leaf: " + str(k) + " dice: " + str(dicePerLeaf.myGet(k)) + " iof: " + str(iofPerLeaf.myGet(k)) + "\n")
diceFinal = diceFinal/5
iofFinal = iofFinal/5
print("for the whole sample: dice: ", diceFinal, " iof: ", iofFinal)
f.write("for the whole sample: dice: " + str(diceFinal) + " iof: " + str(iofFinal) + "\n")
f.close()
