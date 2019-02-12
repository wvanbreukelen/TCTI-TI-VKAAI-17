import numpy as np
import random

# Calculate distance K-means

def ParseValuesFromDataset(filename):
    data = np.genfromtxt(filename, delimiter=";", usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
    return data

def ParseDatesFromDataset(filename):
    dates = np.genfromtxt(filename, delimiter=";", usecols=[0])
    return dates

def FormLabels(dates):
    labels = []
    for label in dates:
        dateWithoutYear = label%10000
        if dateWithoutYear < 301:
            labels.append("winter")
        elif 301 <= dateWithoutYear < 601:
            labels.append("lente") 
        elif 601 <= dateWithoutYear < 901:
            labels.append("zomer") 
        elif 901 <= dateWithoutYear < 1201: 
            labels.append("herfst")
        else: # from 01-12 to end of year 
            labels.append("winter")
    return labels

def CalculateEuclideanDistance(pointA, pointB):
    if len(pointA) != len(pointB):
        return None
    sum = 0
    for i in range(0, len(pointA)):
        sum += (pointA[i] - pointB[i])**2
    sum = sum/len(pointA)
    return sum

# def DetermineClosestIndices(k, distances):
#     closestIndices = []
#     currentIndex = 0
#     while currentIndex < k:
#         print(currentIndex)
#         indexOfSmallest = distances.index(min(distances))
#         closestIndices.append(indexOfSmallest)
#         del(distances[indexOfSmallest])
#         currentIndex += 1
#     return closestIndices

# def predictLabel(labels):    
#     predictedLabel = max(set(labels), key=labels.count)

#     winters = labels.count("winter")
#     lentes = labels.count("lente")
#     zomers = labels.count("zomer")
#     herfsts = labels.count("herfst")

    # print("Winters:",winters,", Lentes:",lentes, ", Zomers:",zomers, ", Herfsts:", herfsts)
    
    return predictedLabel

def main():
    originalData = ParseValuesFromDataset("assignment_k_means\\dataset.csv")
    originalDates = ParseDatesFromDataset("assignment_k_means\\dataset.csv")
    originalLabels = FormLabels(originalDates)

    newData = ParseValuesFromDataset("assignment_k_means\\validation1.csv")
    newDates = ParseDatesFromDataset("assignment_k_means\\validation1.csv")
    newLabels = FormLabels(newDates)

    # ksForDict = []
    # correctForDict = []

    # k = 1
    # while k <= 100:

    #     # ksForDict.append(k)
    #     currentPointIndex = 0
    #     correct = 0
    #     for newPoint in newData:
    #         labels = []
    #         distances = []
    #         for oldPoint in originalData:
    #             distances.append(CalculateEuclideanDistance(newPoint, oldPoint))
    #         closestIndices = DetermineClosestIndices(k, distances)
    #         for index in closestIndices:
    #             labels.append(originalLabels[index])
    #         predictedLabel = predictLabel(labels)
    #         realLabel = newLabels[currentPointIndex]
            

    #         if predictedLabel == realLabel:
    #             correct += 1
    #         # else:
    #         #     print("Predicted:",predictedLabel,"||| Real:",realLabel)

    #         currentPointIndex += 1
    #     # correctForDict.append(correct)
    #     print("K =", k, "- correct answers:", correct)
    #     k += 1

    
    # print(dict(zip(ksForDict,correctForDict)))
    

# Parsing function
if __name__ == "__main__":
    main()