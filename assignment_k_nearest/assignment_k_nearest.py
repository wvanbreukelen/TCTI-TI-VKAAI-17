import numpy as np
import random
import math
from collections import Counter
from operator import itemgetter

# Calculate distance K-nearest, by Wiebe van Breukelen and Kevin Nijmeijer.


def ParseDataset(file, isValidationSet):
    data = np.genfromtxt(file, delimiter=";", usecols=[1, 2, 3, 4, 5, 6, 7], converters={
                         5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
    dates = np.genfromtxt(
        file, delimiter=";", usecols=[0])

    labels = []

    if isValidationSet:
        for label in dates:
            if label < 20010301:
                labels.append("winter")
            elif 20010301 <= label < 20010601:
                labels.append("lente")
            elif 20010601 <= label < 20010901:
                labels.append("zomer")
            elif 20010901 <= label < 20011201:
                labels.append("herfst")
            else:  # from 01-12 to end of year
                labels.append("winter")
    else:
        for label in dates:
            if label < 20000301:
                labels.append("winter")
            elif 20000301 <= label < 20000601:
                labels.append("lente")
            elif 20000601 <= label < 20000901:
                labels.append("zomer")
            elif 20000901 <= label < 20001201:
                labels.append("herfst")
            else:  # from 01-12 to end of year
                labels.append("winter")

    return {'data': data, 'labels': labels}


def CalculateEuclideanDistance(pointA, pointB):
    # Check if we are dealing with the same number of dimensions.
    if len(pointA) != len(pointB):
        return None
    sum = 0
    for i in range(len(pointA)):
        sum += math.pow(pointA[i] - pointB[i], 2)

    return math.sqrt(sum)


def GetNeighbours(trainingSet, pointSearch, k):
    distances = []

    for index in range(len(trainingSet['data'])):
        distances.append(
            [index, CalculateEuclideanDistance(
                trainingSet['data'][index], pointSearch)])

    # Sort the points by their distances.
    distances.sort(key=itemgetter(1))

    result = []
    for x in range(k):
        result.append(distances[x])

    return result


def GetMostCommon(classifiedPoints):
    # Select the most common season within all classified points.
    return Counter(classifiedPoints).most_common()[0][0]


def main():
    # Parse both datasets.
    dataset = ParseDataset("assignment_k_nearest\\dataset.csv", False)
    trainingSet = ParseDataset("assignment_k_nearest\\validation1.csv", True)

    totalSuccess = 0
    totalFail = 0

    # Check every k from 2 to 63
    for k in range(2, 63):
        # Iterate through the whole dataset.
        for index in range(len(dataset['data'])):
            neighbours = GetNeighbours(trainingSet, dataset['data'][index], k)

            classifiedPoints = []

            # For each shortest distance, get the matching season out of the set.
            for neighbour in neighbours:
                classifiedPoints.append(trainingSet['labels'][neighbour[0]])

            result = GetMostCommon(classifiedPoints)

            if result == dataset['labels'][index]:
                # We guessed it right, the season is correct.
                totalSuccess += 1
            else:
                # We failed.
                totalFail += 1

        print("=============== K: {} ===============\n".format(k))

        print("Error percentage: {}%".format(
            (100 * totalFail) / (totalSuccess + totalFail)))

        print("Success percentage: {}%\n".format(
            (100 * totalSuccess) / (totalSuccess + totalFail)))


# Invoke the main function.
if __name__ == "__main__":
    main()

# def GenerateCentroids(data, k):
#     if len(data) <= k:
#         return
#     centroids = []
#     for n in range(k):
#         centroids.append(data[random.randint(0, len(data))])
#     return centroids
