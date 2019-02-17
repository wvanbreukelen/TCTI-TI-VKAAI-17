import numpy as np
import random
import math
from collections import Counter
from operator import itemgetter


def ParseDataset(file, parseLabels=True):
    """ Parse all data points within a given .csv dataset.

    Arguments:
        file {string} -- File path.
        parseLabels {bool} -- Parse labels in the first column (used for validation).

    Returns:
        nparray -- Numpy array containing all data points.
        nparray -- Only when parseLabels is true; numpy array containing all labels.
    """

    data = np.genfromtxt(file, delimiter=";", usecols=[1, 2, 3, 4, 5, 6, 7], converters={
        5: lambda s: 0 if s == b"-1" else float(s),
        7: lambda s: 0 if s == b"-1" else float(s)
    })

    if parseLabels:
        labels = np.genfromtxt(
            file, delimiter=";", usecols=[0])

        return data, labels

    return data


def DateToSeason(date):
    dateWithoutYear = date % 10000

    if dateWithoutYear < 301:
        return "winter"
    elif 301 <= dateWithoutYear < 601:
        return "lente"
    elif 601 <= dateWithoutYear < 901:
        return "zomer"
    elif 901 <= dateWithoutYear < 1201:
        return "herfst"
    else:  # from 01-12 to end of year
        return "winter"


def CalculateEuclideanDistance(pointA, pointB):
    # Check if we are dealing with the same number of dimensions.
    if len(pointA) != len(pointB):
        raise ValueError(
            "Inconsistant number of dimensions. Point A: {}, Point B: {}".format(pointA, pointB))
    sum = 0

    for i in range(len(pointA)):
        sum += math.pow(pointA[i] - pointB[i], 2)

    return math.sqrt(sum)


# def GenerateCentroids(trainingSet, k):
#     size = len(trainingSet)
#     centroids = []

#     for i in range(k):
#         centroids.append(trainingSet[random.randint(0, size - 1)])

#     return centroids

def GenerateClusters(trainingSet, k):
    size = len(trainingSet)
    return [Cluster(trainingSet[random.randint(0, size - 1)]) for i in range(k)]


# def FindNearestToPoint(point, centroids):
#     distance = CalculateEuclideanDistance(
#         point, centroids[0][0])
#     nearest = [0, point]

#     for centroidIndex in range(1, len(centroids)):
#         print(centroids)
#         tempDistance = CalculateEuclideanDistance(
#             point, centroids[centroidIndex][0])
#         if tempDistance < distance:
#             nearest[0] = centroidIndex
#             nearest[1] = point
#             distance = tempDistance

#     return nearest

def FindNearestToPoint(point, centroids):
    distance = CalculateEuclideanDistance(
        point, centroids[0])
    nearest = [0, point]

    for centroidIndex in range(1, len(centroids)):
        tempDistance = CalculateEuclideanDistance(
            point, centroids[centroidIndex])
        if tempDistance < distance:
            nearest[0] = centroidIndex
            nearest[1] = point
            distance = tempDistance

    return nearest


def GetDistanceToCluster(point, cluster):
    return CalculateEuclideanDistance(point, cluster.GetCentroid()[0])


# def UpdateClusters(dataset, clusters, centroids):
#     if len(clusters) == 0:
#         return None

#     # Clear all cluster points.
#     for cluster in clusters:
#         cluster["points"].clear()

#     for point in dataset:
#         nearest = FindNearestToPoint(
#             point[0], centroids)

#         for cluster in clusters:
#             if cluster["centroid_index"] == nearest[0]:
#                 # Append the new cluster point
#                 cluster["points"].append(nearest[1])

#     return clusters

# def UpdateCluster(dataset, cluster, centroids):
#     # Clear all cluster points.
#     cluster["points"].clear()

#     for point in dataset:
#         nearest = FindNearestToPoint(
#             point[0], centroids)

#         if nearest[0] == cluster["centroid_index"]:
#             cluster["points"].append(nearest[1])

#     return cluster


def UpdateClusters(dataset, clusters):
    for point in dataset:
        distanceToClusters = []

        for cluster in clusters:
            distanceToClusters.append(
                [GetDistanceToCluster(point[0], cluster), cluster])

        closestCluster = min(distanceToClusters, key=lambda x: x[0])

        closestCluster[1].AddPoint(point)

    return clusters


def UpdateCentroids(clusters):
    # Calculate the new centroids
    for cluster in clusters:
        if not cluster.RecalculateCentroid():
            return False

    return True


def AnalyseClusters(clusters):
    for cluster in clusters:
        for point in cluster["points"]:
            # print(point)
            pass


class Cluster:
    def __init__(self, centroid, points=[]):
        self.centroid = centroid
        self.points = []

    def AddPoint(self, point):
        self.points.append(point)

    def GetPoints(self):
        return self.points

    def RecalculateCentroid(self):
        if not self.GetSize():
            return False

        self.centroid = np.average(self.points, axis=0)

        return True

    def GetCentroid(self):
        return self.centroid

    def GetSize(self):
        return len(self.points)

    def GetSpread(self):
        labels = {}

        for point in self.points:
            label = DateToSeason(point[1])

            if label not in labels.keys():
                labels[label] = 0

            labels[label] += 1

        return labels


def main():
    # Parse both datasets.
    dataset, datasetLabels = ParseDataset(
        "assignment_k_nearest\\dataset.csv")

    dataset = list(zip(dataset, datasetLabels))

    k = 4

    clusters = GenerateClusters(dataset, k)

    isChanging = True

    while isChanging:
        oldClusters = clusters
        newClusters = UpdateClusters(dataset, clusters)

        if not UpdateCentroids(clusters):
            print("Generating new clusters...", flush=True)
            clusters = GenerateClusters(dataset, k)
            continue

        for oldCluster in oldClusters:
            for newCluster in newClusters:
                if np.array_equiv(oldCluster.GetCentroid(), newCluster.GetCentroid()):
                    isChanging = False
                else:
                    isChanging = True

        if isChanging:
            clusters = newClusters

    for cluster in clusters:
        spread = sorted(cluster.GetSpread().items(),
                        key=itemgetter(1), reverse=True)

        if len(spread):
            print("Spread: {}".format(spread))
            print("Cluster label is {}".format(spread[0][0]))


# Invoke the main function.
if __name__ == "__main__":
    main()
