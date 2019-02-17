import numpy as np
import random
import math
import matplotlib.pyplot as plt
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


def GenerateClusters(trainingSet, k):
    size = len(trainingSet)
    return [Cluster(trainingSet[random.randint(0, size - 1)]) for i in range(k)]


def GetDistanceToCluster(point, cluster):
    return CalculateEuclideanDistance(point, cluster.GetCentroid()[0])


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


def CalculateIntraDistance(cluster):
    sum = 0.0

    for point in cluster.GetPoints():
        sum += CalculateEuclideanDistance(
            point[0], cluster.GetCentroid()[0])

    return sum / len(cluster.GetPoints())


def main():
    # Parse both datasets.
    dataset, datasetLabels = ParseDataset(
        "assignment_k_means\\dataset.csv")

    dataset = list(zip(dataset, datasetLabels))

    intraDistances = []

    maxK = 10
    rangeK = range(1, maxK)

    for k in rangeK:
        clusters = GenerateClusters(dataset, k)

        isChanging = True

        while isChanging:
            oldClusters = clusters
            newClusters = UpdateClusters(dataset, clusters)

            if UpdateCentroids(clusters):
                for oldCluster in oldClusters:
                    for newCluster in newClusters:
                        if np.array_equiv(oldCluster.GetCentroid(), newCluster.GetCentroid()):
                            isChanging = False
                        else:
                            isChanging = True

                if isChanging:
                    clusters = newClusters
            else:
                # We've got empty clusters, recalculate...
                clusters = GenerateClusters(dataset, k)

        intraDistanceSum = 0.0

        for cluster in clusters:
            spread = sorted(cluster.GetSpread().items(),
                            key=itemgetter(1), reverse=True)

            intraDistanceSum += CalculateIntraDistance(cluster)

            # print("Spread: {}".format(spread))
            # print("Cluster label is {}".format(spread[0][0]))
            # print("Intra distance: {}".format(CalculateIntraDistance(cluster)))

        intraDistances.append(intraDistanceSum / len(clusters))

        print("K = {} -> Intra-distance = {}".format(k,
                                                     intraDistanceSum / len(clusters)))

    largestDerivative = 0.0
    optimalK = 0

    for index in range(k - 3):
        derivative = np.diff(intraDistances[index:index + 3], n=2)

        if derivative > largestDerivative:
            largestDerivative = derivative
        else:
            optimalK = index + 3
            break

    print("Optimal K: {}".format(optimalK))

    plt.subplot(2, 1, 1)
    plt.plot(np.array(rangeK), intraDistances)
    plt.title('Scree plot')

    plt.xlabel('k')
    plt.ylabel('Intra-distance')

    plt.subplot(2, 1, 2)
    plt.title('Second derivative')
    plt.plot(np.diff(intraDistances, n=2))

    plt.show()


# Invoke the main function.
if __name__ == "__main__":
    main()
