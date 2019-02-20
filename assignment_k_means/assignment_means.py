import numpy as np
import random
import math
import copy
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
    """ Parse date to assign season label

    Arguments:
        date {integer} -- the date is assigned to a point in the data

    Returns:
        String -- The corresponding season string
    """

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
    """ This function calculates the euclidean distance between two points of both n-length in dimensions

    Arguments:
        pointA {numpyArray} -- This is the source point that is used in the calculation
        pointB {numpyArray} -- This is the target point that is used in the calculation

    Raises:
        ValueError -- This error is thrown when the number of dimensions between pointA and pointB differ

    Returns:
        Float -- the calculated euclidean distance
    """

    # Check if we are dealing with the same number of dimensions.
    if len(pointA) != len(pointB):
        raise ValueError(
            "Inconsistant number of dimensions. Point A: {}, Point B: {}".format(pointA, pointB))
    sum = 0

    for i in range(len(pointA)):
        sum += math.pow(pointA[i] - pointB[i], 2)

    return math.sqrt(sum)


def GenerateClusters(trainingSet, k):
    """ This function is used to start a clusterset of k clusters using the data in the trainingSet

    Arguments:
        trainingSet {npArray} -- The dataset that is used for the generation
        k {integer} -- the target amount of clusters to be created

    Returns:
        Array -- An array of k empty clusters with a random point from the dataset as its centroid
    """

    return [Cluster(trainingSet[random.randint(0, len(trainingSet) - 1)]) for i in range(k)]


def GetDistanceToCluster(point, cluster):
    """ This function is used to calculate the distance between a point and the centroid of a given cluster

    Arguments:
        point {numpyArray} -- The point to be used in the distance calculation
        cluster {Cluster} -- the cluster that contains the to be used centroid

    Returns:
        float -- The euclidean distance between the point and the centroid of the cluster
    """

    return CalculateEuclideanDistance(point, cluster.GetCentroid()[0])


def UpdateClusters(dataset, clusters):
    """ This function checks all points in the dataset and ads the point to the cluster with the closest centroid

    Arguments:
        dataset {numpyArray} -- The set that contains all points to be assigned to a cluster
        clusters {Cluster} -- A list of clusters that get the points from the dataset assigned to

    Returns:
        list [Cluster] -- A new list of clusters containing altered points within said clusters
    """

    [cluster.ResetPoints() for cluster in clusters]

    for point in dataset:
        distanceToClusters = []

        for cluster in clusters:
            distanceToClusters.append(
                [GetDistanceToCluster(point[0], cluster), cluster])

        closestCluster = min(distanceToClusters, key=lambda x: x[0])

        closestCluster[1].AddPoint(point)

    # clusters = UpdateCentroids(clusters)

    return clusters


def UpdateCentroids(clusters):
    """ A helper function to recalculate all centroids within the list of clusters

    Arguments:
        clusters {Cluster} -- The list of clusters that need a recalculation of the centroids

    Returns:
        Boolean -- if the recalculation of clusters is succesful it returns True, if there's a problem it returns False
    """

    # Calculate the new centroids
    for cluster in clusters:
        if not cluster.RecalculateCentroid():
            return False

    return True


""" Abstract Data Type for a cluster.

A cluster is build out of cluster points. These cluster points determines the cluster centroid, the mean of all cluster points. 
"""


class Cluster:

    """ Constructs a cluster object.

    Arguments:
        centroid {np.array} -- Cluster start centroid, most likely random selected out of the dataset.
        points {[np.array]} -- Optional: points in the cluster.
    """

    def __init__(self, centroid, points=[]):
        self.centroid = centroid
        self.points = []

    """ Add a point to the cluster.
    
    Argument:
        point {np.array} -- Point to add.
    """

    def AddPoint(self, point):
        self.points.append(point)

    """ Get all points within the cluster.
    
    Returns:
        {[np.array]} -- Points within cluster.
    """

    def GetPoints(self):
        return self.points

    def ResetPoints(self):
        self.points.clear()

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
    """This function calculates the squared Intra-Distance of a given cluster

    Arguments:
        cluster {Cluster} -- the cluster of points and the centroid to calculate the intraDistance

    Returns:
        float -- the sum of the squared euclidean distance between each point in the cluster and the clusters centroid
    """

    sum = 0.0

    for point in cluster.GetPoints():
        sum += CalculateEuclideanDistance(
            point[0], cluster.GetCentroid()[0])**2

    return sum


def calculateSecondDerivative(xValue, xAxisValues, yAxisValues):

    dataIndex = xAxisValues.index(xValue)

    firstVal = yAxisValues[dataIndex]
    secondVal = yAxisValues[dataIndex + 1]
    thirdVal = yAxisValues[dataIndex + 2]

    diffA = secondVal-firstVal
    diffB = thirdVal-secondVal

    diffC = diffB-diffA

    return diffC


def main():
    # Parse both datasets.
    dataset, datasetLabels = ParseDataset(
        "assignment_k_means\\dataset.csv")


def PerformKMeans(iterations, dataset, k):
    kAccumDistances = 0.0

    for z in range(iterations):
        clusters = GenerateClusters(dataset, k)

        isChanging = True

        while isChanging:
            oldClusters = clusters
            newClusters = copy.deepcopy(
                UpdateClusters(dataset, oldClusters))

            if UpdateCentroids(newClusters):
                flags = []

                for index in range(len(oldClusters)):
                    if np.array_equiv(np.array(np.around(oldClusters[index].GetCentroid()[0], 1)), np.array(np.around(newClusters[index].GetCentroid()[0], 1))):
                        flags.append(False)
                    else:
                        flags.append(True)

                if True in flags:
                    clusters = copy.deepcopy(newClusters)
                else:
                    isChanging = False
            else:
                # We've got empty clusters, recalculate...
                clusters = GenerateClusters(dataset, k)

        intraDistanceSum = 0.0

        for cluster in clusters:
            # spread = sorted(cluster.GetSpread().items(),
            #                 key=itemgetter(1), reverse=True)

            intraDistanceSum += CalculateIntraDistance(cluster)

            # print("Spread: {}".format(spread))
            # print("Cluster label is {}".format(spread[0][0]))
            # print("Intra distance: {}".format(CalculateIntraDistance(cluster)))
        kAccumDistances += intraDistanceSum

    #intraDistances.append(intraDistanceSum / iterations)

    return kAccumDistances / iterations


def main():
    # Parse both datasets.
    dataset, datasetLabels = ParseDataset(
        "assignment_k_means\\dataset.csv")

    dataset = list(zip(dataset, datasetLabels))

    intraDistances = []

    maxK = 10
    rangeK = range(1, maxK)
    iterations = 5

    for k in rangeK:
        intraDistances.append(PerformKMeans(iterations, dataset, k))

        print("K = {} -> Intra-distance = {}".format(k,
                                                     intraDistances[-1]))

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
    plt.xlabel('k')
    plt.plot(range(3, maxK), np.diff(intraDistances, n=2))

    plt.show()


# Invoke the main function.
if __name__ == "__main__":
    main()
