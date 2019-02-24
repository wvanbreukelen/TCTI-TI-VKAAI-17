import numpy as np
import random
import math
import copy
import matplotlib.pyplot as plt
from collections import Counter
from operator import itemgetter


class Cluster:
    """ Abstract Data Type (ADT) for a K-means cluster. """

    def __init__(self, centroid, points=[]):
        """ Constructs an cluster object.

        Arguments:
            centroid {np.array} -- Cluster start centroid. Should be random.

        Keyword Arguments:
            points {np.array} -- List of points of type numpy array (default: {[]})
        """

        self.centroid = centroid
        self.points = []

    def AddPoint(self, point):
        """ Add a point to the cluster.

        Arguments:
            point {np.array} -- Point to add.
        """

        self.points.append(point)

    def GetPoints(self):
        """ Return all points within the cluster.

        Returns:
            list[np.array] -- List containg numpy arrays.
        """

        return self.points

    def ClearPoints(self):
        """ Clear all points within the cluster. """

        self.points.clear()

    def RecalculateCentroid(self):
        """ Recalculate the cluster centroid by calculating the average point over all cluster points.

        Returns:
            bool -- Returns False if there are not any points within the cluster.
        """

        if not self.GetSize():
            return False

        self.centroid = np.average(self.points, axis=0)

        return True

    def GetCentroid(self):
        """ Get the cluster centroid.

        Returns:
            np.array -- Cluster centroid.
        """

        return self.centroid

    def GetSize(self):
        """ Returns the number of points within the cluster.

        Returns:
            int -- Amount of points within the cluster.
        """

        return len(self.points)

    def GetSpread(self):
        """ Get the label spread within the cluster.

        Returns:
            dict -- Dictionary containing the labels as keys and the amount of occurances as their values.
        """

        labels = {}

        for point in self.points:
            label = DateToSeason(point[1])

            if label not in labels.keys():
                labels[label] = 0

            labels[label] += 1

        return labels


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
    """Parse date to assign season label

    Arguments:
        date {[integer]} -- the date is assigned to a point in the data

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
    """This function calculates the euclidean distance between two points of both n-length in dimensions

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
    """This function is used to start a clusterset of k clusters using the data in the trainingSet

    Arguments:
        trainingSet {npArray} -- The dataset that is used for the generation
        k {integer} -- the target amount of clusters to be created

    Returns:
        Array -- An array of k empty clusters with a random point from the dataset as its centroid
    """

    return [Cluster(trainingSet[random.randint(0, len(trainingSet) - 1)]) for i in range(k)]


def GetDistanceToCluster(point, cluster):
    """This function is used to calculate the distance between a point and the centroid of a given cluster

    Arguments:
        point {numpyArray} -- The point to be used in the distance calculation
        cluster {Cluster} -- the cluster that contains the to be used centroid

    Returns:
        float -- The euclidean distance between the point and the centroid of the cluster
    """

    return CalculateEuclideanDistance(point, cluster.GetCentroid()[0])


def UpdateClusters(dataset, clusters):
    """This function checks all points in the dataset and ads the point to the cluster with the closest centroid

    Arguments:
        dataset {numpyArray} -- The set that contains all points to be assigned to a cluster
        clusters {Cluster[]} -- A list of clusters that get the points from the dataset assigned to

    Returns:
        list [Cluster] -- A new list of clusters containing altered points within said clusters
    """

    [cluster.ClearPoints() for cluster in clusters]

    for point in dataset:
        distanceToClusters = []

        for cluster in clusters:
            distanceToClusters.append(
                [GetDistanceToCluster(point[0], cluster), cluster])

        closestCluster = min(distanceToClusters, key=lambda x: x[0])

        closestCluster[1].AddPoint(point)

    return clusters


def UpdateCentroids(clusters):
    """A helper function to recalculate all centroids within the list of clusters

    Arguments:
        clusters {Cluster[]} -- The list of clusters that need a recalculation of the centroids

    Returns:
        Boolean -- if the recalculation of clusters is succesful it returns True, if there's a problem it returns False
    """

    # Calculate the new centroids
    for cluster in clusters:
        if not cluster.RecalculateCentroid():
            return False

    return True


def CalculateIntraDistance(cluster):
    """ This function calculates the squared Intra-Distance of a given cluster

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


def CalculateSecondDerivative(xValue, xAxisValues, yAxisValues):
    """ This function calculates the second derivative of the point xValue in the dataset of yAxisValues

    Arguments:
        xValue {integer} -- The value on the xAxis whereof the second derivative needs to be calculated
        xAxisValues {list} -- List from first k to maximum k
        yAxisValues {list} -- List of values corresponding to the calculated intradistance per K

    Returns:
        float -- the second differential on point xValue
    """

    dataIndex = xAxisValues.index(xValue)

    firstVal = yAxisValues[dataIndex]
    secondVal = yAxisValues[dataIndex + 1]
    thirdVal = yAxisValues[dataIndex + 2]

    diffA = secondVal - firstVal
    diffB = thirdVal - secondVal

    diffC = diffB - diffA

    return diffC


def PerformKMeans(dataset, iterations, k, intraDistances):
    """ Perform the K-Means algorithm with the given dataset.

    Arguments:
        dataset {list} -- Dataset.
        iterations {int} -- Number of iterations for the given K-value.
        k {int} -- K input for K-means.
        intraDistances {float} -- List reference for storing cluster intra distances.
    """

    intraDistances.append(0.0)
    for run in range(0, iterations):
        print("Iteration", run + 1, "for k =", k)
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
            intraDistanceSum += CalculateIntraDistance(cluster)

        intraDistances[k - 1] += intraDistanceSum

        print("K = {} -> Intra-distance = {}".format(k,
                                                     intraDistanceSum))
    intraDistances[k-1] = intraDistances[k-1]/iterations


def main():
    # K-means config.
    maxK = 10
    iterationsEveryK = 10

    # Parse both datasets.
    dataset, datasetLabels = ParseDataset(
        "assignment_k_means\\dataset.csv")

    dataset = list(zip(dataset, datasetLabels))
    rangeK = range(1, maxK)

    intraDistances = []

    for k in rangeK:
        PerformKMeans(dataset, iterationsEveryK, k, intraDistances)

    optimalK = 0
    optimumFound = False
    usedKs = list(range(1, maxK))

    derivatives = []

    for eachK in range(1, len(usedKs) - 1):
        derivative = CalculateSecondDerivative(eachK, usedKs, intraDistances)
        derivatives.append(derivative)

        # Find the first < 0.
        if derivative < 0:
            if optimumFound == False:
                optimumFound = True
                optimalK = eachK

        # Find the dip.
        if derivative > derivatives[eachK - 2]:
            if optimumFound == False:
                optimumFound = True
                optimalK = eachK - 1

    print("Optimal K: {}".format(optimalK))

    # Plot scree.
    plt.subplot(2, 1, 1)
    plt.plot(np.array(rangeK), intraDistances)
    plt.title('Scree plot')

    plt.xlabel('k')
    plt.ylabel('Intra-distance')

    # Plot second derivative.
    plt.subplot(2, 1, 2)
    plt.title('Second derivative')
    plt.plot(range(1, maxK-2), derivatives)

    plt.show()


# Invoke the main function.
if __name__ == "__main__":
    main()
