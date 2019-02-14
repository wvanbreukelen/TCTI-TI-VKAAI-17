import numpy as np
import random
import math
from collections import Counter
from operator import itemgetter

# Calculate distance K-nearest, by Wiebe van Breukelen.


def ParseDataset(file):
    """ Parse all data points within a given .csv dataset.

    Arguments:
        file {string} -- File path.

    Returns:
        nparray -- Numpy array containing all data points.
        nparray -- Numpy array containing all labels.
    """

    data = np.genfromtxt(file, delimiter=";", usecols=[1, 2, 3, 4, 5, 6, 7], converters={
        5: lambda s: 0 if s == b"-1" else float(s),
        7: lambda s: 0 if s == b"-1" else float(s)
    })

    labels = np.genfromtxt(
        file, delimiter=";", usecols=[0])

    return data, labels


def DateToSeason(date):
    """ Convert any given date to a season (in Dutch).

    Arguments:
        date {int} -- Date.

    Returns:
        string -- Season in Dutch.
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
    """ Calculate the enclidean distance between two multidimensional points.

    Note that the amount of dimensions of both points are required to be the same.

    Arguments:
        pointA {nparray} -- First point.
        pointB {nparray} -- Second point.

    Returns:
        float -- The calculated euclidean distance.
    """

    # Check if we are dealing with the same number of dimensions.
    if len(pointA) != len(pointB):
        return None
    sum = 0
    for i in range(len(pointA)):
        sum += math.pow(pointA[i] - pointB[i], 2)

    return math.sqrt(sum)


def GetNeighbours(dataset, labels, referencePoint, k):
    """ Receive the k closest points to a given reference point by calculating the using the enclidean distance.

    Arguments:
        dataset {nparray} -- Numpy array containing the position points.
        labels {nparray} -- Numpy array containing all the labels of the points within the dataset.
        referencePoint {list} -- Reference point.
        k {int} -- Amount of points to return closest to the reference point.

    Returns:
        list -- K closest points to the reference point.
    """

    distances = []

    for index in range(len(dataset)):
        distances.append(
            [CalculateEuclideanDistance(
                referencePoint, dataset[index]), DateToSeason(labels[index])])

    # Sort the points by their distances.
    distances.sort(key=itemgetter(0))

    return distances[:k]


def MostCommonInList(list):
    """ Return the most common value within a list. Behaviour is undefined when using multitype lists.

    Arguments:
        list {list} -- List containing values.

    Returns:
        [mizxed] -- Most common value within the list.
    """

    # Select the most common season within all classified points.
    return Counter(list).most_common()[0][0]


def main():
    # Parse both datasets.
    dataset, datasetLabels = ParseDataset(
        "assignment_k_nearest\\validation1.csv")
    trainingSet, trainingSetLabels = ParseDataset(
        "assignment_k_nearest\\dataset.csv")

    maxK = 63

    # Check every k from 2 to maxK
    for k in range(2, maxK):
        # Iterate through the whole dataset.

        totalSuccess = 0

        for index in range(len(dataset)):
            # print(entry)
            neighbours = GetNeighbours(
                trainingSet, trainingSetLabels, dataset[index], k)

            # print(neighbours)

            labels = [i[1] for i in neighbours]

            result = MostCommonInList(labels)
            # result = get_most_common_label(neighbours)
            if result == DateToSeason(datasetLabels[index]):
                totalSuccess += 1

        print("=============== K: {} ===============\n".format(k))
        print("Success percentage: {}%\n".format(
            (100 * totalSuccess) / len(dataset)))


# Invoke the main function.
if __name__ == "__main__":
    main()
