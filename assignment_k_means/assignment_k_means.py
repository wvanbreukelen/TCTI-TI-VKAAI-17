import numpy as np
import matplotlib.pyplot as plt
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
    sum = sum**(1/len(pointA))
    return sum

def PickKRandomPointsFromDataset(dataset, k):
    kPoints = []
    for i in range(0,k): #Make this prettier
        kPoints.append(random.choice(dataset))

    return kPoints

def CalculateCentroidsFromClusters(clusters):
    centroids = []
    for cluster in clusters:
        # centroidMean = np.mean(cluster, axis = 0)
        if len(cluster) != 0:
            centroidMean = [0]*len(cluster[0])
            for point in cluster:
                centroidMean = np.add(centroidMean, point)
            for dimension in range(0, len(centroidMean)):
                centroidMean[dimension] = round(centroidMean[dimension] / len(cluster), 1)
                # centroidMean[dimension] = centroidMean[dimension] / len(cluster)
            centroids.append(centroidMean)
        else:
            print("Empty cluster found, assigning new centroid")
            randC = random.choice(clusters)
            while randC == []:
                randC = random.choice(clusters)
                print("Empty cluster picked itself, retry")
            randP = random.choice(randC)
            centroids.append(randP)
    return np.asarray(centroids)

def DetermineLabels(cluster, dataset, labels):
    labelsInCluster = []
    
    
    for point in cluster:
        labelFound = False
        indexOfPointInDataset = 0
        for data in dataset:
            if labelFound == False:
                if all(point == data):
                    labelsInCluster.append(labels[indexOfPointInDataset])
                    labelFound = True
            indexOfPointInDataset += 1


    return labelsInCluster

def CalculateIntraDistance(cluster, centroid):
    intradistance = 0.0

    for point in cluster:
        intradistance += CalculateEuclideanDistance(point, centroid)**2
        

    return intradistance


def main():
    originalData = ParseValuesFromDataset("assignment_k_means\\dataset.csv")
    originalDates = ParseDatesFromDataset("assignment_k_means\\dataset.csv")
    # originalLabels = FormLabels(originalDates)

    # newData = ParseValuesFromDataset("assignment_k_means\\validation1.csv")
    # newDates = ParseDatesFromDataset("assignment_k_means\\validation1.csv")
    # newLabels = FormLabels(newDates)

    attemptsPerK = 1
    k = 1
    kMax = 10
    
    kAxis = []
    vAxis = []

    while k <= kMax:
        # kAxis.append(k)
        # vAxis.append(0)

        # totalCorrectness = 0.0
        for attempts in range(0,attemptsPerK):
            # Pick K random points from dataset as centroids
            centroids = PickKRandomPointsFromDataset(originalData, k)
            clusters = []
            

            for clusterCount in range(0,len(centroids)):
                clusters.append([])

            # print(clusters)
            # While centroids change
            recalculationCount = 0

            centroidsHaveChanged = True
            while centroidsHaveChanged:
                # print(centroids)
                # For each point in dataset, 
                for point in originalData:
                    # assign closest centroid
                    distances = []

                    for centroidIndex in range(0,len(centroids)):
                    # for centroid in centroids
                        distances.append(CalculateEuclideanDistance(point, centroids[centroidIndex]))                 
                    closestCluster = distances.index(min(distances))
                    # Add point to cluster list
                    clusters[closestCluster].append(point)

                # Recalculate centroid
                newCentroids = CalculateCentroidsFromClusters(clusters)
                if np.array_equiv(centroids, newCentroids):
                    centroidsHaveChanged = False
                    # print("Not Changed, done with k", k)
                else:
                    recalculationCount += 1
                    centroids = newCentroids
                    # print("Changed:", recalculationCount)
                    
                    # print("changed")
                
                

            # For Each cluster, 
            V = 0.0
            for clusterindex in range(0,len(clusters)):
                V  += CalculateIntraDistance(clusters[clusterindex], centroids[clusterindex])
                
                # labelsInCluster = DetermineLabels(cluster, originalData, originalLabels)
                # mostOccuringLabel = max(set(labelsInCluster), key=labelsInCluster.count)
                
                # totalCorrectness += (labelsInCluster.count(mostOccuringLabel)/len(labelsInCluster))*100
        
            print("K:", k, "Recalculations:", recalculationCount)
            print("Intradistance :", V)
            kAxis.append(k)
            vAxis.append(V)
        # print("Total Correctness:", totalCorrectness/(k*attemptsPerK))
        
        
        k += 1
    # Determine second derivative of calculated values
    
    print(vAxis)
    print(kAxis)

    plt.plot(kAxis, vAxis)
    plt.show()
    # plt.ylabel
         
    

# Parsing function
if __name__ == "__main__":
    main()