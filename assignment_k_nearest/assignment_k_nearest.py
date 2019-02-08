import numpy as np
import random

# Calculate distance K-means

def ParseValuesFromDataset():
    data = np.genfromtxt("assignment_k_nearest\\dataset.csv", delimiter=";", usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
    return data

def ParseDatesFromDataset():
    dates = np.genfromtxt("assignment_k_nearest\\dataset.csv", delimiter=";", usecols=[0])
    return dates

def FormLabels(dates):
    labels = []
    for label in dates:
        if label < 20000301:
            labels.append("winter")
        elif 20000301 <= label < 20000601:
            labels.append("lente") 
        elif 20000601 <= label < 20000901:
            labels.append("zomer") 
        elif 20000901 <= label < 20001201: 
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

        # print(sum)

    sum = sum/len(pointA)
    return sum
    # print(sum)

def GenerateCentroids(data, k):
    if len(data) <= k:
        return
    centroids = [None]*k
    for n in range(k):
        centroids[n]  = data[random.randint(0, len(data))]
    return centroids

def main():
    data = ParseValuesFromDataset()
    dates = ParseDatesFromDataset()
    labels = FormLabels(dates)

    k = 2

    centroids = GenerateCentroids(data, k)
    
    print(centroids)



# Parsing function
if __name__ == "__main__":
    main()