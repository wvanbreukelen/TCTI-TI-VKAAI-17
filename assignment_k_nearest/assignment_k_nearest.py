import numpy as np

# Calculate distance K-means




def main():
    data = np.genfromtxt("assignment_k_nearest\dataset.csv", delimiter=";", usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
    dates = np.genfromtxt("assignment_k_nearest\dataset.csv", delimiter=";", usecols=[0])
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
    print("I did a thing, but who knows what?")

# Parsing function
if __name__ == "__main__":
    print("Hello World!")
    main()