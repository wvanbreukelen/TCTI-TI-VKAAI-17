import math
import random
import numpy as np


class Neuron:
    def __init__(self, defaultOutput=None):
        self.weights = []
        self.inputs = []
        self.isInputNeuron = (defaultOutput is not None)
        self.output = defaultOutput
        self.error = None

    def SetInputWeights(self, weights):
        self.weights = weights

    def AddInputWeight(self, weight):
        self.weights.append(weight)

    def CalculateOutput(self, inputs):
        if self.isInputNeuron:
            return self.output

        self.inputs = inputs
        self.output = 0.0

        for index in range(len(self.inputs)):
            self.output += self.inputs[index] * self.weights[index]

        self.output = math.tanh(self.output)

    def __str__(self):
        return "Input weights: {}\nInput values: {}\nOutput: {}\nIs preset: {}\nError: {}\n".format(
            self.weights, self.inputs, self.output, self.isInputNeuron, self.error)


class NeuronLayer:
    def __init__(self, neuronAmount, defaultOutputs=[]):
        if len(defaultOutputs):
            self.neurons = [Neuron(defaultOutputs[i])
                            for i in range(neuronAmount)]
        else:
            self.neurons = [Neuron()
                            for i in range(neuronAmount)]

    def SetInputs(self, inputs):
        # if len(inputs) != len(self.neurons):
        #     raise Exception("Amount of inputs does not match with the amount of neurons!")

        for i in range(len(inputs)):
            self.neurons[i].output = inputs[i]
            self.neurons[i].isInputNeuron = True

    def GetOutput(self):
        result = []

        for neuron in self.neurons:
            result.append(neuron.output)

        return result

    def SetOutput(self, inputs):
        for neuron in self.neurons:
            neuron.CalculateOutput(inputs)

    def IsInputLayer(self):
        for neuron in self.neurons:
            if not neuron.isInputNeuron:
                return False
        return True

    def __repr__(self):
        print("Is input layer: {}\n".format(self.IsInputLayer()))

        for neuron in self.neurons:
            print(neuron)

        return str()


class NeuralNetwork:
    def __init__(self, learnRate, inputs, neuronsInHiddenLayers: list, neuronsInOutputLayer, bias):
        # self.inputs = inputs
        self.learnRate = learnRate
        self.inputLayer = NeuronLayer(len(inputs[0]), inputs[0])

        # Add the bias neuron
        self.inputLayer.neurons.append(Neuron(bias))

        self.hiddenLayers = []

        for neuronsInHiddenLayer in neuronsInHiddenLayers:
            hiddenLayer = NeuronLayer(neuronsInHiddenLayer)
            # Add the bias neuron
            hiddenLayer.neurons.append(Neuron(bias))

            self.hiddenLayers.append(hiddenLayer)

        self.outputLayer = NeuronLayer(neuronsInOutputLayer)

        # Link input and first hidden layer
        self.InitializeWeightsBetweenLayers(
            self.inputLayer, self.hiddenLayers[0])

        for hiddenLayerIndex in range(len(self.hiddenLayers) - 1):
            self.InitializeWeightsBetweenLayers(
                self.hiddenLayers[hiddenLayerIndex], self.hiddenLayers[hiddenLayerIndex + 1])

        # link final hidden layer and output layer
        self.InitializeWeightsBetweenLayers(
            self.hiddenLayers[len(self.hiddenLayers) - 1], self.outputLayer)

    def FeedForward(self):
        self.hiddenLayers[0].SetOutput(self.inputLayer.GetOutput())

        for hiddenLayerIndex in range(1, len(self.hiddenLayers)):
            self.hiddenLayers[hiddenLayerIndex].SetOutput(
                self.hiddenLayers[hiddenLayerIndex - 1].GetOutput())

        self.outputLayer.SetOutput(
            self.hiddenLayers[len(self.hiddenLayers) - 1].GetOutput())

    def CalculateErrorBetweenLayers(self, currentHiddenLayer, previousLayer, nextLayer):
        for currentNeuronIndex in range(len(currentHiddenLayer.neurons)):
            sumOfErrors = 0.0
            currentNeuron = currentHiddenLayer.neurons[currentNeuronIndex]

            for previousNeuronIndex in range(len(previousLayer.neurons)):
                previousNeuron = previousLayer.neurons[previousNeuronIndex]
                if not previousNeuron.isInputNeuron:
                    sumOfErrors += (1 - (math.tanh(currentNeuron.output))) * \
                        previousNeuron.weights[currentNeuronIndex] * \
                        previousNeuron.error

                    previousNeuron.weights[currentNeuronIndex] += self.learnRate * \
                        currentNeuron.output * previousNeuron.error

            currentNeuron.error = sumOfErrors

            for weightIndex in range(len(nextLayer.neurons)):
                if not currentNeuron.isInputNeuron:
                    currentNeuron.weights[weightIndex] += self.learnRate * \
                        nextLayer.neurons[weightIndex].output * \
                        currentNeuron.error

    def CalculateErrors(self, realOutput: list):
        # Calculate errors for output layer.
        for index in range(len(self.outputLayer.neurons)):
            currentNeuron = self.outputLayer.neurons[index]

            # For each output neuron, we calculate the error by performing the logistic function over the difference in expected output.
            currentNeuron.error = (
                1 - (math.tanh(currentNeuron.output)))*(realOutput[index] - currentNeuron.output)

        # Calculate errors for the hidden layers and the input layer.
        for hiddenLayerIndex in range(len(self.hiddenLayers) - 1, -1, -1):
            previousLayer = []
            currentHiddenLayer = self.hiddenLayers[hiddenLayerIndex]

            if hiddenLayerIndex == len(self.hiddenLayers) - 1:
                previousLayer = self.outputLayer
            else:
                previousLayer = self.hiddenLayers[hiddenLayerIndex + 1]

            if hiddenLayerIndex > 0:
                nextLayer = self.hiddenLayers[hiddenLayerIndex - 1]
            else:
                nextLayer = self.inputLayer

            self.CalculateErrorBetweenLayers(
                currentHiddenLayer, previousLayer, nextLayer)

        # Set new weights between input and hidden

    def Train(self, trainingSet, expectedOutputs, iterations):
        for it in range(iterations):
            print("Iteration {}".format(it))
            for dataIndex in range(len(trainingSet)):
                self.inputLayer.SetInputs(trainingSet[dataIndex])
                self.FeedForward()
                self.BackPropagate(expectedOutputs[dataIndex])

    def BackPropagate(self, targetOutputs):
        self.CalculateErrors(targetOutputs)

    def InitializeWeightsBetweenLayers(self, layerOne, layerTwo):
        weightIndex = 0
        # Initialize random weights from layerTwo to layerOne.
        for layerTwoNeuron in layerTwo.neurons:
            if not layerTwoNeuron.isInputNeuron:
                for layerOneNeuron in layerOne.neurons:
                    layerTwoNeuron.AddInputWeight(random.uniform(0, 1))

                    weightIndex += 1

    def ProcessPoint(self, input):
        self.inputLayer.SetInputs(input)
        self.FeedForward()

        outputs = []
        for out in self.outputLayer.neurons:
            outputs.append(out.output)

        return outputs

    def __repr__(self):
        print("INPUT LAYER")
        print(self.inputLayer)

        print("HIDDEN LAYER")
        print(self.hiddenLayers)

        print("OUTPUT LAYER")
        print(self.outputLayer)

        return str()


def ParseIrisDataset(file, parseLabels=True):
    """ Parse all data points within a given .csv dataset.

    Arguments:
        file {string} -- File path.
        parseLabels {bool} -- Parse labels in the first column (used for validation).

    Returns:
        nparray -- Numpy array containing all data points.
        nparray -- Only when parseLabels is true; numpy array containing all labels.
    """

    data = np.genfromtxt(file, delimiter=",", usecols=[
        0, 1, 2, 3], dtype=float)

    if parseLabels:
        labels = np.genfromtxt(
            file, delimiter=",", usecols=[4], dtype=str)

        return data, labels

    return data


def ConvertLabelsToExpectedOutputs(labels):
    expectedOutputs = []

    for label in labels:
        if label == "Iris-setosa":
            expectedOutputs.append([1.0, 0.0, 0.0])
        elif label == "Iris-versicolor":
            expectedOutputs.append([0.0, 1.0, 0.0])
        elif label == "Iris-virginica":
            expectedOutputs.append([0.0, 0.0, 1.0])
        else:
            raise Exception("Unknown label in dataset: {}.".format(label))

    return expectedOutputs


def main():

    learnRate = 0.1
    iterations = 100
    hiddenNeurons = [15, 15]
    #hiddenNeurons = [50]
    outputs = 3
    bias = -1
    testsetSize = 15

    dataset, labels = ParseIrisDataset("assignment_nn_4_2/irisDataset.csv")

    zippedInput = list(zip(dataset, labels))
    random.shuffle(zippedInput)
    testDataset, testLabels = zip(*zippedInput[::testsetSize])
    del zippedInput[:testsetSize]
    dataset, labels = zip(*zippedInput)
    expectedOutputs = ConvertLabelsToExpectedOutputs(labels)

    nn = NeuralNetwork(learnRate, dataset,
                       hiddenNeurons, outputs, bias)

    nn.Train(dataset, expectedOutputs, iterations)

    for testIndex in range(len(testDataset)):
        testResult = nn.ProcessPoint(testDataset[testIndex])
        print("Iris Setosa:\t\t{}\nIris Versicolor:\t{}\nIris Virginica:\t\t{}".format(
            testResult[0], testResult[1], testResult[2]))
        print("Actual label:\t\t{}".format(testLabels[testIndex]))
        print('\n')


# Invoke the main function.
if __name__ == "__main__":
    main()
