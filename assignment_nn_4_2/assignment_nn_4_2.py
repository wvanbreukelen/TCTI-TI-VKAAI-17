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
    def __init__(self, learnRate, inputs, neuronsInHiddenLayer, neuronsInOutputLayer, bias, hiddenLayerWeights=[], outputLayerWeights=[]):
        # self.inputs = inputs
        self.learnRate = learnRate

        self.inputLayer = NeuronLayer(len(inputs[0]), inputs[0])

        # Add the bias neuron
        self.inputLayer.neurons.append(Neuron(bias))

        self.hiddenLayer = NeuronLayer(neuronsInHiddenLayer)

        # Add the bias neuron
        self.hiddenLayer.neurons.append(Neuron(bias))

        self.outputLayer = NeuronLayer(neuronsInOutputLayer)

        self.InitializeWeightsBetweenLayers(
            self.inputLayer, self.hiddenLayer, hiddenLayerWeights)
        self.InitializeWeightsBetweenLayers(
            self.hiddenLayer, self.outputLayer, outputLayerWeights)

    def FeedForward(self):
        self.hiddenLayer.SetOutput(self.inputLayer.GetOutput())
        self.outputLayer.SetOutput(self.hiddenLayer.GetOutput())

    def CalculateErrors(self, realOutput: list):
        # outputLayer
        # Calculate errors for output layer
        for index in range(len(self.outputLayer.neurons)):
            currentNeuron = self.outputLayer.neurons[index]
            currentNeuron.error = (
                1-(math.tanh(currentNeuron.output)))*(realOutput[index]-currentNeuron.output)

        # hiddenLayer
        # Calculate errors for hidden layer
        for index in range(len(self.hiddenLayer.neurons)):
            sumOfErrors = 0.0
            currentNeuron = self.hiddenLayer.neurons[index]

            for outIndex in range(len(self.outputLayer.neurons)):
                currentOutputNeuron = self.outputLayer.neurons[outIndex]
                sumOfErrors += (1 - (math.tanh(currentNeuron.output))) * \
                    currentOutputNeuron.weights[index] * \
                    currentOutputNeuron.error  # TODO Check of dit goed gaat

                currentOutputNeuron.weights[index] += self.learnRate * \
                    currentNeuron.output*currentOutputNeuron.error

            currentNeuron.error = sumOfErrors

            for currentInputIndex in range(len(self.inputLayer.neurons)):
                if not currentNeuron.isInputNeuron:
                    currentNeuron.weights[currentInputIndex] += self.learnRate * \
                        self.inputLayer.neurons[currentInputIndex].output * \
                        currentNeuron.error

        # Set new weights between input and hidden
    def Train(self, trainingSet, expectedOutputs, iterations):
        for it in range(iterations):
            for dataIndex in range(len(trainingSet)):
                self.inputLayer.SetInputs(trainingSet[dataIndex])
                self.FeedForward()
                self.BackPropagate(expectedOutputs[dataIndex])

                if it + 1 == iterations:
                    print("Inputs: {}, Expected Output: {}, Output: {}".format(
                        trainingSet[dataIndex], expectedOutputs[dataIndex], self.outputLayer.neurons[0].output))

    def BackPropagate(self, targetOutputs):
        self.CalculateErrors(targetOutputs)

    def InitializeWeightsBetweenLayers(self, layerOne, layerTwo, hiddenLayerWeights=[]):
        weightIndex = 0
        # Initialize the we-> hidden layer neurons.
        for layerTwoNeuron in layerTwo.neurons:
            if not layerTwoNeuron.isInputNeuron:
                for layerOneNeuron in layerOne.neurons:
                    # Skip neurons with predetermined output. @wvanbreukelen fix naming convention.
                    if len(hiddenLayerWeights) > weightIndex:
                        layerTwoNeuron.AddInputWeight(
                            hiddenLayerWeights[weightIndex])
                    else:
                        layerTwoNeuron.AddInputWeight(random.uniform(0, 1))

                    weightIndex += 1

    def __repr__(self):
        print("INPUT LAYER")
        print(self.inputLayer)

        print("HIDDEN LAYER")
        print(self.hiddenLayer)

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


def main():

    dataset, labels = ParseIrisDataset("assignment_nn_4_2/irisDataset.csv")

    print(dataset)
    print(labels)

   # inputValues = [[0, 0], [0, 1], [1, 0], [1, 1]]
    #inputValues = dataset
    # print(inputValues)
    learnRate = 0.1
    hiddenNeurons = 10
    outputs = 3
    bias = -1
    expectedOutputs = [[0], [1], [1], [0]]
    nn = NeuralNetwork(learnRate, dataset,
                       hiddenNeurons, outputs, bias)

    nn.Train(dataset, labels, 1000)


# Invoke the main function.
if __name__ == "__main__":
    main()
