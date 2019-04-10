"""
Assignment 4.2 Neural Networks By Kevin Nijmeijer and Wiebe van Breukelen

The functions in this assignment were checked by hand multiple times to ensure that the mathematics were correct, which they were.
This assignments works properly with the XOR data from the earlier assignment (Code is in the main aswell). 
We've also tried a different activation function (Sigmoid), however this did not improve the results. 
There are cases where this code reaches the 95% validation mark, but the results fluctuate too much over multiple runs.
Multiple attempts were made to check where the issue with calculation could lie, but in the end we could not find the problem.
"""


import math
import random
import numpy as np
import matplotlib.pyplot as plt
from math import e


class Neuron:
    """
    Abstract Data Type for a Neuron.

    """

    def __init__(self, defaultOutput=None):
        """ Construct a new neuron.

        Keyword Arguments:
            defaultOutput {mixed} -- Default output of the neuron. May be used in case of an input layer (default: {None}).
        """

        self.weights = []
        self.inputs = []
        self.isPresetNeuron = (defaultOutput is not None)
        self.output = defaultOutput
        self.error = None

    def SetInputWeights(self, weights: list):
        """ Set the input weights of the neuron.

        Arguments:
            weights {list} -- Weights.
        """

        self.weights = weights

    def AddInputWeight(self, weight: float):
        """ Add a new input with a weight to the neuron.

        Arguments:
            weight {float} -- Input weight.
        """

        self.weights.append(weight)

    def CalculateOutput(self, inputs: list):
        """ Calculate the output of the neuron using the given input and the weights set within the neuron.

        Arguments:
            inputs {list} -- Data input of the neuron.

        Raises:
            Exception -- Exception is throw when the amount of inputs does not match with the amount of weights.
        """

        if self.isPresetNeuron:
            raise ValueError("Unable to calculate output for preset neuron!")

        if len(inputs) != len(self.weights):
            raise Exception(
                "Amount of inputs does not match with the amount of weights within the neuron.")

        self.inputs = inputs
        self.output = 0.0

        for index in range(len(self.inputs)):
            self.output += self.inputs[index] * self.weights[index]

        self.output = math.tanh(self.output)
        # self.output = sigmoid(self.output)

    def __str__(self):
        output = str([round(weight, 4) for weight in self.weights])

        if self.output:
            output += " -> Output: " + str(round(self.output, 4))

        if self.error:
            output += ", Error: " + str(round(self.error, 4))

        return output


class NeuronLayer:
    """ One layer existing out of neurons of a neural network.

    This class can be used to define:
    - An input layer.
    - An hidden layer.
    - A output layer.
    """

    def __init__(self, neuronAmount: int, defaultOutputs=[]):
        """ Construct a new neuron layer.

        Arguments:
            neuronAmount {int} -- Amount of neurons in this layer.

        Keyword Arguments:
            defaultOutputs {list} -- Default output values of the neurons, may be used for an input layer (default: {[]})
        """

        if len(defaultOutputs):
            self.isInputLayer = True
            self.neurons = [Neuron(defaultOutputs[i])
                            for i in range(neuronAmount)]
        else:
            self.isInputLayer = False
            self.neurons = [Neuron()
                            for i in range(neuronAmount)]

    def SetOutputs(self, inputs: list):
        """ Set preset outputs of all neurons. All neurons will be classified as preset neurons, their value should not change anymore.

        Arguments:
            inputs {list} -- Preset outputs as a list.
        """

        if not self.isInputLayer:
            raise ValueError(
                "Cannot invoke SetOutputs(), layer is not a input layer.")

        for i in range(len(inputs)):
            self.neurons[i].output = inputs[i]

    def GetOutput(self) -> list:
        """ Get the output of all neurons within the layer.

        Returns:
            list -- Neuron outputs.
        """
        outputs = []

        for neuron in self.neurons:
            outputs.append(neuron.output)

        return outputs

    def CalculateOutputs(self, inputs: list):
        """ Calculate the new outputs of all neurons within the layer.

        Arguments:
            inputs {list} -- Neuron inputs.
        """

        for neuron in self.neurons:
            if not neuron.isPresetNeuron:
                neuron.CalculateOutput(inputs)

    def __repr__(self) -> str:
        print("Neuron weights: ", end='')
        for neuron in self.neurons:
            print(neuron, end=' ')

        return str()


class NeuralNetwork:
    """
    Object oriented style neural network implementation.

    """

    def __init__(self, learnRate: float, inputs: list, neuronsInHiddenLayers: list, neuronsInOutputLayer: int, bias: float):
        """ Construct a new neural network representation.

        Arguments:
            learnRate {float} -- Learn rate of the network.
            inputs {list} -- Learning input of the network.
            neuronsInHiddenLayers {list} -- Amount of neurons within each hidden layer.
            neuronsInOutputLayer {[type]} -- Amount of neurons in the output layer.
            bias {float} -- Network bias.
        """

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

        # Link final hidden layer and output layer
        self.InitializeWeightsBetweenLayers(
            self.hiddenLayers[len(self.hiddenLayers) - 1], self.outputLayer)

    def FeedForward(self):
        """
        Perform a feed forward operation upon the network.

        """
        # Calculate the first hidden layer
        self.hiddenLayers[0].CalculateOutputs(self.inputLayer.GetOutput())

        # Calculate other hidden layers
        for hiddenLayerIndex in range(1, len(self.hiddenLayers)):
            self.hiddenLayers[hiddenLayerIndex].CalculateOutputs(
                self.hiddenLayers[hiddenLayerIndex - 1].GetOutput())

        # Calculate output layer
        self.outputLayer.CalculateOutputs(
            self.hiddenLayers[len(self.hiddenLayers) - 1].GetOutput())

    def CalculateErrorBetweenLayers(self, currentHiddenLayer: NeuronLayer, previousLayer: NeuronLayer, nextLayer: NeuronLayer):
        """ Calculate the neuron errors between network layers.

        Arguments:
            currentHiddenLayer {NeuronLayer} -- Current layer.
            previousLayer {NeuronLayer} -- Previous linked layer.
            nextLayer {NeuronLayer} -- Next linked layer.
        """

        # for every neuron in the current hidden layer
        for currentNeuronIndex in range(len(currentHiddenLayer.neurons)):
            sumOfErrors = 0.0
            currentNeuron = currentHiddenLayer.neurons[currentNeuronIndex]

            for previousNeuronIndex in range(len(previousLayer.neurons)):
                previousNeuron = previousLayer.neurons[previousNeuronIndex]
                if not previousNeuron.isPresetNeuron:
                    sumOfErrors += previousNeuron.weights[currentNeuronIndex] * \
                        previousNeuron.error

                    previousNeuron.weights[currentNeuronIndex] += self.learnRate * \
                        currentNeuron.output * previousNeuron.error

            currentNeuron.error = (
                1-(math.tanh(currentNeuron.output))) * sumOfErrors
            # currentNeuron.error = derivative_sigmoid(
            #    currentNeuron.output) * sumOfErrors

            if nextLayer.isInputLayer:
                for weightIndex in range(len(nextLayer.neurons)):
                    if not currentNeuron.isPresetNeuron:
                        currentNeuron.weights[weightIndex] += self.learnRate * \
                            nextLayer.neurons[weightIndex].output * \
                            currentNeuron.error

    def BackPropagate(self, realOutput: list):
        """ Perform backpropagation upon the network based on the desired/perfect output.

        Arguments:
            realOutput {list} -- Desired output.
        """

        # Calculate errors for output layer.
        for index in range(len(self.outputLayer.neurons)):
            currentNeuron = self.outputLayer.neurons[index]

            # For each output neuron, we calculate the error by performing the logistic function over the difference in expected output.
            currentNeuron.error = (
                1 - (math.tanh(currentNeuron.output))) * (realOutput[index] - currentNeuron.output)
            # currentNeuron.error = derivative_sigmoid(
            #    currentNeuron.output)*(realOutput[index] - currentNeuron.output)

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

            # Weights are also set in this function.
            self.CalculateErrorBetweenLayers(
                currentHiddenLayer, previousLayer, nextLayer)

    def Train(self, trainingSet: np.array, expectedOutputs: list, iterations: int, validationData, validationOutputs):
        """ Train the network a given number of iterations using a training set and the desired outputs.

        Arguments:
            trainingSet {np.array} -- Numpy array style training set.
            expectedOutputs {list} -- Desired outputs of the network.
            iterations {int} -- Amount of training iterations.
        """
        errorHistogram = []
        correctness = []

        for it in range(iterations):
            nrOfCorrect = 0
            print("--> Iteration {}".format(it))
            for dataIndex in range(len(trainingSet)):
                self.inputLayer.SetOutputs(trainingSet[dataIndex])
                self.FeedForward()
                self.BackPropagate(expectedOutputs[dataIndex])

            MSE = 0.0
            for dataIndex in range(len(validationData)):
                processedOutput = self.ProcessPoint(validationData[dataIndex])

                localError = 0.0
                for outputIndex in range(len(processedOutput)):
                    localError += (validationOutputs[dataIndex]
                                   [outputIndex] - processedOutput[outputIndex])**2

                MSE += localError/len(processedOutput)

                if IsCorrect(processedOutput, validationOutputs[dataIndex]):
                    nrOfCorrect += 1

            correctness.append(nrOfCorrect/len(validationData) * 100)

            MSE = MSE/len(validationData)
            errorHistogram.append(MSE)

        return errorHistogram, correctness

    def InitializeWeightsBetweenLayers(self, layerOne: NeuronLayer, layerTwo: NeuronLayer):
        """ Initialize weights between two network layers.

        Arguments:
            layerOne {NeuronLayer} -- First layer.
            layerTwo {NeuronLayer} -- Second layer.
        """
        weightIndex = 0
        # Initialize random weights from layerTwo to layerOne.
        for layerTwoNeuron in layerTwo.neurons:
            if not layerTwoNeuron.isPresetNeuron:
                for _layerOneNeuron in layerOne.neurons:
                    layerTwoNeuron.AddInputWeight(random.uniform(0.01, 1))

                    weightIndex += 1

    def ProcessPoint(self, input: list):
        """ Perform feed forward operation upon a single data point.

        Arguments:
            input {list} -- Data point.

        Returns:
            list -- Result of feed forward operation.
        """

        self.inputLayer.SetOutputs(input)
        self.FeedForward()

        outputs = []
        for out in self.outputLayer.neurons:
            outputs.append(out.output)

        return outputs

    def __repr__(self):
        print("\nINPUT LAYER:    ", end='')
        print(self.inputLayer)

        for x in range(len(self.hiddenLayers)):
            print("HIDDEN LAYER {}: ".format(x + 1), end='')

            print(self.hiddenLayers[x])

        print("OUTPUT LAYER:   ", end='')
        print(self.outputLayer, end='\n\n')


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


def ConvertLabelsToExpectedOutputs(labels: list):
    """ Convert all labels within the iris dataset to a number format.

    Arguments:
        labels {list} -- Data labels.

    Raises:
        Exception -- When unknown label is found.

    Returns:
        list -- Numeral representation of the outputs.
    """

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


def IsCorrect(testResult, expectedResult):
    """ Check if the solution is correct by using the most vote strategy.

    Arguments:
        testResult {list} -- List of outputs out of the output layer.
        expectedResult {list} -- List of expected outputs.

    Returns:
        bool -- Is correct solution
    """

    iTest = testResult.index(max(testResult))
    iExpected = expectedResult.index(max(expectedResult))

    return iTest == iExpected


def sigmoid(x):
    """Standard sigmoid; since it relies on ** to do computation, it broadcasts on vectors and matrices"""
    return 1 / (1 - (e**(-x)))


def derivative_sigmoid(x):
    """Expects input x to be already sigmoid-ed"""
    return x * (1 - x)


def main():
    learnRate = 0.05
    iterations = 350
    hiddenNeurons = [5, 3]

    #xorData = [[0,0],[0,1],[1,0],[1,1],[1,0],[1,1],[1,1],[0,1],[1,0],[0,0]]

    #xorOutputs = [[0],[1],[1],[0],[1],[0],[0],[1],[1],[0]]

    #xorTestData = [[0,1],[0,0],[0,1],[1,1],[1,0]]
    #xorTestOutput = [[1],[0],[1],[0],[1]]

    outputs = 3
    bias = -1
    validationSetSize = 35

    dataset, labels = ParseIrisDataset("assignment_nn_4_2/irisDataset.csv")

    zippedInput = list(zip(dataset, labels))

    arrayIndex = 0

    splitInput = [[]]

    for index in range(0, len(zippedInput)):
        if index != len(zippedInput) - 1:
            splitInput[arrayIndex].append(zippedInput[index])
            if zippedInput[index][1] != zippedInput[index + 1][1]:

                splitInput.append([])
                arrayIndex += 1
        else:
            splitInput[arrayIndex].append(zippedInput[index])

    validationData = []
    validationLabels = []

    dataset = []
    labels = []

    for irisType in splitInput:
        random.shuffle(irisType)
        valD, valL = zip(*irisType[:validationSetSize])
        d, l = zip(*irisType[validationSetSize:])

        validationData += valD
        validationLabels += valL
        dataset += d
        labels += l

    zippedInput = list(zip(dataset, labels))
    random.shuffle(zippedInput)

    dataset, labels = zip(*zippedInput)
    expectedOutputs = ConvertLabelsToExpectedOutputs(labels)
    expectedValidationOutput = ConvertLabelsToExpectedOutputs(validationLabels)

    nn = NeuralNetwork(learnRate, dataset,
                       hiddenNeurons, outputs, bias)
    errorHistogram, correctness = nn.Train(
        dataset, expectedOutputs, iterations, validationData, expectedValidationOutput)

    # Plot scree.
    plt.subplot(2, 1, 1)
    plt.plot(np.array(range(iterations)), errorHistogram)
    plt.title('MSE')

    plt.xlabel('Iteration')
    plt.ylabel('MSE')

    # Plot second derivative.
    plt.subplot(2, 1, 2)
    plt.title('Correctness')
    plt.xlabel("Iteration")
    plt.ylabel("Correct %")
    plt.plot(np.array(range(iterations)), correctness)

    plt.show()

    """    
    Attempts with a XOR NN training were successfull:

    random.seed(12345)

    nn = NeuralNetwork(learnRate, xorData, hiddenNeurons, 1, bias)

    errorHistogram, correctness = nn.Train(xorData, xorOutputs, iterations, xorTestData, xorTestOutput)

    nn = NeuralNetwork(learnRate, [[1, 1]], [2, 2], 1, -1)

    errorHistogram, correctness = nn.Train([[1, 1]], [[0]], 1, [[1, 1]], [[0]])
    """


# Invoke the main function.
if __name__ == "__main__":
    main()
