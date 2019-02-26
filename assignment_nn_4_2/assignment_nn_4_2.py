import math
import random


# class Bias:
#     self __init__(self, value, weight):
#         self.weight = weight
#         self.value = value


class Neuron:
    def __init__(self, bias, weights=[], defaultOutput=None):
        self.bias = bias
        self.weights = weights
        self.inputs = []
        self.output = defaultOutput

    def SetInputWeights(self, weights):
        self.weights = weights

    def AddInputWeight(self, weight):
        print("Adding weight {} to neuron...".format(weight))
        self.weights.append(weight)

    def CalculateOutput(self, inputs):
        self.inputs = inputs
        self.output = 0.0

        for index in range(len(self.inputs)):
            self.output += self.inputs[index] * self.weights[index]

        self.output = self.PerformSigmoid(self.output)

        return self.output

    def PerformSigmoid(self, value):
        return 1 - math.tanh(math.tanh(value))

    def IsInputNeuron(self):
        return (len(self.weights) is 0)


class NeuronLayer:
    def __init__(self, neuronAmount, bias, defaultOutputs=[]):
        if len(defaultOutputs):
            self.neurons = [Neuron(bias, defaultOutputs[i])
                            for i in range(neuronAmount)]
        else:
            self.neurons = [Neuron(bias)
                            for i in range(neuronAmount)]

    def IsInputLayer(self):
        for neuron in self.neurons:
            if not neuron.IsInputNeuron():
                return False

        return True


class NeuralNetwork:
    def __init__(self, inputs, neuronsInHiddenLayer, neuronsInOutputLayer, bias, hiddenLayerWeights=[], outputLayerWeights=[]):
        #self.inputs = inputs
        self.inputLayer = NeuronLayer(len(inputs[0]), bias, inputs)
        self.hiddenLayer = NeuronLayer(neuronsInHiddenLayer, bias)
        self.outputLayer = NeuronLayer(neuronsInOutputLayer, bias)

        self.initializeHiddenLayerWeights(hiddenLayerWeights)
        self.initializeOutputLayerWeights(outputLayerWeights)

    def FeedForward(self):
        pass

    def initializeHiddenLayerWeights(self, hiddenLayerWeights=[]):
        weightIndex = 0

        # Initialize the we-> hidden layer neurons.
        for hiddenLayerNeuron in self.hiddenLayer.neurons:
            for inputLayerNeuron in self.inputLayer.neurons:
                if len(hiddenLayerWeights):
                    hiddenLayerNeuron.AddInputWeight(
                        hiddenLayerWeights[weightIndex])
                else:
                    hiddenLayerNeuron.AddInputWeight(random.uniform(0, 1))

                weightIndex += 1

    def initializeOutputLayerWeights(self, outputLayerWeights=[]):
        weightIndex = 0

        # Initialize the we-> hidden layer neurons.
        for outputLayerNeuron in self.outputLayer.neurons:
            for hiddenLayerNeuron in self.hiddenLayer.neurons:
                if len(outputLayerWeights):
                    outputLayerNeuron.AddInputWeight(
                        outputLayerWeights[weightIndex])
                else:
                    outputLayerNeuron.AddInputWeight(random.uniform(0, 1))

                weightIndex += 1


def main():
    print("Hello World!")

    nn = NeuralNetwork([[0, 0], [0, 1], [1, 0], [1, 1]],
                       2, 1, -1, [2.0, 3.0, 4.0, 1.0], [0.9, 0.2])

    # nn = NeuralNetwork([[0, 0], [0, 1], [1, 0], [1, 1]],
    #                   2, 1, -1)


# Invoke the main function.
if __name__ == "__main__":
    main()
