import math
import random


class Neuron:
    def __init__(self, defaultOutput=None):
        self.weights = []
        self.inputs = []
        self.isInputNeuron = (defaultOutput is not None)
        self.output = defaultOutput

    def SetInputWeights(self, weights):
        self.weights = weights

    def AddInputWeight(self, weight):
        print("Adding weight {} to neuron...".format(weight))
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
        return "Weights: {}\n Inputs: {}\nOutputs: {}\nIs preset: {}\n".format(
            self.weights, self.inputs, self.output, self.isInputNeuron)


class NeuronLayer:
    def __init__(self, neuronAmount, defaultOutputs=[]):
        if len(defaultOutputs):
            self.neurons = [Neuron(defaultOutputs[i])
                            for i in range(neuronAmount)]
        else:
            self.neurons = [Neuron()
                            for i in range(neuronAmount)]

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
    def __init__(self, inputs, neuronsInHiddenLayer, neuronsInOutputLayer, bias, hiddenLayerWeights=[], outputLayerWeights=[]):
        # self.inputs = inputs
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

    def Backpropagate(self):
        pass

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


def main():
    print("Hello World!")

    nn = NeuralNetwork([[1, 1], [0, 1], [1, 0], [1, 1]],
                       2, 1, -1)

    # nn = NeuralNetwork([[0, 1], [0, 1], [1, 0], [1, 1]],
    #                    2, 1, 1, [1.0, 1.0, 0.0, -1.0, -1.0, 0.0], [1.0, 1.0, 0.0])

    print(nn)

    nn.FeedForward()

    print(nn)

    # nn = NeuralNetwork([[0, 0], [0, 1], [1, 0], [1, 1]],
    #                   2, 1, -1)


# Invoke the main function.
if __name__ == "__main__":
    main()
