class Bias:
    self __init__(self, value, weight):
        pass


class Neuron:
    def __init__(self, bias, weights=[]):
        self.bias = bias
        self.weights = weights
        self.outputValue = None

    def CalculateOutput(self, input):
        sum = 0.0

        for weight in weights:
            sum += input * weight

        sum += bias

        # Sum is a value not between 0.0 and 1.0

        return sum


class NeuronLayer:
    def __init__(self, neuronAmount, bias):
        self.neurons = [Neuron(bias)
                        for i in range(neuronAmount)]

        for i in range(neuronAmount):


class NeuralNetwork:
    def __init__(self, inputs, neuronsInHiddenLayer, neuronsInOutputLayer, bias):
        self.inputs = inputs
        self.hiddenLayer = NeuronLayer(neuronsInHiddenLayer, bias)
        self.outputLayer = NeuronLayer(neuronsInOutputLayer, bias)

    def initializeHiddenWeights():
        # Initialize the we-> hidden layer neurons.
        pass


def main():
    print("Hello World!")

    nn = NeuralNetwork([[0, 0], [0, 1], [1, 0], [1, 1]], 2, 1, -1)


# Invoke the main function.
if __name__ == "__main__":
    main()
