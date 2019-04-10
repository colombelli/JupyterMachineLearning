import numpy as np
import types  # used for adding the activation function to the neuron class with self parameter



def _softmax(self, inp, maxZ, yExpSum):  # the softmax function which returns probabilities and has a better gradient stepness for very incorrect guesses
    self.output = np.exp(inp - maxZ) / yExpSum
    return self.output


def _ReLU(self, inp, maxZ, yExpSum):
    self.output = max(0, inp)
    return self.output



class Neuron:
    def __init__(self, weights, bias):  # class constructor
        self.weights = weights
        self.bias = bias
        self.output = 0


    def z(self, inputs):  # outputs the sum of every input times its respective weight and add a bias in the final result
        zOut = 0
        for i in range(len(inputs)):
            zOut += inputs[i] * self.weights[i]
        return zOut + self.bias


    def y(self, z, maxZ, yExpSum):  # final output: neuron value when activated
        return self.activation_function(z, yExpSum, maxZ)


    def dEdz(self, dEdy):  # ReLU derivative which respect to z: if z > 0, dydz = 1; else, dydz = 0
        if (self.output > 0):
            return dEdy  # dEdy * dydz = dEdy * 1
        else:
            return 0  # dEdy * dydz = dEdy * 0


    def dEdw(self, inputs, dEdz):  # dEdw = dzdw * dydz * dEdy = dzdw * dEdz     , where dz(i)dw is simply the i input
        dEdw = []
        for i, w in enumerate(self.weights):
            dEdw.append(inputs[i] * dEdz)
        return dEdw


    def updateWeights(self, learningRate, dEdw):
        for i in range(len(self.weights)):
            self.weights[i] -= learningRate * dEdw[i]


    def updateBias(self, learningRate, dEdz):
        self.bias -= learningRate * dEdz  # because dEdb = dzdz * dEdz = 1 * dEdz = dEdz



class DenseLayer:
    def __init__(self, num_of_inputs, num_of_neurons, lastLayer):
        self.num_of_inputs = num_of_inputs
        self.neurons = []
        self.lastLayer = lastLayer  # (true/false)

        for i in range(num_of_neurons):  # creating the neurons for the layer
            weights = np.random.uniform(-0.3, 0.3, [num_of_inputs])
            bias = np.random.uniform(0, 0.3)  # randomizing a bias
            self.neurons.append(Neuron(weights, bias))
            # adding the right activation function method depending on the neuron layer:
            if lastLayer:
                self.neurons[i].activation_function = types.MethodType(
                    _softmax, self.neurons[i])
            else:
                self.neurons[i].activation_function = types.MethodType(
                    _ReLU, self.neurons[i])


    def feedForward(self, inputs):  # activates every neuron in the layer, outputting their respestive results
        self.inputs = inputs  # save the inputs as an attribute in order to use it in the dzdw calculus
        lisOut = []
        zArray = []
        yExpSum = 0
        maxZ = 0
        teste = False
        for neuron in self.neurons:  # calculates z values
            zArray.append(neuron.z(inputs))

        if (self.lastLayer):  # softmax preparation
            # extract the maximum value of the array for computing the softmax as "exp(a-max(a)) / sum(exp(a-max(a))"
            maxZ = max(zArray)
            yExpSum = np.sum(np.exp(np.array(zArray) - maxZ))

        for i, neuron in enumerate(self.neurons):
            lisOut.append(neuron.y(zArray[i], yExpSum, maxZ))
        return lisOut



class NeuralNetwork:
    def __init__(self, num_of_inputs, num_of_neurons_at_each_layer):
        self.num_of_inputs = num_of_inputs
        self.layers = []

        # the first layer will have the given number of inputs for each of its neurons
        num_of_inputs_next_neuron = num_of_inputs
        for idx, num in enumerate(num_of_neurons_at_each_layer):
            # an if test is needed to check if the current layer is the last one
            if (idx + 1) == len(num_of_neurons_at_each_layer):  # then, this is the last layer
                self.layers.append(DenseLayer(
                    num_of_inputs_next_neuron, num, True))
            else:
                self.layers.append(DenseLayer(
                    num_of_inputs_next_neuron, num, False))
                # the next layers will have, as its number of inputs, the number of neurons of the previous layer
                num_of_inputs_next_neuron = num


    def feedForward(self, inputs):
        for layer in self.layers:
            inputs = layer.feedForward(inputs)
        return inputs


    def cross_entropy_cost_function(self, expected_output, network_output):
        cost = 0
        for j in range(len(network_output)):
            # note that if we have a nn that only one output value X is different from zero, then we could just return -log(X), but for reusability let's keep it this way
            cost += expected_output[j] * np.log(network_output[j])
        return -cost


    def backpropagation(self, value, result, learningRate):
        first_dEdz = np.subtract(np.array(result), np.array(value))  # because of softmax and cross entropy cost function

        # update the weights for each neuron in each layer (starting at the end)
        for layer in reversed(self.layers):
            new_dEdy = np.zeros(len(layer.neurons[0].weights))

            for i, neuron in enumerate(layer.neurons):
                if (layer.lastLayer):
                    dEdz = first_dEdz[i]
                else:
                    # necessary to update the bias and for calculating the new dEdy used in the next iterable layer
                    dEdz = neuron.dEdz(dEdy[i])
                dEdw = neuron.dEdw(layer.inputs, dEdz)
                neuron.updateWeights(learningRate, dEdw)
                neuron.updateBias(learningRate, dEdz)

                # sums up the right weight multiplying by the dEdz in the right place of the array (the actual backpropagation value)
                for w in range(len(neuron.weights)):
                    new_dEdy[w] += neuron.weights[w] * dEdz

            # finally, with the new dEdy array constructed, we update it and propagates it to next iterable layer
            dEdy = new_dEdy
