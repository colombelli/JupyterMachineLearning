import numpy as np


class Neuron:
    def __init__(self, weights, bias):  # class constructor
        self.weights = weights
        self.bias = bias
        self.output = 0

    def activation_function(self, input, lastLayer, yExpSum=None, maxZ=None):
        if lastLayer:  # the softmax funtion is used
            # the softmax function returns probabilities and has a better gradient stepness for very incorrect outputs
            self.output = np.exp(input - maxZ) / yExpSum
        else:  # we use the ReLU for non-last-layers in order to reach conversion faster
            self.output = max(0, input)
        return self.output

    def z(self, inputs):  # outputs the sum of every input times its respective weight
        zOut = 0
        for i in range(len(inputs)):
            zOut += inputs[i] * self.weights[i]
        return zOut + self.bias

    # final output: neuron value when activated
    def y(self, inputs, lastLayer, yExpSum=None, maxZ=None):
        return self.activation_function(self.z(inputs), lastLayer, yExpSum, maxZ)


    def dEdz(self, dEdy):
        # ReLU derivative (if z > 0, dydz = 1; else, dydz = 0)
        if (self.output > 0):
            return dEdy  # dEdy * dydz = dEdy * 1
        else:
            return 0  # dEdy * dydz = dEdy * 0


    # dEdw = dzdw * dydz * dEdy = dzdw * dEdz     , where dz(i)dw is simply the i input
    def dEdw(self, dEdy, inputs, lastLayer, dEdz=None):
        if not lastLayer:  # if it was the last layer, then the dEdz must have been passed as a parameter
            dEdz = self.dEdz(dEdy)
        dEdw = []
        for i, w in enumerate(self.weights):
            dEdw.append(inputs[i] * dEdz)
        return dEdw


    # we have to update the weights and biases based on the learning rate and the derivatives
    def updateWeights(self, learningRate, dEdw):
        for i in range(len(self.weights)):
            self.weights[i] -= learningRate * dEdw[i]


    def updateBias(self, learningRate, dEdz):
        self.bias -= learningRate * dEdz  # dEdb = dzdz * dEdz = 1 * dEdz = dEdz



class DenseLayer:
    def __init__(self, num_of_inputs, num_of_neurons, lastLayer):
        self.num_of_inputs = num_of_inputs
        self.neurons = []
        self.lastLayer = lastLayer  # a variable indicating if this layer is the last one (true/false)

        for i in range(num_of_neurons):  # creating the neurons for the layer
            # randomizing "num_inputs" weights with a value between -0.3 and 0.3
            weights = np.random.uniform(-0.3, 0.3, [num_of_inputs])
            bias = np.random.uniform(0, 0.3)  # randomizing a bias
            self.neurons.append(Neuron(weights, bias))


    # activates every neuron in the layer, outputting their respestive results
    def feedForward(self, inputs):
        # save the inputs as an attribute in order to use it in the dzdw calculus
        self.inputs = inputs
        lisOut = []
        yExpSum = 0
        maxZ = 0
        if (self.lastLayer):  # softmax preparation
            zArray = []
            # we can't simply sum everything up; first, we need to get the array of z outputs and then do a trick to avoid nan/inf values
            for neuron in self.neurons:
                zArray.append(neuron.z(inputs))
            # we extract the maximum value of the array for computing the softmax as "exp(a-max(a)) / sum(exp(a-max(a))"
            maxZ = max(zArray)
            for z in zArray:  # sum everything with the above introduced subtraction
                yExpSum += np.exp(z - maxZ)

        for neuron in self.neurons:
            lisOut.append(neuron.y(inputs, self.lastLayer, yExpSum, maxZ))
        return lisOut



class NeuralNetwork:
    def __init__(self, num_of_inputs, num_of_neurons_at_each_layer):
        self.num_of_inputs = num_of_inputs
        self.layers = []

        # creates a dense layer for every int on the list num_of_neurons_at_each_layer
        # the first layer will have the given number of inputs for each of its neuron
        num_of_inputs_next_neuron = num_of_inputs
        for idx, num in enumerate(num_of_neurons_at_each_layer):
            # an if test is needed to check if the current layer is the last one
            if (idx + 1) == len(num_of_neurons_at_each_layer):  # means that this is the last layer
                self.layers.append(DenseLayer(
                    num_of_inputs_next_neuron, num, True))
            else:
                self.layers.append(DenseLayer(
                    num_of_inputs_next_neuron, num, False))
                # the next layers will have the number of neurons of the previous layer as its number of inputs
                num_of_inputs_next_neuron = num


    def feedForward(self, inputs):
        for layer in self.layers:  # keeps picking the outputs of every layer and passing them as inputs to the next layer
            inputs = layer.feedForward(inputs)
        # the outputs will be the final result of the inputs variable, after the end of the loop above
        return inputs


    def cross_entropy_cost_function(self, expected_output, network_output):
        cost = 0
        for i in range(len(network_output)):
            # note that if we have a nn that only one output value X is different from zero, then we could just return -log(X), but for reusability let's keep it this way
            cost += expected_output[i] * np.log(network_output[i])
        return -cost


    def backpropagation(self, value, result, learningRate):
        dEdy = np.zeros(len(value))  # initializes dEdy
        first_dEdz = np.subtract(np.array(result), np.array(value))  # easy to calculate because of cross entropy with softmax

        # update the weights for each neuron in each layer (starting at the end)
        for layer in reversed(self.layers):
            new_dEdy = np.zeros(len(layer.neurons[0].weights))

            for i, neuron in enumerate(layer.neurons):  # updates the neurons weights and biases in the layer
                # first we calculate the dEdz, necessary to update the bias and for calculating the new dEdy used in the next iterable layer
                if (layer.lastLayer):
                    dEdz = first_dEdz[i]
                else:
                    dEdz = neuron.dEdz(dEdy[i])

                # computates dEdw for given neuron
                dEdw = neuron.dEdw(dEdy[i], layer.inputs,
                                   layer.lastLayer, dEdz)
                neuron.updateWeights(learningRate, dEdw)
                neuron.updateBias(learningRate, dEdz)

                # dEdy of the next layer: sums up the right weight multiplying by the dEdz in the right place of the array
                for w in range(len(neuron.weights)):
                    new_dEdy[w] += neuron.weights[w] * dEdz
            # finally, with the new dEdy array constructed, we update it and  propagates it to next iterable layer
            dEdy = new_dEdy
