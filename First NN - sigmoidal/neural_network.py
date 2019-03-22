import numpy as np

class Neuron:
    def __init__(self, weights, bias):  # class constructor
        self.weights = weights
        self.bias = bias
        self.output = 0


    def activation_function(self, input):  # used to strech or contract outputs between some range
        input = np.float64(input)  # avoids the overflow
        self.output = 1.0 / (1.0 + np.exp(-(input)))  # the sigmoid activation function normally used
        return self.output


    def z(self, inputs):  # outputs the sum of every input times its respective weight, which are always 1 for this specific Task; and add a bias in the final result (0 in this particular Task)
        zOut = 0
        for i in range(len(inputs)):
            zOut += inputs[i] * self.weights[i]
        return zOut + self.bias


    def y(self, inputs):  # final output - neuron value when activated
        return self.activation_function(self.z(inputs))


    def dEdz(self, dEdy):
        # assuming that our activation function is a sigmoid one, dEdz = dEdy * dydz   -> dEdz = dEdy * y * (1 - y)
        return dEdy * self.output * (1 - self.output)


    def dEdw(self, dEdy, inputs):  # dEdw = dzdw * dydz * dEdy = dzdw * dEdz     , where dz(i)dw is simply the i input
        dEdz = self.dEdz(dEdy)
        dEdw = []
        for i, w in enumerate(self.weights):
            dEdw.append(inputs[i] * dEdz)
        return dEdw


    # we have to update the weights based on the learning rate and the derivatives
    def updateWeights(self, learningRate, dEdw):
        for i in range(len(self.weights)):
            self.weights[i] -= learningRate * dEdw[i]


    # the same goes to the bias
    def updateBias(self, learningRate, dEdz):
        self.bias -= learningRate * dEdz  # because dEdb = dzdz * dEdz = 1 * dEdz = dEdz



class DenseLayer:
    def __init__(self, num_of_inputs, num_of_neurons):
        self.num_of_inputs = num_of_inputs
        self.neurons = []


        for i in range(num_of_neurons):  # creating the neurons for the layer
            weights = np.random.uniform(-1,1,[num_of_inputs])  # randomizing "num_inputs" weights with a value between 0 and 1
            bias = np.random.uniform(0,1)  # randomizing a bias
            self.neurons.append(Neuron(weights, bias))  # appending the new neuron to the layer


    def feedForward(self, inputs):  # activates every neuron in the layer, outputting their respestive results
        self.inputs = inputs  # save the inputs as an attribute in order to use it in the dzdw calculus
        lisOut = []
        for neuron in self.neurons:
            lisOut.append(neuron.y(inputs))
        return lisOut



class NeuralNetwork:
    def __init__(self, num_of_inputs, num_of_neurons_at_each_layer):
        self.num_of_inputs = num_of_inputs
        self.layers = []

        # creates a dense layer for every int on the list num_of_neurons_at_each_layer
        num_of_inputs_next_neuron = num_of_inputs  # the first layer will have the given number of inputs for each of its neuron
        for num in num_of_neurons_at_each_layer:
            self.layers.append(DenseLayer(num_of_inputs_next_neuron, num))
            num_of_inputs_next_neuron = num  # the next layers will have the number of neurons of the previous layer as its number of inputs


    def feedForward(self, inputs):
        for layer in self.layers:  # keeps picking the outputs of every layer and passing them as inputs to the next layer
            inputs = layer.feedForward(inputs)
        # the outputs will be the final result of the inputs variable, after the end of the loop above
        outputs = inputs  # for code clarity, this variable is created before the method returns
        return outputs


    def derivative_of_the_error(self, value, result):
        # the derivative of the RSS with respect of each dimension  is: 2 * (dimensionResult - dimensionValue)
        # returns an array with the result of each value
        dEdy = []
        for i in range(len(value)):  # calculates each value of that derivative
            dEdy.append((result[i] - value[i]) * 2)
        return dEdy


    def backpropagation(self, value, result, learningRate):

        dEdy = self.derivative_of_the_error(value, result)  # calculates the first dEdy related to the final output

        flagFirstLayer = 1  # indicates that the dEdy is already calculated for that layer (the first starting at the end)
        for layer in reversed(self.layers):  # update the weights for each neuron in each layer (starting at the end)

            new_dEdy = np.zeros(len(layer.neurons[0].weights))

            for i, neuron in enumerate(layer.neurons):  # updates the neurons weights and biases in the layer

                dEdw = neuron.dEdw(dEdy[i], layer.inputs)  # computates dEdw for given neuron
                dEdz = neuron.dEdz(dEdy[i])  # necessary to update the bias and for calculating the new dEdy used in the next iterable layer
                neuron.updateWeights(learningRate, dEdw)
                neuron.updateBias(learningRate, dEdz)

                # sums up the right weight multiplying by the dEdz in the right place of the array
                for w in range(len(neuron.weights)):
                    new_dEdy[w] += neuron.weights[w] * dEdz

            # finally, with the new dEdy array constructed, we update it and  propagates it to next iterable layer
            dEdy = new_dEdy


def error_function(value, result):
# where the value represents what the network should output and result represents what it actually outputted
    RSS = 0
    for i in range(len(value)):
        RSS += (result[i]- value[i])**2
    return RSS
