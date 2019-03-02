from neural_network import *
import pickle
import os
import tensorflow as tf
import time


# function that clears the console output
def clear_output():
    if os.name == 'nt':  # for windows
        _ = os.system('cls')
    else: # for mac and linux
        _ = os.system('clear')


# function that saves a given neural network object into a binary file
def save_nn(filename, nn, time_taken, samples):

    # builds a string responsible for logs into a text file reporting the time taken for train that amount of samples
    stringReport = str(samples) + " samples | time " + str(time_taken) + " minutes\n"

    with (open(filename, 'wb')) as openfile:  # saves the trained neural network
        pickle.dump(nn, openfile)
    with (open("reports.txt", 'a')) as openfile:  # saves the report string without overwriting, that's way we open it with the append (a) mode
        openfile.write(stringReport)


# loads the dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Training a network with the following structure
# 784 neurons for input
# 50 neurons for the first hidden layer
# 30 neurons for the second hideen layer
# 10 neurons for output (representing the numbers from 0 to 10)
numInputs = 784
neuronsEachLayer = [50, 30, 10]
network = NeuralNetwork(numInputs, neuronsEachLayer)
learning_rate = 0.001
repeatTrainSamples = 14
MAX_TRAINING_SAMPLES = len(y_train) * repeatTrainSamples  # samples: 840.000

sample_n = 0
matches = 0
total_loss = 0
startTime = time.time()

for i in range(repeatTrainSamples):  # iterates through the 60k train samples more than once

    for sample, label in zip(x_train, y_train):
        sample_n += 1

        if (sample_n % 500) == 0:
            # prints status of the training
            time_taken = (time.time() - startTime) / 60
            clear_output()
            print("%d samples out of %d" %(sample_n, MAX_TRAINING_SAMPLES))
            print("%.2f minutes used" %time_taken)  # prints how much minutes were already used to train the nn

            # saves intermidiate trained networks
            if sample_n == 60000:
                save_nn("nn_60k.bin", network, time_taken, sample_n)

            elif sample_n == 100000:
                save_nn("nn_100k.bin", network, time_taken, sample_n)

            elif sample_n == 200000:
                save_nn("nn_200k.bin", network, time_taken, sample_n)

            elif sample_n == 400000:
                save_nn("nn_400k.bin", network, time_taken, sample_n)

            elif sample_n == 600000:
                save_nn("nn_600k.bin", network, time_taken, sample_n)

            elif sample_n == 840000:
                save_nn("nn_840k.bin", network, time_taken, sample_n)

        sample = np.concatenate(sample)  # shapes the sample in the format of an array of 784 lenght
        sample = np.array(sample, dtype=np.float64)  # avoids the possible overflow about to come with the exp function

        # convert the output to one hot encoded
        expected_output = np.zeros(10)
        expected_output[label] = 1

        network_output = network.feedForward(sample)  # computes network output
        loss = error_function(expected_output, network_output)  # computes loss

        nn_guess = np.argmax(network_output)  # gets the index of the highest value in the array
        if nn_guess == label:  # check if the network got a right
            matches += 1

        # update weights and biases
        network.backpropagation(expected_output, network_output, learning_rate)

        # checks if it needs to continue the loop
        # now we are controlling it by the number of loop iterations, so we don't need this if statement
        #if sample_n == MAX_TRAINING_SAMPLES:
        #    time_taken = time.time() - startTime
        #    break

print("Time taken: ", time_taken)
print("\n\n")
