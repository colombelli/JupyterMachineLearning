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
    # builds a string responsible for logging into a text file reporting the time taken for train that amount of samples
    stringReport = str(samples) + " samples | time " + str(time_taken) + " minutes\n"

    with (open(filename, 'wb')) as openfile:  # saves the trained nn
        pickle.dump(nn, openfile)
    with (open("reports.txt", 'a')) as openfile:  # saves the report string without overwriting, that's way we open it with the append (a) mode
        openfile.write(stringReport)



# loads the dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Training a network with the following structure
# 784 neurons for input
# 85 neurons for the hidden layer
# 10 neurons for output (representing the numbers from 0 to 9)
numInputs = 784
neuronsEachLayer = [85, 10]
network = NeuralNetwork(numInputs, neuronsEachLayer)
learning_rate = 0.01
repeatTrainSamples = 1
MAX_TRAINING_SAMPLES = 50000 #len(y_train) * repeatTrainSamples  # samples: 240.000
sample_n = 0
startTime = time.time()


for i in range(repeatTrainSamples):  # iterates through the 60k train samples more than once

    for sample, label in zip(x_train, y_train):
        sample_n += 1

        if (sample_n % 1000) == 0:  # prints status of the training
            time_taken = (time.time() - startTime) / 60
            clear_output()
            print("%d samples out of %d" %(sample_n, MAX_TRAINING_SAMPLES))
            print("%.2f minutes used" %time_taken)

            # saves intermidiate trained networks
            if sample_n == 1000:
                save_nn("nn_1k.bin", network, time_taken, sample_n)
            elif sample_n == 5000:
                save_nn("nn_5k.bin", network, time_taken, sample_n)
            elif sample_n == 10000:
                save_nn("nn_10k.bin", network, time_taken, sample_n)
            elif sample_n == 50000:
                save_nn("nn_50k.bin", network, time_taken, sample_n)
                break

        sample = np.concatenate(sample)/255  # shapes the sample in the format of an array of 784 lenght also normalizing it
        sample = np.array(sample, dtype=np.float64)  # avoids the possible overflow about to come with the exp function

        # convert the output to one hot encoded
        expected_output = np.zeros(10)
        expected_output[label] = 1

        network_output = network.feedForward(sample)  # computes network output
        network.backpropagation(expected_output, network_output, learning_rate)

        '''
        Now we don't need this test because we are going to loop 4 times trough the entire training set
        # checks if it needs to continue the loop
        if sample_n == MAX_TRAINING_SAMPLES:
            time_taken = time.time() - startTime
            break
        '''

print("Time taken: ", time_taken)
print("\n\n")
