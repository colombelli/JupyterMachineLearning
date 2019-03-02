from neural_network import *
import pickle
import os
import tensorflow as tf


# function that clears the console output
def clear_output():
    if os.name == 'nt':  # for windows
        _ = os.system('cls')
    else: # for mac and linux
        _ = os.system('clear')


# function that prints the parcial results on console
def print_results(sample_num, label, nn_guess, matches, loss):
    clear_output()
    print("Sample %d | Label = %d | Output = %d | %d matches" % (sample_num, label, nn_guess, matches))
    print("Loss: ", loss)
    print("Hit rate: %.2f%%" % (matches / sample_num * 100))
    print("\n")


filename = input("File name: ")  # asks for user to input the .bin file with the containing the trained network
while (not os.path.exists(filename)):  # keeps asking for the file name if the given one doesn't exists
    filename = input("Please, give an existing file name with its extension: ")

with open(filename,'rb') as openfile:  # after validating that the file exists in the directory, opens it in read mode
    network = pickle.load(openfile)


# loads the dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

sample_num = 0
matches = 0
total_loss = 0
MAX_TEST_SAMPLES = 10000  # all test samples in the dataset
for sample, label in zip(x_test, y_test):

    sample_num += 1
    sample = np.concatenate(sample)

    # convert the output to one hot encoded
    expected_output = np.zeros(10)
    expected_output[label] = 1

    network_output = network.feedForward(sample)  # computes network output
    loss = error_function(expected_output, network_output)  # computes loss
    total_loss += loss  # sums up the loss for calculating the final average loss

    nn_guess = np.argmax(network_output)  # gets the index of the highest value in the array
    if nn_guess == label:  # check if the network got a right guess
        matches += 1

    if (sample_num % 100) == 0:  # prints parcial results from 100 to 100 tested samples
        print_results(sample_num, label, nn_guess, matches, loss)

    if sample_num == MAX_TEST_SAMPLES:
      break


print("\nAverage loss: %.2f" %(total_loss / sample_num))
print("Final hit rate: %.2f%%" % (matches / sample_num * 100))
