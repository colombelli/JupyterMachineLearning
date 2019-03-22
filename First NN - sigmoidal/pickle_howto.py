import pickle
import os

# initialize random dict to save and easy print in order to test
person = {"name": "Felipe", "age": 21}

# creates a string variable responsible for indicating the name of the file
# the same name for writing and reading the file will be used
filename = "person.bin"  # the .bin extension save the file as a binary one

# save the data in a bin file
with (open(filename, 'wb')) as openfile:  # the with syntax guarantees that the file will be closed after being loaded to the memory
    pickle.dump(person, openfile)

if os.path.exists(filename):  # checks if the file name given actually exists, then opens it (for reading)
    with open(filename,'rb') as openfile:
        openedFile = pickle.load(openfile)

else:  # else, prints an error message
    print("error! file does not exists in this directory")


# comparing results (the two prints should output the same thing)
# and a person.bin file should be created
print(person)
print(openedFile)
