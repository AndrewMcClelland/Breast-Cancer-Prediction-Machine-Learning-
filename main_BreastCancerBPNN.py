# Andrew McClelland - November 13th, 2016
# CMPE 452 - Project: Backpropagation Neural Network to Classify Breast Cancer Diagnoses
#
# This program uses a backpropagation neural network to learn and classify whether a given patient's breast cancer is
# malignant or benign. It uses 30 attributes, each representing a different characteristic of the breast cancer, as well
# as an identifier to confirm the accuracy of the network's predictions (supervised learning).

# The program accomplishes this in the following (briefly summarized) steps (see report for details):
#
# 1. Parse the data into a training dataset (70%) and a testing dataset (30%)
# 2. Gather user inputs for number of hidden nodes, data points to be used, and epochs
# 3. Create and initialize the neural network
# 4. For each training data point in training data, predict the output, backpropagate to adjust the weights and calculate mean summed error for this training data point
# 5. Test the final weights (from the training portion) by predicting the classification of each data point in the testing data set, and calculate classification accuracy by comparing predictions to actual classification
# 6. Output the original test data points and the algorithm
# 7. End

# Importing relavant libraries
import random
import numpy as np
import sys
import math
import time


# Class that holds all relevant methods for creating, training, and testing the neural network
class backPropogation_NN:

    # Initialization function that initializes weights, biases, activations for each node / layer
    def __init__(self, numInputNodes, numHiddenNodes, numOutputNodes):

        # Initializing number of input, hidden, and output nodes
        self.numInputNodes = numInputNodes
        self.numHiddenNodes = numHiddenNodes
        self.numOutputNodes = numOutputNodes

        # Initialize weights for input -> hidden layer and hidden -> output layer. dimensions = (first layer, second layer)
        self.inputHiddenWeight = np.zeros((self.numInputNodes, self.numHiddenNodes))
        self.hiddenOutputWeight = np.zeros((self.numHiddenNodes, self.numOutputNodes))

        # Randomly assign weight values to both matrices (random float from -0.5 to 0.5)
        for iNode in range(self.numInputNodes):
            for hNode in range(self.numHiddenNodes):
                self.inputHiddenWeight[iNode][hNode] = random.uniform(-0.5, 0.5)
        for hNode in range(self.numHiddenNodes):
            for oNode in range(self.numOutputNodes):
                self.hiddenOutputWeight[hNode][oNode] = random.uniform(-0.5, 0.5)

        # Randomly assign bias weight (float from -0.5 to 0.5) for hidden and output layers (the bias within a layer is same for all nodes in that layer)
        self.hiddenBiasWeight = random.uniform(-0.5, 0.5)
        self.outputBiasWeight = random.uniform(-0.5, 0.5)

        # Initializing activation for each function as an array (each node is an element in an array of that layer)
        self.inputActivation = [1.0] * self.numInputNodes
        self.hiddenActivation = [1.0] * self.numHiddenNodes
        self.outputActivation = [1.0] * self.numOutputNodes

        # Previous weight change for momentum
        self.momentumInput = np.zeros((self.numInputNodes, self.numHiddenNodes))
        self.momentumOutput = np.zeros((self.numHiddenNodes, self.numOutputNodes))

        # Lists that hold the original data input (from test file) and the algorithm's corresponding prediction
        self.output_original_dataPoint = []
        self.output_predicted_dataPoint = []
        self.output_actual_classification = []
        self.num_class_errors = 0


    # Forward pass of the algorithm - predicting output of given attributes
    # 'attributes' is a list of the 30 attributes of the current patient (instance), 'actual_classification' is the
    #  actual classification of these attributes (benign (0) or malignant (1))
    def feed_forward(self, attributes, actual_classification):

        # If we entered a different number of inputs than the number of attributes the dataset contains, create error
        if len(attributes) != num_input_neurons:
            sys.exit("Invalid number of input neurons.")

        # Setting the inputActivation equal to the attributes of the current training digit
        for i in range(self.numInputNodes):
            self.inputActivation[i] = attributes[i]

        # Calculating the activation of each hidden layer node ( sigmoid( weight*attribute + bias ) )
        for j in range(self.numHiddenNodes):
            hNode_input = 0
            for i in range(self.numInputNodes):  # sum the inputs of the input nodes to the current hidden node
                hNode_input = hNode_input + (self.inputHiddenWeight[i][j] * self.inputActivation[i])
            hNode_input = hNode_input + self.hiddenBiasWeight  # add the bias
            self.hiddenActivation[j] = sigmoid_function(hNode_input)  # calculate the sigmoid of the total input to the hidden node and append to the list

        # Calculating the activation of each output layer node ( sigmoid( weight*attribute + bias ) )
        for j in range(self.numOutputNodes):
            oNode_input = 0
            for i in range(self.numHiddenNodes):  # sum the inputs of the hidden nodes to the current output node
                oNode_input = oNode_input + (self.hiddenOutputWeight[i][j] * self.hiddenActivation[i])
            oNode_input = oNode_input + self.outputBiasWeight  # add the bias
            self.outputActivation[j] = sigmoid_function(oNode_input)  # calculate the sigmoid of the total input to the output node and append to the list

        self.desired_output = []  # list for what our desired outputActivation should be

        # create 2 element list that indicates the actual classification of the current attribute inputs (1)... other = 0
        for i in range(self.numOutputNodes):
            if i == actual_classification:  # desired output node should be 1 if it's the actual_classification
                self.desired_output.append(1)
            else:  # all other nodes should be 0
                self.desired_output.append(0)

        # In the case where the output nodes activations are equal, randomly choose 1 so we don't get a bias of always
        # choosing 0 (max function will always choose first index if they are identical values)
        if self.outputActivation[0] == self.outputActivation[1]:
            return random.randint(0,1)
        else:  # if they are different output activations
            return self.outputActivation.index(max(self.outputActivation))  # return the index of the predicted digit


    # Backpropagate to correct weights
    # 'actual_classification' is the correct identification of the current instance
    def backpropagation(self, actual_classification, learning_rate, momentum_factor):

        # Calculating mean squared sum error of the network's classification
        mean_error = 0  # initializing mean squared error calculation to 0
        for i in range(len(self.desired_output)):  # run through each of the 2 output classifications
            mean_error = mean_error + 0.5 * (((self.desired_output[i] - self.outputActivation[i])) ** 2)  # calculate mean squared sum error of this data input

        # Calculate error from output nodes
        output_delta = []  # list of delta errors for output nodes
        for i in range(self.numOutputNodes):  # for each output node
            output_error = self.desired_output[i] - self.outputActivation[i]  # calculate error: desired output (1 or 0) - output node activation
            output_delta.append(sigmoid_deriv_function(self.outputActivation[i]) * output_error)  # append the change to the ouput node to the list

        # Calculate error from hidden nodes
        hidden_delta = []  # list of delta errors for input nodes
        for i in range(self.numHiddenNodes):  # for each hidden node
            hidden_error = 0
            for j in range(self.numOutputNodes):  # for each output node that connects to the current hidden node
                hidden_error = hidden_error + output_delta[j] * self.hiddenOutputWeight[i][j]  # calculate sum of hidden error
            hidden_delta.append(sigmoid_deriv_function(self.hiddenActivation[i]) * hidden_error)  # append this error to the list

        # Update weights from hidden layer --> output layer
        for i in range(self.numHiddenNodes):  # for each hidden node
            for j in range(self.numOutputNodes):  # for each output node attached to the hidden node
                momentum_change = output_delta[j] * self.hiddenActivation[i]  # calculate the momentum change
                self.hiddenOutputWeight[i][j] = self.hiddenOutputWeight[i][j] + learning_rate * momentum_change + momentum_factor * self.momentumOutput[i][j]  # update weight
                self.momentumOutput[i][j] = momentum_change  # store change in momentum in matrix

        # Update weights from input layer --> hidden layer
        for i in range(self.numInputNodes):  # for each input node
            for j in range(self.numHiddenNodes):  # for each hidden node attached to the input node
                momentum_change = hidden_delta[j] * self.inputActivation[i]  # calculate the momentum change
                self.inputHiddenWeight[i][j] = self.inputHiddenWeight[i][j] + learning_rate * momentum_change + momentum_factor * self.momentumInput[i][j]  # update weight
                self.momentumInput[i][j] = momentum_change  # store change in momentum in matrix

        return mean_error  # return mean squared sum error for this data point (calculated at start of this method)


    # Trains the neural network using the training set
    # 'attributes' is a list of the current inputs attributes, 'actual_classification' is the correct classification of the current instance
    def train_alg(self, attributes, actual_classification, learning_rate, momentum_factor, max_dataset_iterations):

        dataset_iterations = 0  # initialize the number of times we've run through the dataset to 1

        # Loop through dataset until we've reached the max_dataset_iterations (user entered # epochs)
        while dataset_iterations < max_dataset_iterations:

            total_error = 0  # mean square error of classifying dataset

            # Running through the entire training dataset
            for each_input, each_actual_classification in zip(attributes, actual_classification):
                self.feed_forward(each_input, each_actual_classification)  # forward propagate through the network and predict outputs
                total_error = total_error + self.backpropagation(each_actual_classification, learning_rate, momentum_factor)  # compute the total error for this epoch
            dataset_iterations += 1  # increment the dataset iteration counter
            print("Dataset iteration:", dataset_iterations) #, "\tMean Square Error:", total_error, "\n")

            # Error breaking condition
            if total_error < 0.05:
                break

        print("Final mean square error:", total_error, "\n")


    # Tests the trained neural network using the testing set
    # 'attributes' is a list of the current inputs attributes, 'actual_classification' is the correct classification of the current instance
    def test_alg(self, attributes, actual_classification):
        # Run through the entire testing data set
        for each_input, each_actual_classification in zip(attributes, actual_classification):
            predicted_digit = self.feed_forward(each_input, each_actual_classification)  # passing the datapoint to the feedforward method to predict classification
            self.output_original_dataPoint.append(each_input)  # appending the original instance to the list (for txt file creation later)
            self.output_actual_classification.append(each_actual_classification)  # appending the actual classification of this instance
            self.output_predicted_dataPoint.append(predicted_digit)  # appending the prediction to the list (for txt file creation later)
            print('Predicted =', predicted_digit, '\tActual =', each_actual_classification)  # print to terminal
            if predicted_digit != each_actual_classification:  # if we classified incorrectly, increment error counter
                self.num_class_errors += 1

        print("\nNumber of classification errors:", self.num_class_errors, "\nNumber of points:", len(attributes))
        print("\nClassification accuracy:", 100*(len(attributes) - self.num_class_errors)/len(attributes), "%")


# Function reads in the training data from the provided text file
def parse_data():

    data_file = "WisconsinDataset_Diagnosis.txt"  # train data file name

    train_data = []  # list that holds all lines of the train data
    test_data = []  # list that holds all lines of the train data
    train_digit_identifier = []  # holds the classification for each instance of training dataset (benign = 0, malignant = 1)
    test_digit_identifier = []   # holds the classification for each instance of testing dataset (benign = 0, malignant = 1)

    train_data_length = 399  # use 70% of dataset as training data (569 instances total)
    test_data_length = 170  # use 30% of dataset as testing data (569 instances total)

    line_counter = 1  # counter to separate training and testing data set

    # Put training data into a list
    with open(data_file, 'r') as file:
        for line in file:
            line = [value for value in line.replace('\n', '').split(',')]  # separate data values and convert them to list of ints
            if line[1] == "B":  # replace 'benign' with 0
                digit_identifier = 0
            elif line[1] == "M":  # replace 'malignant' with 1
                digit_identifier = 1
            del line[0]  # remove the first useless instance identifier (ID #)
            del line[0]  # remove the identifier from the set
            line = [float(value) for value in line]  # now left with only the 30 useful attributes
            if line_counter <= train_data_length:  # if we're still appending to training data
                train_data.append(line)
                train_digit_identifier.append(digit_identifier)
            else:  # append to test data
                test_data.append(line)
                test_digit_identifier.append(digit_identifier)

            line_counter += 1

    return train_data, test_data, train_digit_identifier, test_digit_identifier


# Sigmoidal output function
def sigmoid_function(net):
    if net < -500:
        net = -500
    return 1.0 / (1.0 + (math.exp(-net)))


# Derivative of sigmoidal output function
def sigmoid_deriv_function(net):
    func_output = sigmoid_function(net)
    return (func_output)*(1 - func_output)


# function that creates the output text file of the testing data's classification
def text_output():
    # Creating the output text file
    output_file = open('breastCancer_backPropagation_output.txt', 'w')

    # Writing header to the output file
    output_file.write("Output for the back propagation neural network classification on the provided test dataset.\n\n"
                      "Each original data element will be listed on the left, with the algorithm's prediction listed on the right.\n\n")

    # Writing number of errors and classification accuracy
    num_error_string = "Number of points: 170\nNumber of classification errors: " + str(neuralNet.num_class_errors) + "\n"
    class_accuracy_string = "Classification accuracy = " + str(100*((170 - neuralNet.num_class_errors) / 170)) + "%\n\n"
    output_file.write(num_error_string)
    output_file.write(class_accuracy_string)
    output_file.write("Actual Classification\t\t\t\tPrediction\t\t\t\tCorrect?\n")

    # Running through each testing point in test data (in order) and printing the point's actual classification and the predicted classification
    for each_actual_classification, each_prediction in zip(neuralNet.output_actual_classification, neuralNet.output_predicted_dataPoint):
        each_actual_classification_string = "\t\t" + str(each_actual_classification) + "\t\t\t\t\t\t\t\t"
        each_prediction_string = str(each_prediction) + "\t\t\t\t\t\t"
        if each_actual_classification == each_prediction:
            correct_string = "YES\n"
        else:
            correct_string = "NO\n"
        output_file.write(each_actual_classification_string)
        output_file.write(each_prediction_string)
        output_file.write(correct_string)

    # Close the file
    output_file.close()


# main function
if __name__ == '__main__':
    print("Parsing the training and testing data from the dataset file...")
    train_data, test_data, train_digit_identifier, test_digit_identifier = parse_data()  # parse the training data
    print("Done!\n")

    learning_rate = 1  # initialize learning rate to 1
    momentum_factor = 0.01  # initialize momentum rate to 0.1

    # Use 3 layers of neurons
    num_input_neurons = 30  # 32 neurons in input layer (1 neuron for each attribute; only 3-32 are useful)

    while True:  # user input for number of hidden neurons
        num_hidden_neurons = int(input("Please enter the desired number of hidden neurons (recommended is 10):"))
        if (num_hidden_neurons > 0):
            break
        print("You entered incorrectly. Try again.\n")

    print("Hidden neurons:", num_hidden_neurons, "\n")

    num_output_neurons = 2  # 2 neurons in output layer (malignant or benign diagnosis) --> output neuron with highest activation level is chosen classification

    print("Creating neural network...")

    # Instantiate object 'neuralNet' of class 'backPropogation_NN'
    neuralNet = backPropogation_NN(num_input_neurons, num_hidden_neurons, num_output_neurons)  # creating a NN 'neuralNet' with 3 layers of nodes

    print("Done!\n")

    # User input for entering number of training data points to used
    while True:
        number_data_points = input("Please enter the number of desired datapoints to me used in training the network...enter 'All' to use all training points):")
        if number_data_points.upper() == 'ALL':
            number_data_points = 399  # use all the data points to train
            break
        elif int(number_data_points) > 0:
            number_data_points = int(number_data_points)
            break
        print("You entered incorrectly. Try again.\n")

    print("Number of training data points to be used:", number_data_points, "\n")

    # User input for entering number of training epochs to be used
    while True:
        number_data_iterations = int(input("Please enter the number of desired training epochs (recommended is max 50 (due to time of execution)):"))
        if (number_data_iterations > 0):
            break
        print("You entered incorrectly. Try again.\n")

    print("Number of iterations through the training data set:", number_data_iterations, "\n")

    # Outputting to terminal
    print("Beginning to train in:\n3")
    time.sleep(1)
    print("2")
    time.sleep(1)
    print("1")
    time.sleep(1)
    print("Network training has begun!\n")

    # Train the neural network
    neuralNet.train_alg(train_data[0:number_data_points], train_digit_identifier[0:number_data_points], learning_rate, momentum_factor, number_data_iterations)

    # Outputting to terminal
    print("Algorithm training has finished. Classification of testing data will begin in:\n3")
    time.sleep(1)
    print("2")
    time.sleep(1)
    print("1")
    time.sleep(1)

    # Test the neural network using the final trained weights
    neuralNet.test_alg(test_data, test_digit_identifier)

    print("Test data classification has completed.\n")

    # User input for creating a text file
    while True:
        user_file_output = input("Would you like to create a text file with the classification outputs of the testing data?\nEnter 'Yes' or 'No':")
        if user_file_output.upper() == 'YES':
            create_file = 1
            break
        elif user_file_output.upper() == 'NO':
            create_file = 0
            break
        else:
            print("You entered incorrectly. Please try again.\n")

    # If user wants to create the output text file, call the text_output function
    if create_file == 1:
        print("Creating txt file output...")
        text_output()
        print("Done! 'breastCancer_backPropagation_output.txt' was created.\n")

    # Finished
    input("Please hit the 'return' key when you would like to close this terminal.")