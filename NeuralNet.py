import numpy as np
class NeuralNetwork():
    def __init__(self):
        np.random.seed(1)

        self.synaptic_weights = 2*np.random.random((3,1)) - 1

    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))

    def sigmoid_derivative(self,x):
        return x * (1 - x)

    def train(self,training_set_inputs,training_set_outputs,number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            output = self.predict(training_set_inputs)
            error = training_set_outputs - output
            adjustment = np.dot(training_set_inputs.T,error*self.sigmoid_derivative(output))
            self.synaptic_weights += adjustment

    def predict(self,inputs):
        return self.sigmoid(np.dot(inputs,self.synaptic_weights))



if __name__ == "__main__":

    #initialize a single neuron neural network.
    neural_network = NeuralNetwork()

    print("Random starting synaptic weights:")
    print(neural_network.synaptic_weights)

    # The training set. We have 4 examples,each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
    training_set_outputs = np.array([[0,1,1,0]]).T

    #Train the neural network using a training set.
    #Do it 10,000 times and make small adjustments each time
    neural_network.train(training_set_inputs,training_set_outputs,10000)
    print("New synaptic weights after training:")
    print(neural_network.synaptic_weights)

    #Test the neural network with new situation.
    print("Considering new situation [1,0,0] -> ?:")
    print(neural_network.predict(np.array([1,0,0])))

import numpy as np
inputs = np.array([1,0,0])
print("inputs",inputs)   #inputs (3,)
synaptic_weights = 2*np.random.random((3,1)) - 1 #synaptic_weights (3, 1)
print("synaptic_weights",synaptic_weights)
results = np.dot(inputs,synaptic_weights)
print("results",results)


#import numpy as np
#x = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
#print(x.shape)
#y = np.array([[0,1,1,0]]).T
#print(y.shape)
#syn0 = 2*np.random.random((3,4)) - 1
#print("syn0=",syn0)
#syn1 = 2*np.random.random((4,1)) - 1
#print("syn1=",syn1)
#for i in range(60000):
    #l1 = 1/(1 + np.exp(-(np.dot(x,syn0))))
    #print(l1)
    #l2 = 1/(1 + np.exp(-(np.dot(l1,syn1))))
    #print(l2)
    #l2_delta = (y - l2)*(l2*(1-l2))
    #l1_delta = l2_delta.dot(syn1.T)*(l1*(1-l1))
    #syn1 += l1.T.dot(l2_delta)
    #syn0 += x.T.dot(l1_delta)
#print("syn0=",syn0)
#print("syn1=",syn1)
