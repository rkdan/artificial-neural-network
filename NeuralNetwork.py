# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 14:44:17 2019

@author: Ryan
"""

import numpy as np
import scipy.special as special

class NeuralNet:
    
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learn_rate):
        # number of input, hidden, and output nodes
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        
        # learning rate
        self.learn_rate = learn_rate
        
        # weight matrices
        # weights are w_i_j from node i to node j
        self.w_ih = (np.random.rand(self.hidden_nodes, self.input_nodes)) - 0.5
        self.w_ho = (np.random.rand(self.output_nodes, self.hidden_nodes)) - 0.5
        
        # activation function
        self.act_func = lambda x: special.expit(x)
    
    def train(self, inputs, targets):
        """Trains the NN"""
        inputs = np.array(inputs, ndmin=2).T
        targets = np.array(targets, ndmin=2).T
        
        # signals into the hidden layer
        hidden_inputs = np.dot(self.w_ih, inputs)
        # signals out of the hidden layer
        hidden_outputs = self.act_func(hidden_inputs)
        
        # signals into the output layer
        final_inputs = np.dot(self.w_ho, hidden_outputs)
        # signals out of the output layer
        final_outputs = self.act_func(final_inputs)
                                      
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.w_ho.T, output_errors)
        
        self.w_ho += self.learn_rate * np.dot((output_errors * final_outputs * (1 - final_outputs)),
                                              np.transpose(hidden_outputs))
        
        self.w_ih += self.learn_rate * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                             np.transpose(inputs))
        
    
    def query(self, inputs):
        """Takes a list of inputs and returns the predicted final output."""
        inputs = np.array(inputs, ndmin=2).T
                                      
        # signals into the hidden layer
        hidden_inputs = np.dot(self.w_ih, inputs)
        # signals out of the hidden layer
        hidden_outputs = self.act_func(hidden_inputs)
        
        # signals into the output layer
        final_inputs = np.dot(self.w_ho, hidden_outputs)
        # signals out of the output layer
        final_outputs = self.act_func(final_inputs)
        
        return final_outputs
    

