import tensorflow as tf
import numpy as np
from datetime import datetime

class Logistic_Regression:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.W2 = tf.Variable(tf.random_normal)

        self.Z1 = tf.Variable(tf.random_normal([1, input_nodes]))
        self.A1 = tf.Variable(tf.random_normal([1, input_nodes]))

        self.Z2 = tf.Variable(tf.random_normal([1, hidden_nodes]))
        self.A2 = tf.Variable(tf.random_normla([1, hidden_nodes]))

        self.Z3 = tf.Variable(tf.random_normal([1, output_nodes]))
        self.A3 = tf.Variable(tf.random_normal([1, output_nodes]))